from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3Model,
    Qwen3ForCausalLM,
    Qwen3PreTrainedModel,
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
    Qwen3RMSNorm

)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache,DynamicCache
from transformers.utils import TransformersKwargs
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.generation import GenerationMixin
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch 
from torch import nn
from torch.nn import functional as F
from typing import Optional,Union
from vision_config import VisionConfig
import numpy as np

vconfig = VisionConfig()
class CrossAttention(nn.Module):
    def __init__(self,v_hidden_size,l_hidden_size,num_heads=vconfig.num_heads):
        super().__init__()
        self.v_hidden_size = v_hidden_size
        self.l_hidden_size = l_hidden_size
        self.hidden_size = self.l_hidden_size   #中间层的维度和llm的隐藏层维度一致
        self.num_heads = num_heads
        self.head_dim = self.hidden_size//self.num_heads

        #线性映射层
        self.q_proj = nn.Linear(self.l_hidden_size,self.hidden_size)
        self.k_proj = nn.Linear(self.v_hidden_size,self.hidden_size)
        self.v_proj = nn.Linear(self.v_hidden_size,self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size,self.l_hidden_size)

    def forward(self,v_hidden_states,l_hidden_states):
        batch_size,l_seq_len,_ = l_hidden_states.shape  
        _,v_seq_len,_ = v_hidden_states.shape   
        q = self.q_proj(l_hidden_states)
        k = self.k_proj(v_hidden_states)
        v = self.v_proj(v_hidden_states)
        #转为多头
        q = q.reshape(batch_size,l_seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k = k.reshape(batch_size,v_seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v = v.reshape(batch_size,v_seq_len,self.num_heads,self.head_dim).transpose(1,2)

        # q = q.reshape(-1,l_seq_len,self.head_dim)
        # k = k.reshape(-1,v_seq_len,self.head_dim)
        # v = v.reshape(-1,v_seq_len,self.head_dim)
        scale_factor = torch.sqrt(torch.tensor(self.hidden_size, dtype=q.dtype, device=q.device))
        attn_weight = torch.matmul(q,k.transpose(-1,-2))/scale_factor
        attn_weight = F.softmax(attn_weight,dim=-1)

        attn_output = torch.matmul(attn_weight,v)
        attn_output = attn_output.transpose(1,2).reshape(batch_size,l_seq_len,self.hidden_size)
        attn_output = self.out_proj(attn_output)
        return attn_output  

class CrossDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.v_hidden_size = vconfig.v_hidden_size
        self.cross_attn = CrossAttention(v_hidden_size=self.v_hidden_size,l_hidden_size=self.hidden_size)
        self.cross_attention_layernorm =  Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    def forward(
        self,
        hidden_states: torch.Tensor,
        v_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        if v_hidden_states is not None:
            #Cross Attention
            residual = hidden_states
            hidden_states = self.cross_attention_layernorm(hidden_states)
            hidden_states = self.cross_attn(v_hidden_states,hidden_states)
            hidden_states = residual + hidden_states    

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    

class MyQwen3Model(Qwen3Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [CrossDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        v_hidden_states: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                v_hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class VLMModel(Qwen3ForCausalLM):
    # _tied_weights_keys = ["lm_head.weight"]
    # _tp_plan = {"lm_head": "colwise_rep"}
    # _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = MyQwen3Model(config)
        self.vit_model = AutoModelForImageClassification.from_pretrained(vconfig.model_name)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        with torch.no_grad():
            v_output = self.vit_model(pixel_values,output_hidden_states=True)
            v_hidden_states = v_output.hidden_states[-1]
            
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            v_hidden_states=v_hidden_states,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


