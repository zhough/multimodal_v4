from model import VLMModel
from transformers import AutoConfig,AutoImageProcessor,AutoTokenizer
from vision_config import VisionConfig
from tools import process_conversation
from PIL import Image
import torch
vconfig = VisionConfig()
llmconfig = AutoConfig.from_pretrained(vconfig.llm)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VLMModel(llmconfig)
state_dict = torch.load('./model.pth', map_location=device)

# 移除所有权重键的 module. 前缀
new_state_dict = {}
for key, value in state_dict.items():
    # 去掉键开头的 module.（若存在）
    if key.startswith('module.'):
        new_key = key[len('module.'):]  # 从第 7 个字符开始截取（module. 共 7 个字符）
    else:
        new_key = key  # 若没有前缀，直接保留原键
    new_state_dict[new_key] = value
# 2.3 加载处理后的权重到模型
model.load_state_dict(new_state_dict)
print('成功加载模型')
tokenizer = AutoTokenizer.from_pretrained(vconfig.llm)
image_processor = AutoImageProcessor.from_pretrained(vconfig.model_name)

# 1. 准备文本提示
prompt = "describe this picture."
# Qwen 系列通常使用特定的聊天模板，最好用 tokenizer.apply_chat_template
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 2. 准备图像
image = './men.jpg'
image = Image.open(image).convert('RGB')

# 3. 使用 processor 和 tokenizer 处理输入
# 处理图像
image_inputs = image_processor(images=image, return_tensors="pt").to(device)
# 处理文本
text_inputs = tokenizer(text, return_tensors="pt").to(device)
#pixel_values = torch.zeros_like(image_inputs['pixel_values'],device=device)
# 4. 合并成一个字典，这将作为 generate() 的输入
# 关键：确保键名（如 'pixel_values'）与你 forward 方法中接收的参数名一致
inputs = {**text_inputs, **image_inputs}
#inputs = {**text_inputs,'pixel_values':pixel_values}
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,  # 控制生成的最大长度
        temperature=0.7,     # 控制生成的随机性
        do_sample=True,      # 启用采样，如果为 False，则使用贪心解码
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# 解码生成的 tokens
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

print("生成的回答:")
print(generated_text)
