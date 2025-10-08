# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
# from transformers import CLIPModel
from transformers import CLIPVisionModel
# from transformers.models.vit import modeling_vit
# MODEL_NAME = "Qwen/Qwen3-0.6B" 

# print(f"正在从 {MODEL_NAME} 加载预训练模型...")
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     dtype="auto",  # 自动选择合适的数据类型（如 bfloat16）
# )
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# print("模型加载成功！")

# # --- 2. 打印参数名 ---

# print("打印所有可训练参数的名称和形状")
# print("-" * 50)
# # `model.named_parameters()` 返回一个迭代器，每次迭代产生一个 (name, parameter) 元组。
# # `parameter` 是一个 torch.Tensor 对象。
# count = 0
# for name, param in model.named_parameters():
#     # 打印参数名和其形状
#     print(f"名称: {name:<80} | 形状: {param.shape}")

#=============================================================================
# # 1. 导入需要的库
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# from PIL import Image
# import requests
# import torch

# # 2. 定义模型名称
# # 这个名称会告诉 transformers 库去下载对应的权重和配置
# model_name = "google/vit-base-patch16-224"

# # 3. 加载处理器 (Processor) 和模型
# # 处理器负责图像预处理 (如 resize, normalize)
# # AutoModelForImageClassification 会自动加载带有分类头的模型
# processor = AutoImageProcessor.from_pretrained(model_name)
# model = AutoModelForImageClassification.from_pretrained(model_name)

# # 4. 准备一张图片
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
# #image.save('cat.jpg')
# # 5. 预处理图像
# # return_tensors="pt" 表示返回 PyTorch Tensor
# inputs = processor(images=image, return_tensors="pt")
# print(f'input :{inputs}')   
# # 6. 进行推理
# # 使用 torch.no_grad() 提高效率
# with torch.no_grad():
#     outputs = model(**inputs,output_hidden_states=True)

# # 7. 解析结果
# # logits 是模型输出的原始分数
# logits = outputs.logits

# # 将 logits 转换为概率
# probabilities = torch.nn.functional.softmax(logits, dim=-1)

# # 获取概率最高的前 5 个类别
# top5_prob, top5_cat_id = torch.topk(probabilities, 5)

# # 将类别 ID 转换为标签名称
# top5_labels = [model.config.id2label[idx.item()] for idx in top5_cat_id[0]]

# # 打印结果
# print("--- 分类预测结果 ---")
# for label, prob in zip(top5_labels, top5_prob[0]):
#     print(f"类别: {label:<25} | 置信度: {prob.item():.2%}")

# # 这就是你想要的“所有维度”的输出
# hidden_states = outputs.hidden_states[-3:]
# hidden_states = torch.cat(hidden_states,dim=2)
# print(f"Last hidden state: {hidden_states}")
# print(f'hidden_states_shape:{hidden_states.shape}')
#=======================================================================

from model import VLMModel
from transformers import AutoTokenizer,AutoConfig,AutoModelForImageClassification
from vision_config import VisionConfig
vconfig = VisionConfig()

config = AutoConfig.from_pretrained(vconfig.llm)    
vit_model = AutoModelForImageClassification.from_pretrained(vconfig.model_name)
model = VLMModel(config,vit_model=vit_model)

for name,param in model.named_parameters():
         print(f"名称: {name:<80} | 形状: {param.shape}")

print('--------------------------------------------')
total_num_params = sum(p.numel() for name, p in model.named_parameters())
print(f"\n参数总数: {total_num_params:,}")

# for layer in model.model.layers:
#     for name,param in layer.named_parameters():
#         print(name)

# import pandas as pd

# file = pd.read_csv('results.csv',sep='|')
# print(file.columns)