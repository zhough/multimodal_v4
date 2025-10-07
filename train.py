from model import VLMModel
from transformers import AutoConfig,AutoModelForCausalLM
from vision_config import VisionConfig
import torch
from torch.utils.data import Dataset,DataLoader
vconfig = VisionConfig()
from transformers import AutoImageProcessor,AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms
import json
import swanlab
from tools import process_conversation
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torch.optim as optim
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import argparse

# 初始化分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
# 清理分布式环境
def cleanup():
    dist.destroy_process_group()



class Config():
    def __init__(self):
        self.epochs = 1
        self.batch_size = 2
        self.learning_rate = 1e-5
        self.min_learning_rate = 1e-7
        self.weight_decay = 1e-4
        self.step = 0
        self.layers_to_unfreeze = 2
        #config.total_steps = 4000
        # 分布式训练参数
        self.world_size = torch.cuda.device_count()
        self.dist_url = "env://"
        self.local_rank = -1
        
        self.swanlab_project_name = 'multimodal_v4'
        self.image_dir = '/kaggle/input/coco-2017-dataset/coco2017/train2017/'
        self.train_json_file = '/kaggle/input/multimodal-coco/coco.json'
        self.val_json_file = '/kaggle/working/multimodal/data_val.json'  
        self.save_model = './output/model.pth'
        self.best_model = './output/best_model.pth'
        self.trained_model = '/kaggle/input/vlm/transformers/default/1/model.pth'
config = Config()


class VLMDataset(Dataset):
    def __init__(self, json_file_path, tokenizer, processor):
        # 1. 原始数据加载
        raw_data = []
        with open(json_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.tokenizer = tokenizer
        self.processor = processor
        
        # 2. 数据过滤：只保留图像文件真实存在的样本
        self.data = []
        invalid_count = 0
        
        print(f"开始加载数据集，原始样本数: {len(raw_data)}")
        
        for item in raw_data:
            # 检查是否存在 "image" 字段
            if "image" not in item or not item["image"]:
                invalid_count += 1
                continue # 跳过没有 "image" 字段的样本
            
            image_name = item["image"]
            image_path = os.path.join(config.image_dir, image_name)
            
            # 检查文件是否存在
            if os.path.exists(image_path):
                self.data.append(item)
            else:
                invalid_count += 1
             
        print(f"数据集加载完成！有效样本数: {len(self.data)}, 已跳过无效样本数: {invalid_count}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 调用上面定义的处理函数
        processed_item = process_conversation(item["conversations"],tokenizer=self.tokenizer)
        
        # 如果你还需要处理图像，可以在这里加载图像并进行预处理
        image_id = item["image"]
        image_name = image_id
        image_path = config.image_dir + image_name
        image = Image.open(image_path).convert('RGB')
        processed_image = self.processor(images=image,return_tensors="pt")
        processed_item["pixel_values"] = processed_image["pixel_values"].squeeze(0) 
        
        return processed_item

# --- DataLoader 示例 ---
# dataset = ConversationDataset("output_data.json", tokenizer)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: x) # 简单的 collate_fn

def init_model(tokenizer,trained_model=config.trained_model,rank=0,total_steps=1000):
    lconfig = AutoConfig.from_pretrained(vconfig.llm)
    model = VLMModel(lconfig)
    model.to(rank)
    if trained_model is None:
        llm = AutoModelForCausalLM.from_pretrained(vconfig.llm)
        model.load_state_dict(
            llm.state_dict(),
            strict=False
        )
        print('加载官方llm初次训练')
    else:
        # 加载权重文件（指定 map_location 到当前 GPU）
        device = torch.device('cuda', rank)
        state_dict = torch.load(trained_model, map_location=device)
        
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
        print('成功加载模型继续训练')

    #冻结视觉模块
    for name, param in model.vit_model.named_parameters():
        param.requires_grad = False
        if rank == 0:
            print(f'成功冻结: {name}')
    #冻结自注意力层和前馈层
    for name,param in model.model.named_parameters():
        param.requires_grad = False
    if rank == 0:
        print('冻结所有llm模块')
    for layer in model.model.layers:
        cross_attn_module = layer.cross_attn
        for name,param in cross_attn_module.named_parameters():
            param.requires_grad = True
    if rank == 0:
        print(f'成功解冻交叉注意力层')
    #解冻最后4层
    # for layer in model.model.layers[-config.layers_to_unfreeze:]:
    #     for name,param in layer.named_parameters():
    #         param.requires_grad = True
    if rank == 0:
        print(f'成功解冻最后 {config.layers_to_unfreeze} 层')

    if rank == 0:
        print("\n--- 可训练参数列表 (requires_grad=True) ---")
        trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
        
        if not trainable_params:
            print("警告：没有找到任何可训练的参数！")
        else:
            for name in trainable_params:
                print(name)
            
            # 计算并打印可训练参数的总数
            total_trainable_params = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad)
            print(f"\n可训练参数总数: {total_trainable_params:,}")
        print("----------------------------------------")
            


    if config.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=False
        )

    #criterion = nn.CrossEntropyLoss().to(rank)

    # 优化器：使用 AdamW（带权重衰减的 Adam）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay  # 权重衰减，防止过拟合
    )
    # 学习率调度器：随训练步数衰减学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config.min_learning_rate  # 最小学习率
    )
    scaler = GradScaler('cuda')
    return model, optimizer, scheduler ,scaler

def train_epoch(model, tokenizer, dataloader, optimizer, scheduler, scaler, config,rank):
    model.train()  # 训练模式（启用 dropout 等）
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", disable=(rank != 0)):
        if rank == 0:
            config.step = config.step + 1
        input_ids = batch['input_ids'].to(rank)
        attention_mask = batch['attention_mask'].to(rank)
        labels = batch['labels'].to(rank)
        pixel_values = batch['pixel_values'].to(rank)

        optimizer.zero_grad()
        with autocast('cuda'):

            output = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels,
                pixel_values = pixel_values,
            )
            loss = output.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()

        if config.step % 200 == 0 and rank == 0:
            swanlab.log({
                f'step_loss': loss.item(),
            },step = config.step)
        if config.step % 1000 == 0 and rank == 0:
            model_path = config.save_model
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(),model_path)
            print(f'成功保存当前最新模型参数到{model_path}') 
    avg_loss = total_loss / len(dataloader)
        
    return avg_loss

def main(rank,world_size,config):
    setup(rank, world_size)
    if rank == 0:
        swanlab.login(api_key="Nj75sPpgjdzUONcpKxlg6")
        swanlab.init(
            project=config.swanlab_project_name,
            experiment_name="train",
            config=vars(config)  # 自动记录所有超参数
        )
    tokenizer = AutoTokenizer.from_pretrained(vconfig.llm)
    image_processor = AutoImageProcessor.from_pretrained(vconfig.model_name)

    train_dataset = VLMDataset(config.train_json_file,tokenizer,image_processor)
    #val_dataset = VLMDataset(config.val_json_file,tokenizer,image_processor)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    #val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_dataloader = DataLoader(train_dataset,batch_size=config.batch_size,
                                  num_workers=4,sampler=train_sampler,pin_memory=True,drop_last=True)
    # val_dataloader = DataLoader(val_dataset,batch_size=config.batch_size,
    #                              num_workers=4,sampler=val_sampler,pin_memory=True)
    total_steps = len(train_dataloader)*config.epochs
    if rank == 0:
        print(f'总训练步数:{total_steps}')
    model, optimizer, scheduler, scaler = init_model(tokenizer=tokenizer,rank=rank,total_steps=total_steps)

    # 开始训练
    print(f"开始训练")
    #best_validate_loss = float('inf')
    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)
        train_loss = train_epoch(model,tokenizer,train_dataloader,optimizer,scheduler,scaler,config,rank)
        # if rank == 0:
        #     os.makedirs(os.path.dirname(config.latest_model), exist_ok=True)
        #     torch.save(model.state_dict(),config.latest_model)
        #     print(f'成功保存当前最新轮次模型参数到{config.latest_model}')

        #     swanlab.log({
        #         "train/epoch_avg_loss": train_loss,  # 每轮平均损失
        #     }, step=epoch + 1)  # 以 epoch 为步长
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='多模态训练')
    parser.add_argument('--learning_rate',type=float,default=config.learning_rate,help='学习率')
    parser.add_argument('--epochs',type=int,default=config.epochs,help='epochs')
    parser.add_argument('--swanlab_project_name',type=str,default=config.swanlab_project_name,help='swanlab项目名')
    parser.add_argument('--image_dir',type=str,default=config.image_dir,help='训练图像文件夹路径')
    parser.add_argument('--trained_model',type=str,default=config.trained_model,help='预训练模型路径')
    parser.add_argument('--save_model',type=str,default=config.save_model,help='模型保存路径')
    args = parser.parse_args()
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            setattr(config, key, value)
            
    mp.spawn(
        main,
        args=(config.world_size, config),
        nprocs=config.world_size,
        join=True
    )