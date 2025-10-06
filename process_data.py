# import csv
# import json
# from pathlib import Path
# import pandas as pd
# def csv_to_json_converter(csv_file_path, json_file_path):

#     # 确保输出目录存在
#     Path(json_file_path).parent.mkdir(parents=True, exist_ok=True)

#     data = []
#     try:
#         df = pd.read_csv(csv_file_path,sep='|')

#         for row in df.values:
#             # 格式化 question 和 answer 内容
#             formatted_question = "<|im_start|>user\ndescribe this picture.<|im_end|>"
#             formatted_answer = f"<|im_start|>assistant\n{row[2]}<|im_end|>"
            
#             # 构建 conversations 列表
#             conversations = [
#                 {"from": "user", "value": formatted_question},
#                 {"from": "assistant", "value": formatted_answer}
#             ]
            
#             # 创建最终的 JSON 对象
#             json_object = {
#                 "conversations": conversations,
#                 "image": row[0]
#             }
            
#             data.append(json_object)
    
#         # 将收集到的数据写入 JSON 文件
#         with open(json_file_path, mode='w', encoding='utf-8') as json_file:
#             # 使用 ensure_ascii=False 确保中文等非 ASCII 字符正常显示
#             # indent=2 让输出的 JSON 更易读
#             json.dump(data, json_file, ensure_ascii=False, indent=2)
            
#         print(f"✅ 转换成功！文件已保存到: {json_file_path}")

#     except FileNotFoundError:
#         print(f"❌ 错误: 找不到 CSV 文件 '{csv_file_path}'")
#     except Exception as e:
#         print(f"❌ 发生未知错误: {e}")

# if __name__ == '__main__':
#     # 假设你的 CSV 文件名为 'data.csv'，并且和这个 Python 脚本在同一目录下
#     input_csv_path = './results.csv'
#     # 输出的 JSON 文件名
#     output_json_path = './frickr30k.json'
#     # 调用函数进行转换
#     csv_to_json_converter(input_csv_path, output_json_path)


import json
import os

def convert_conversation_format(original_item):
    """
    将单条原始数据转换为目标格式（不改变图像路径）。
    """
    converted_conversations = []
    
    for turn in original_item.get("conversations", []):
        speaker = turn.get("from")
        content = turn.get("value", "")
        
        if speaker == "human":
            new_speaker = "user"
            cleaned_content = content.strip()
            formatted_content = f"<|im_start|>user\n{cleaned_content}<|im_end|>"
        elif speaker == "gpt":
            new_speaker = "assistant"
            formatted_content = f"<|im_start|>assistant\n{content}<|im_end|>"
        else:
            # 跳过未知角色，继续处理下一条
            continue
        
        converted_conversations.append({
            "from": new_speaker,
            "value": formatted_content
        })

    # 直接使用原始的 image 路径
    image_path = original_item.get("image", "")
    image_name = os.path.basename(image_path)
    return {
        "conversations": converted_conversations,
        "image": image_name
    }

def process_json_file(input_path, output_path):
    """
    读取JSON文件，批量转换格式，并保存到新文件。
    """
    print(f"开始处理文件: {input_path}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在于 '{input_path}'")
        return

    try:
        # 1. 读取原始JSON文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确保数据是一个列表
        if not isinstance(data, list):
            print("错误: 输入文件内容不是一个有效的JSON列表。")
            return

        # 2. 批量转换数据格式
        converted_data = []
        for item in data:
            converted_item = convert_conversation_format(item)
            converted_data.append(converted_item)
            
            # 打印进度（每处理1000条打印一次）
            if len(converted_data) % 1000 == 0:
                print(f"  -> 已处理 {len(converted_data)} 条数据...")

        print(f"\n处理完成，共转换 {len(converted_data)} 条数据。")

        # 3. 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 4. 将转换后的数据写入新的JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
            
        print(f"\n✅ 转换成功！文件已保存到: {output_path}")

    except json.JSONDecodeError:
        print(f"❌ 错误: 输入文件 '{input_path}' 不是一个有效的JSON格式。")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

# --- 主程序入口 ---
if __name__ == '__main__':
    # --- 在这里修改你的文件路径 ---
    # 原始JSON文件的路径
    INPUT_FILE = './sharegpt4v_instruct_gpt4-vision_cap100k.json' 
    
    # 转换后新JSON文件的保存路径
    OUTPUT_FILE = './coco.json'
    
    # 调用函数执行转换
    process_json_file(INPUT_FILE, OUTPUT_FILE)