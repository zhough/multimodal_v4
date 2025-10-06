def process_conversation(conversation,tokenizer):
    """
    处理单个对话，生成 input_ids, attention_mask, 和 labels。
    """
    # 1. 拼接对话成一个字符串
    # 注意：要包含所有特殊token和换行符
    dialogue_text = ""
    for turn in conversation:
        role = turn["from"]
        content = turn["value"]
        # 你的 value 已经是格式化好的了，直接拼接
        dialogue_text += content
    
    # 2. Tokenize
    # return_tensors='pt' 会返回 PyTorch tensors
    encoding = tokenizer(dialogue_text, return_tensors='pt', padding="max_length", truncation=True, max_length=512)
    
    input_ids = encoding["input_ids"][0] # 移除 batch 维度
    attention_mask = encoding["attention_mask"][0]
    
    # 3. 创建 Labels
    labels = input_ids.clone()
    
    # 4. 找到 user 部分并屏蔽它
    # 找到 <|im_end|> 的位置，它标志着 user 输入的结束
    # 注意：一个对话里可能有多个 user/assistant 轮次，但通常一次训练只处理一轮
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    im_end_positions = (input_ids == im_end_token_id).nonzero(as_tuple=True)[0]
    
    if len(im_end_positions) > 0:
        # 假设第一个 <|im_end|> 就是 user 输入的结束
        user_part_end_idx = im_end_positions[0]
        
        # 将 user 部分（从开始到第一个 <|im_end|>）的 labels 设为 -100
        labels[:user_part_end_idx + 1] = -100
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }