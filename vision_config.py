import torch
class VisionConfig():
    def __init__(self):
        self.model_name = "google/vit-large-patch16-224"
        self.num_layers = 3
        self.v_hidden_size = 1024
        self.num_heads = 8
        #self.fusion_layers = [0,1,2,10,11,12,13,14,15,20,21,22]
        self.fusion_layers = [i for i in range(6,28,2)]
        self.llm = "Qwen/Qwen3-0.6B"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
