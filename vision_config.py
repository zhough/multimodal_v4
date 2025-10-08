class VisionConfig():
    def __init__(self):
        self.model_name = "google/vit-base-patch16-224"
        self.num_layers = 3
        self.v_hidden_size = 768
        self.num_heads = 8
        self.llm = "Qwen/Qwen3-0.6B"