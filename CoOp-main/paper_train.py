from coopvptdann import *
import clip

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-B/16", device=device, jit=False)


class Config:
    def __init__(self, data):
        self.__dict__ = data


cfg_data = {
    "TRAINER_COOP_PREC": "fp32",
    "TRAINER_COOP_N_CTX": 5,
    "TRAINER_COOP_CLASS_TOKEN_POSITION": "end",
    "MODEL_BACKBONE_NAME": "ViT-B/16",
    "OPTIM_LR": 1e-6,
    "OPTIM_STEP_SIZE": 10,
    "OPTIM_GAMMA": 0.1,
    "INPUT_SIZE": [224, 224]
}

cfg = Config(cfg_data)


customCLIP = CustomCLIP(cfg, "end", clip_model, 5)
print(customCLIP)
