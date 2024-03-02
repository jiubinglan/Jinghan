import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device) #图像预处理，input是clip的架构图
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device) #文本预处理

with torch.no_grad():
    image_features = model.encode_image(image) #图像编码
    text_features = model.encode_text(text) # 文本编码

    logits_per_image, logits_per_text = model(image, text) #计算图像和每一个文本的相似度
    probs = logits_per_image.softmax(dim=-1).cpu().numpy() # 相似度归一化

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]