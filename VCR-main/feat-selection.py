import torch
import yaml
import os
import pdb


# 加载文本特征
def load_text_feature(textual_dir):
    save_path = textual_dir + "/text_weights.pt"
    clip_weights = torch.load(save_path)
    return clip_weights


# 选择图像视图
def select_image_views(view_feat, clip_weights):
    norm_view_feat = view_feat / view_feat.norm(dim=-1, keepdim=True)  # 归一化
    local_logits = norm_view_feat @ clip_weights   # 计算logits

    # 使用torch.topk函数找到局部logits值的前两个最大值。k=2表示我们想要找到最大的两个值，dim=-1表示在最后一个维度上进行操作。
    logits_values, _ = torch.topk(local_logits, k=2, dim=-1)
    # 计算前两个最大值之间的差值，作为选择图像视图的依据。
    criterion = logits_values[:,:,0] - logits_values[:,:,1]
    # 对差值进行降序排序，并取第一个索引，即差值最大的那个索引。
    local_idx = torch.argsort(criterion, dim=0, descending=True)[:1]
    # 使用torch.take_along_dim函数根据local_idx从view_feat中提取对应的图像视图特征。
    # local_idx[:,:,None]将索引扩展为与view_feat相同的形状，以便进行索引操作。最后，使用squeeze(0)去除多余的维度。
    selected = torch.take_along_dim(view_feat, local_idx[:,:,None], dim=0).squeeze(0)
    return selected

def load_feature(dir, scale, split):
    feat_dir = dir + "/" + split + "_f_" + scale + "_.pt"
    features = torch.load(feat_dir)

    return features

def save_feature(selected_features, save_dir, scale, split):
    save_path = save_dir + "/" + split +"_f_" + scale + ".pt"
    torch.save(selected_features, save_path)
    return


# all_dataset = ["caltech101", 'dtd', 'eurosat', 'fgvc', 'food101', 'imagenet',
#                 'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397', 'ucf101']
all_dataset = ["caltech101"]
for set in all_dataset:
    # path
    textual_dir = os.path.join('./caches', set)
    feat_dir = os.path.join('./caches', set)
    save_dir = os.path.join('./selected', set)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # textual features
    print("Getting textual features as CLIP's classifier.")
    clip_weight = load_text_feature(textual_dir)
    # 遍历不同的尺度（从1到9）
    for scale in range(1, 10):
        print("Getting features.")
        features = load_feature(feat_dir, str(scale), 'val')
        selected_feature = select_image_views(features, clip_weight)
        save_feature(selected_feature, save_dir, str(scale), 'val')
        # 如果不是'imagenet'数据集，则对测试集执行相同的操作
        if not set == 'imagenet':
            features = load_feature(feat_dir, str(scale), 'test')
            selected_feature = select_image_views(features, clip_weight)
            save_feature(selected_feature, save_dir, str(scale), 'test')