import os
import random
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.imagenet import ImageNet
from datasets.utils import build_data_loader
import clip
from utils import *
import json


def extract_text_feature(cfg, classnames, clip_model, template):
    with torch.no_grad():
        clip_weights = []  # 初始化列表，用来存储文本特征
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')

            texts = [t.format(classname) for t in template]  # 根据模板格式化类别名称

            texts_token = clip.tokenize(texts, truncate=True).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts_token)  # 得到文本嵌入
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  # 进行归一化处理
            class_embedding = class_embeddings.mean(dim=0)  # 计算平均值作为该类别的特征向量
            class_embedding /= class_embedding.norm()  # 再次归一化
            clip_weights.append(class_embedding)  # 加入到列表里

        clip_weights = torch.stack(clip_weights, dim=1).cuda()  # 将所有类别的特征向量堆叠成一个张量，并将其移动到GPU上
    torch.save(clip_weights, cfg['cache_dir'] + "/text_weights.pt")  # 将特征向量保存到文件中
    return


def extract_multi_scale_feature(cfg, split, clip_model, loader, scale):
    features, labels = [], []  # 初始化特征和标签列表
    with torch.no_grad():
        for crop_idx in range(cfg['crop_epoch']):  # 遍历裁剪次数
            features_this = []
            for i, (images, target) in enumerate(tqdm(loader)):  # 遍历图像和目标
                images = images.cuda()
                image_features = clip_model.encode_image(images)  # 提取图像特征
                # image_features /= image_features.norm(dim=-1, keepdim=True)
                features_this.append(image_features)  # 添加到裁剪特征中
                if crop_idx == 0:  # 如果是第一次裁剪，将目标也移动到GPU上并添加到标签列表中
                    target = target.cuda()
                    labels.append(target)
            features.append(torch.cat(features_this, dim=0))  # 将当前裁剪的特征拼接起来并添加到总特征列表中
    features, labels = torch.stack(features, dim=0), torch.cat(labels)  # 将所有裁剪的特征堆叠在一起，并将所有标签拼接起来
    torch.save(features, cfg['cache_dir'] + "/" + split + "_f" + "_" + str(scale) + "_.pt")  # 保存特征到缓存目录
    label_path = cfg['cache_dir'] + "/" + split + "_l.pt"  # 检查标签文件是否存在，如果不存在则保存标签
    if not os.path.exists(label_path):
        torch.save(labels, label_path)
    return


def global_feature(cfg, split, clip_model, loader):
    features, labels = [], []
    with torch.no_grad():
        features_this = []
        for i, (images, target) in enumerate(tqdm(loader)):  # 遍历图像和目标
            images = images.cuda()
            image_features = clip_model.encode_image(images)  # 提取图像特征
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.squeeze(0)
            features_this.append(image_features)
            target = target.cuda()
            labels.append(target)
        features.append(torch.cat(features_this, dim=0))
    features, labels = torch.stack(features, dim=0), torch.cat(labels)  # 将所有裁剪的特征堆叠在一起，并将所有标签拼接起来
    torch.save(features, cfg['cache_dir'] + "/" + split + "_f" + ".pt")  # 保存特征到缓存目录
    label_path = cfg['cache_dir'] + "/" + split + "_l.pt"  # 检查标签文件是否存在，如果不存在则保存标签
    if not os.path.exists(label_path):
        torch.save(labels, label_path)
    return


def extract_ten_crop_feature(cfg, split, clip_model, loader):
    features, labels = [], []  # 初始化特征和标签列表
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):  # 遍历数据加载器中的图像和目标
            images = images.cuda()
            features_this = []
            for crops in range(images.shape[1]):  # 遍历图像的所有裁剪版本
                this_images = images[:, crops, :, :]
                image_features = clip_model.encode_image(this_images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features_this.append(image_features)  # 将归一化后的图像特征添加到当前图像的特征列表中
                if crops == 0:  # 如果是第一个裁剪，将目标移动到GPU上并添加到标签列表中
                    target = target.cuda()
                    labels.append(target)
            features.append(torch.stack(features_this, dim=0).mean(dim=0))  # 计算当前图像所有裁剪版本的平均特征，并将其添加到特征列表中
    features = torch.cat(features, dim=0)  # 将所有图像的特征拼接在一起
    features /= features.norm(dim=-1, keepdim=True)
    labels = torch.cat(labels)  # 将所有图像的标签拼接在一起
    torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")  # 保存特征到缓存目录
    label_path = cfg['cache_dir'] + "/" + split + "_l.pt"  # 检查标签文件是否存在，如果不存在则保存标签到缓存目录
    if not os.path.exists(label_path):
        torch.save(labels, label_path)
    return


if __name__ == '__main__':
    # 加载预训练的CLIP模型（RN50）
    clip_model, preprocess = clip.load('RN50')
    clip_model.eval()
    # 定义要处理的数据集列表
    # all_dataset = ["caltech101", 'dtd', 'eurosat', 'fgvc', 'food101', 'imagenet',
    #                'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397', 'ucf101']
    all_dataset = ["caltech101"]
    k_shot = [1]
    data_path = "D:\\Code\\Data"
    this_scale = 0.1
    # 遍历所有数据集
    for set in all_dataset:
        # 读取数据集的配置信息
        cfg = yaml.load(open('configs/{}.yaml'.format(set), 'r'), Loader=yaml.Loader)
        cache_dir = os.path.join('./caches', cfg['dataset'])
        os.makedirs(cache_dir, exist_ok=True)
        cfg['cache_dir'] = cache_dir
        cfg['crop_epoch'] = 100
        # 遍历不同的k-shot值
        for k in k_shot:
            # 设置随机种子以确保结果可重复
            random.seed(1)
            torch.manual_seed(1)

            cfg['shots'] = k
            # 10-crop
            # 定义测试时的数据转换操作
            # test_transform = transforms.Compose([
            #     transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC),
            #     transforms.TenCrop(size=224),
            #     transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            #     transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            #         mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(crop) for crop
            #                                                  in crops])),
            # ])
            # multi-scale
            test_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(this_scale, this_scale),
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))])
            # 根据数据集类型构建数据加载器
            if set == 'imagenet':
                dataset = ImageNet(cfg['root_path'], cfg['shots'], test_transform)
                val_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)
                train_loader_cache = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8,
                                                                 shuffle=False)
            else:
                dataset = build_dataset(set, data_path, k)

                val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False,
                                               tfm=test_transform, shuffle=False)
                test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False,
                                                tfm=test_transform, shuffle=False)

        # Extract multi-scale features
        print("\nLoading visual features and labels from val and test set.")
        # global_feature(cfg, "val", clip_model, val_loader)
        # global_feature(cfg, "test", clip_model, val_loader)
        # extract_ten_crop_feature(cfg, "val", clip_model, val_loader)
        extract_multi_scale_feature(cfg, "val", clip_model, val_loader, this_scale)
        if not set == 'imagenet':
            # extract_ten_crop_feature(cfg, "test", clip_model, test_loader)
            extract_multi_scale_feature(cfg, "test", clip_model, val_loader, this_scale)
    extract_text_feature(cfg, dataset.classnames, clip_model, dataset.template)
