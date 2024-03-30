import os.path as osp
import numpy as np
from torch.nn.modules.utils import _pair

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.autograd import Function

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import math
from functools import reduce
from operator import mul

_tokenizer = _Tokenizer()


# 加载预训练模型到CPU
def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


# GRL函数
class ReverseLayerF(Function):
    r"""Gradient Reverse Layer(Unsupervised Domain Adaptation by Backpropagation)
    Definition: During the forward propagation, GRL acts as an identity transform. During the back propagation though,
    GRL takes the gradient from the subsequent level, multiplies it by -alpha  and pass it to the preceding layer.

    Args:
        x (Tensor): the input tensor
        alpha (float): \alpha =  \frac{2}{1+\exp^{-\gamma \cdot p}}-1 (\gamma =10)
        out (Tensor): the same output tensor as x
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# 文本编码器
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        # tokenized_prompts.argmax(dim=-1) 返回每个序列中最大数字的索引，即每个序列的结束标记（End of Text，EOT）的位置。
        # torch.arange(x.shape[0])生成一个从0到batch_size-1的整数序列，用于在第一个维度上进行索引。
        # x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] 通过上述两个步骤，选取了每个序列的最后一个特征向量，形状为 (batch_size, feature_dim)。
        # @ self.text_projection 将选取的特征向量与 self.text_projection 相乘，得到最终的文本特征表示。

        return x


# 域分类器，用于实现DANN
class Domain(nn.Module):
    r"""Long domain classifier
        connect to tne feature extractor via a gradient reverse layer that multiplies by
        a certain negative constant during the backpropagation-based training

        Distinguish the features as a source or target (minimize domain loss)

        Args:
            in_features: size of input layer unit, default: 256
            hidden_size: size of hidden layer unit, default: 1024
        """

    def __init__(self, in_features=512, hidden_size=1024):
        super(Domain, self).__init__()
        self.fc1_prompt = nn.Linear(in_features, hidden_size)
        self.fc2_prompt = nn.Linear(hidden_size, hidden_size)
        self.fc3_prompt = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

        self.__in_features = 1
        # self.init_weight()

    def forward(self, x, alpha):
        r"""flip all the samples' sign of gradients when back-propagation
        :param x: the input Tensor as [bs, features_dim]
        :param alpha: ratio
        :return: the domain label prediction(1 dimension and use BCEloss)
        """
        x = ReverseLayerF.apply(x, alpha)
        x = self.fc1_prompt(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2_prompt(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3_prompt(x)
        x = self.sigmoid(x)
        return x

    def init_weight(self):
        nn.init.normal_(self.fc1.weight.data, 0, 0.01)
        nn.init.normal_(self.fc2.weight.data, 0, 0.01)
        nn.init.normal_(self.fc3.weight.data, 0, 0.3)

    def get_parameters(self):
        r"""get the different parameters of each layer"""
        return [{'params': self.parameters(), 'lr_mult': 100, 'decay_mult': 2}]

    def output_num(self):
        return self.__in_features


# 定义可学习的提示
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, num_tokens):
        super().__init__()
        n_cls = len(classnames)  # 类别数量
        n_ctx = cfg.TRAINER.COOP.N_CTX  # 上下文长度
        ctx_init = cfg.TRAINER.COOP.CTX_INIT  # 上下文初始化
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

        patch_size = _pair(16)
        prompt_dim = 768
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, num_tokens, prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        total_d_layer = 11  # 深层
        self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
            total_d_layer, num_tokens, prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        self.Domain_discriminator = Domain()

    # 前向传播，生成prompt
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class VPT_VisionTransformer(nn.Module):
    def __init__(self, num_tokens, clip_model):
        super(VPT_VisionTransformer, self).__init__()
        self.visual = clip_model.visual
        self.num_tokens = num_tokens  # number of prompted tokens
        # if project the prompt embeddings
        self.prompt_proj = nn.Identity()

    def incorporate_prompt(self, x, prompt_embeddings):  # 将提示插入类token和image-patch之间
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]

        # after CLS token, all before image patches
        # print("x.dtype:{}".format(x.dtype))
        # print("self.visual.conv1.weight.dtype:{}".format(self.visual.conv1.weight.dtype))
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                   device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # (1 + n_patches, batch_size, hidden_dim)
        x = torch.cat((
            x[:1, :, :],
            self.prompt_proj(prompt_embeddings).expand(B, -1, -1).permute(1, 0, 2),
            x[1:, :, :]
        ), dim=0)
        # (cls_token + n_prompt + n_patches, batch_size, hidden_dim)

        return x

    def forward_deep_prompt(self, embedding_output, deep_prompt_embeddings):
        hidden_states = None
        B = embedding_output.shape[1]
        num_layers = 12

        for i in range(num_layers):  # 在Vit的每一层输入添加提示
            if i == 0:
                embedding_output = embedding_output.type(torch.float16)
                hidden_states = self.visual.transformer.resblocks[i](embedding_output)
            else:
                if i <= deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_proj(
                        deep_prompt_embeddings[i - 1]).expand(B, -1, -1).permute(1, 0, 2)

                    hidden_states = torch.cat((
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[(1 + self.num_tokens):, :, :]
                    ), dim=0)  # 将中间的[1,1+num_tokens)替换成提示
                hidden_states = hidden_states.type(torch.float16)
                hidden_states = self.visual.transformer.resblocks[i](hidden_states)
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = self.visual.ln_post(hidden_states[:, 0, :])

        if self.visual.proj is not None:
            hidden_states = hidden_states @ self.visual.proj

        return hidden_states

    def forward(self, x, prompt_embeddings, deep_prompt_embeddings):
        # this is the default version:
        embedding_output = self.incorporate_prompt(x, prompt_embeddings)

        image_encoded = self.forward_deep_prompt(
            embedding_output, deep_prompt_embeddings)
        return image_encoded


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, num_tokens=0):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model, num_tokens)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        if num_tokens == 0:
            self.image_encoder = clip_model.visual
        else:
            self.image_encoder = VPT_VisionTransformer(num_tokens, clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def visual(self, image):
        return self.image_encoder(image.type(self.dtype), self.prompt_learner.prompt_embeddings,
                                  self.prompt_learner.deep_prompt_embeddings)

    def domain_discriminator(self, image_features, alpha):
        # print(image_features.dtype)
        # print(self.Domain_discriminator.fc1_prompt.weight.dtype)
        return self.prompt_learner.Domain_discriminator(
            image_features.type(self.prompt_learner.Domain_discriminator.fc1_prompt.weight.dtype), alpha)

    def forward_use_image_features(self, image_features):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype), self.prompt_learner.prompt_embeddings,
                                            self.prompt_learner.deep_prompt_embeddings)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class CoOpVPTDANN(TrainerXU):
    """Context Optimization (CoOp) + Vision prompt tuning (VPT).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, num_tokens=3)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            else:
                print(name)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch_x, batch_u):
        image_x, label_x, image_u = self.parse_batch_train(batch_x, batch_u)
        batch_size = len(image_x)

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                img_feat_x = self.model.visual(image_x)
                img_feat_u = self.model.visual(image_u)
                img_feat = torch.cat((img_feat_x, img_feat_u), 0)

                output = self.model.forward_use_image_features(img_feat_x)
                domain_labels = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(
                    self.device)
                alpha = 2. / (
                        1. + np.exp(-10 * float(self.epoch * self.num_batches / self.max_epoch * self.num_batches))) - 1
                domain_output = self.model.domain_discriminator(img_feat, alpha)
                transfer_criterion = nn.BCELoss()
                loss = F.cross_entropy(output, label_x) + transfer_criterion(domain_output, domain_labels)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            img_feat_x = self.model.visual(image_x)
            img_feat_u = self.model.visual(image_u)
            img_feat = torch.cat((img_feat_x, img_feat_u), 0)

            output = self.model.forward_use_image_features(img_feat_x)
            domain_labels = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(self.device)
            alpha = 2. / (
                    1. + np.exp(-10 * float(self.epoch * self.num_batches / self.max_epoch * self.num_batches))) - 1
            domain_output = self.model.domain_discriminator(img_feat, alpha)
            transfer_criterion = nn.BCELoss()
            loss1 = F.cross_entropy(output, label_x)
            loss2 = transfer_criterion(domain_output, domain_labels)
            loss = loss1 + 0.5 * loss2
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "source_loss": loss1,
            "transfer_loss": loss2,
            "acc": compute_accuracy(output, label_x)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, input_u

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
