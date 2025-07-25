# FocusedKAN
# %%
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple
from transformers import PretrainedConfig, EarlyStoppingCallback
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments, TrainerCallback
import copy
from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
from safetensors.torch import load_file
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score


# %%
class Config:
    def __init__(self, n_heads, d_model, attention_dropout, num_transformers, num_classes,):
        self.n_heads = n_heads
        self.dim = d_model
        self.attention_dropout = attention_dropout
        self.num_transformers = num_transformers
        self.num_classes = num_classes


# >KAN class<
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        dim_feedforward,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,  # * Learnable af
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = dim_feedforward
        self.dim_feedforward = dim_feedforward
        self.grid_size = grid_size
        self.spline_order = spline_order

        in_features = self.in_features
        h = (grid_range[1] - grid_range[0]) / grid_size
        # * grid is the interpolated points
        grid = (
            (  # 生成从 -spline_order 到 grid_size + spline_order 的一系列整数点。
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]  # 将整个点序列平移，使其起始点位于 grid_range[0]
            )
            # 将一维的序列扩展为有 in_features 行的二维张量，相当于复制该序列给每个输入特征
            .expand(in_features, -1)
            .contiguous()
        )

        # * register buffer no need to back-propagate for data points
        self.register_buffer("grid", grid)

        # * back-propagation for weights
        self.base_weight = torch.nn.Parameter(
            torch.Tensor(dim_feedforward, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(dim_feedforward, in_features,
                         grid_size + spline_order)
        )
        self.alphas = torch.nn.Parameter(
            torch.ones(self.dim_feedforward) * 0.1)

        # if true spline is scalable
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(dim_feedforward, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        # * initialize weights
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # * 生成随机扰动
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features,
                               self.dim_feedforward)
                    - 1 / 2
                )
                # * standarization
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )

            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
    # * 构建样条函数

    def b_splines(self, x: torch.Tensor):
        """
        Args:       x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:    torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)

        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        # 递归计算样条函数bases，低阶->高阶 x特征映射
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def penalty_regression(self, A, B):
        I = torch.eye(A.size(-1), device=A.device)
        alpha_matrix = self.alphas.view(-1, 1, 1)
        regularized_matrix = A.transpose(1, 2).bmm(
            A) + alpha_matrix * I.unsqueeze(0)
        right_hand_side = A.transpose(1, 2).bmm(B)

        coeff = torch.linalg.solve(regularized_matrix, right_hand_side)
        return coeff.permute(2, 0, 1)

    # * Spline coefficients
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.dim_feedforward)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, dim_feedforward )
        # * A = input, B = target, using penalty solving method
        result = self.penalty_regression(A, B)

        assert result.size() == (
            self.dim_feedforward,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    # * read-only property
    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    # * feedforward -- 线性变换+样条结果
    # * base 路径使用了SiLU激活函数(全局性），spline 路径是纯非线性变换（捕捉局部特征细节）
    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features

        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.dim_feedforward, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.dim_feedforward)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(
            splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] +
                        2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + \
            (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1,
                               device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1,
                               device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(
            self.curve2coeff(x, unreduced_spline_output))

    # * for weights but not grid and spline
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


# >MultiHeadClass<
class MultiHeadAttention(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

        self.n_heads = config.n_heads
        self.num_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.is_causal = False

        if self.dim % self.n_heads != 0:
            raise ValueError(f"self.n_heads: {self.n_heads} must divide self.dim: {
                             self.dim} evenly")

        self.head_dim = self.dim // self.n_heads

        # *attention heads weight matrices*
        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(
            in_features=config.dim, out_features=config.dim)

        self.frozen_heads = [False] * self.n_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        bs, q_length, dim = query.size()
        k_length = key.size(1)

        dim_per_head = self.dim // self.n_heads

        # * reshape input tensors for efficient attention computation then output
        def shape(x: torch.Tensor) -> torch.Tensor:
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        # * actual input tensors into linear transformation and form Q K V matrix
        q = shape(self.q_lin(query))
        k = shape(self.k_lin(key))
        v = shape(self.v_lin(value))

        # * Calculate attention scores
        # * rescale to avoid gradient explosion
        q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))

        # * token mask
        # 1. 计算注意力得分
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # 2. 通过softmax函数获得probabilities
        weights = nn.functional.softmax(scores, dim=-1)
        # 3. drop some weights? how? set to zeros?
        weights = self.dropout(weights)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(
                bs, 1, 1, k_length).expand_as(scores)
            scores = scores.masked_fill(key_padding_mask == 0, float('-inf'))

        # * Head mask/attention mask
        if head_mask is None:
            head_mask = torch.ones(self.n_heads, device=weights.device)
            for idx, is_frozen in enumerate(self.frozen_heads):
                if is_frozen:
                    head_mask[idx] = 0.0
            head_mask = head_mask.view(1, -1, 1, 1)

        # 4. true attention Weights
        weights = weights * head_mask

        # 5. V:value matrix dot-produce with attention weights
        context = torch.matmul(weights, v)
        # 6. expand and through the output linear transformation
        context = unshape(context)
        context = self.out_lin(context)

        # * whether or not to track the weights (for visuals)
        if output_attentions:
            return context, weights
        else:
            return context,


# >Transformer<
class TransformersWithKan(nn.Module):
    def __init__(self, d_model, nheads, dropout):
        super().__init__()
        config = Config(n_heads=nheads, d_model=d_model,
                        attention_dropout=dropout, num_transformers=2, num_classes=2)
        self.self_attn = MultiHeadAttention(config)

        self.linear1 = KANLinear(d_model)
        self.linear2 = KANLinear(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model

    # KANlinear contains af plus linear layer already
    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # passes the normalized tensor for linear transformation
        src2 = self.linear1(src)
        # deleted duplicant framework

        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


# %%

# *Encapsulate as an HF model to facilitate training and subsequent repeated calls
class TextDiseasePredictor(nn.Module):
    def __init__(self, d_model, nheads, num_classes, dropout, num_transformers):
        super(TextDiseasePredictor, self).__init__()
        self.embedding = embedding_model
        self.embedding.embeddings.word_embeddings.weight.data = self.embedding.embeddings.word_embeddings.weight.data.contiguous()
        self.embedding.resize_token_embeddings(len(tokenizer))

        embedding_dim = self.embedding.config.hidden_size if hasattr(
            self.embedding, 'config') else self._get_embedding_dim()
        # * Linear dimension reduction: projects the high-dimensional embeddings down to a specified d_model dimension
        self.d_model = d_model
        self.embedding_dim_reduction = nn.Linear(embedding_dim, self.d_model)

        self.transformers = nn.ModuleList([
            # * call the TransformersWithKan module (a list of transformer layers)
            TransformersWithKan(d_model, nheads, dropout)
            for _ in range(num_transformers)
        ])
        # * fully connected
        self.fc = nn.Linear(d_model, num_classes)

    @staticmethod
    def _expand_weights(old_weight, new_weight, init_scale=0.02):
        new_weight[:old_weight.size(0), :old_weight.size(1)] = old_weight
        if new_weight.size(0) > old_weight.size(0):
            nn.init.uniform_(
                new_weight[old_weight.size(0):, :], -init_scale, init_scale)
        if new_weight.size(1) > old_weight.size(1):
            nn.init.uniform_(
                new_weight[:, old_weight.size(1):], -init_scale, init_scale)

    # * Dynamic Attention Heads Update
    def update_attention_heads(self, new_nheads):
        # * Preserve previous info
        old_nheads = self.transformers[0].self_attn.n_heads
        old_d_model = self.d_model
        old_head_dim = old_d_model // old_nheads

        new_head_dim = old_head_dim
        # * making sure the new d_model is divisible by the new nheads
        new_d_model = new_nheads * new_head_dim

        # * keep track of the change
        print(f"Updating model dimensions: {old_d_model} -> {new_d_model}")
        print(f"Updating attention heads: {old_nheads} -> {new_nheads}")

        old_reduction = self.embedding_dim_reduction
        embedding_dim = old_reduction.in_features
        # * update reduction layer
        new_reduction = nn.Linear(embedding_dim, new_d_model)

        with torch.no_grad():
            if new_d_model >= old_d_model:
                new_reduction.weight.data[:old_d_model,
                                          :] = old_reduction.weight.data
                new_reduction.bias.data[:old_d_model] = old_reduction.bias.data
                nn.init.xavier_uniform_(
                    new_reduction.weight.data[old_d_model:, :])
                nn.init.zeros_(new_reduction.bias.data[old_d_model:])
            else:
                raise ValueError(
                    "reduc: new d_model must be greater than or equal to old d_model")

        self.embedding_dim_reduction = new_reduction
        self.d_model = new_d_model

        for transformer in self.transformers:
            old_self_attn = transformer.self_attn
            new_config = Config(
                n_heads=new_nheads,
                d_model=new_d_model,
                attention_dropout=old_self_attn.dropout.p,
                num_transformers=old_self_attn.config.num_transformers,
                num_classes=old_self_attn.config.num_classes,
            )

            new_self_attn = MultiHeadAttention(new_config)

            # * Transfer existing weights to the new attention layers and initialize additional weights
            with torch.no_grad():
                self._expand_weights(
                    old_self_attn.q_lin.weight.data, new_self_attn.q_lin.weight.data)
                self._expand_weights(
                    old_self_attn.k_lin.weight.data, new_self_attn.k_lin.weight.data)
                self._expand_weights(
                    old_self_attn.v_lin.weight.data, new_self_attn.v_lin.weight.data)
                self._expand_weights(
                    old_self_attn.out_lin.weight.data, new_self_attn.out_lin.weight.data)

                new_self_attn.q_lin.bias.data[:old_self_attn.q_lin.bias.size(
                    0)] = old_self_attn.q_lin.bias.data
                new_self_attn.k_lin.bias.data[:old_self_attn.k_lin.bias.size(
                    0)] = old_self_attn.k_lin.bias.data
                new_self_attn.v_lin.bias.data[:old_self_attn.v_lin.bias.size(
                    0)] = old_self_attn.v_lin.bias.data
                new_self_attn.out_lin.bias.data[:old_self_attn.out_lin.bias.size(
                    0)] = old_self_attn.out_lin.bias.data

            # * keep track of frozen status
            new_self_attn.frozen_heads = [False] * new_nheads
            for idx, is_frozen in enumerate(old_self_attn.frozen_heads):
                if idx < len(new_self_attn.frozen_heads):
                    new_self_attn.frozen_heads[idx] = is_frozen

            transformer.self_attn = new_self_attn
            transformer.linear1 = KANLinear(new_d_model)
            transformer.linear2 = KANLinear(new_d_model)
            transformer.norm1 = nn.LayerNorm(new_d_model)
            transformer.norm2 = nn.LayerNorm(new_d_model)
            transformer.d_model = new_d_model

        old_fc = self.fc
        new_fc = nn.Linear(new_d_model, old_fc.out_features)

        with torch.no_grad():
            if new_d_model >= old_d_model:
                new_fc.weight.data[:, :old_d_model] = old_fc.weight.data
                nn.init.xavier_uniform_(new_fc.weight.data[:, old_d_model:])
            else:
                raise ValueError(
                    "fc: new d_model must be greater than or equal to old d_model")

            new_fc.bias.data = old_fc.bias.data

        self.fc = new_fc

    def forward(self, input_ids, labels=None, attention_mask=None):
        embedded_input = self.embedding(
            input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        embedded_input = self.embedding_dim_reduction(embedded_input)

        x = embedded_input
        for transformer in self.transformers:
            x = transformer(x, src_key_padding_mask=attention_mask)

        logits = self.fc(x.mean(dim=1))

        loss = None
        if labels is not None:
            # * for binary classfication task
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
        return {"loss": loss, "logits": logits}


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# %%
# Phase 2: Testing the model on the test set & begin new-stage training<<
# %%
# >Refrigerator<
class Refrigerator:
    def __init__(self, model, freeze_threshold):
        self.model = model
        # self.entropy_threshold = entropy_threshold
        self.freeze_threshold = freeze_threshold
        self.prev_weights = {}
        self.frozen_heads = {}

        for i, transformer in enumerate(model.transformers):
            self.frozen_heads[i] = []
            self.prev_weights[i] = {}
            n_heads = transformer.self_attn.n_heads
            transformer.self_attn.frozen_heads = [False] * n_heads

    def adaptive_freezer(self):
        for i, transformer in enumerate(self.model.transformers):
            self_attn = transformer.self_attn
            q_weights = self_attn.q_lin.weight.data
            n_heads = self_attn.n_heads
            head_dim = self_attn.dim // n_heads
            for head_idx in range(n_heads):
                if head_idx in self.frozen_heads[i]:
                    continue
                start_idx = head_idx * head_dim
                end_idx = (head_idx + 1) * head_dim
                head_weights = q_weights[:, start_idx:end_idx]
                if head_idx in self.prev_weights[i]:
                    # * calculate attention weights
                    weight_change = (
                        head_weights - self.prev_weights[i][head_idx]).abs().mean().item()
                    # * if change is less than threshold, add head to frozen_heads index list
                    if weight_change < self.freeze_threshold:
                        # print(f"Freezing Layer {i} Head {head_idx} (change: {weight_change:.8f})")
                        self.frozen_heads[i].append(head_idx)
                        self_attn.frozen_heads[head_idx] = True
                self.prev_weights[i][head_idx] = head_weights.clone()

    def show_frozen_status(self):
        print("\n=== Frozen Status ===")
        for i, transformer in enumerate(self.model.transformers):
            total_heads = transformer.self_attn.n_heads
            # No need to convert to list, already a list
            frozen = sorted(self.frozen_heads[i])
            # active = sorted(set(range(total_heads)) - set(self.frozen_heads[i]))  # Use a set for subtraction
            print(f"Layer {i}:")
            print(f"  Total heads: {total_heads}")
            print(f"  Frozen heads: {frozen} ({len(frozen)}/{total_heads})")

    def check_free_heads(self):
        for i, transformer in enumerate(self.model.transformers):
            self_attn = transformer.self_attn
            current_free_heads = self_attn.n_heads - len(self.frozen_heads[i])
            if current_free_heads < self.min_free_heads:
                # * check if the current number of free heads is less than the minimum number of free heads
                self.model.update_attention_heads(
                    self_attn.n_heads + (self.min_free_heads - current_free_heads))


class FreezingCallback(TrainerCallback):
    def __init__(self, freeze_threshold, check_interval):
        super().__init__()
        self.freeze_threshold = freeze_threshold
        # self.entropy_threshold = entropy_threshold
        self.check_interval = check_interval

    def on_train_begin(self, args, state, control, model, **kwargs):
        if not hasattr(self, 'refrigerator'):
            self.refrigerator = Refrigerator(model, self.freeze_threshold)
        else:
            # * No reinitialize
            self.refrigerator.model = model

    def on_step_end(self, args, state, control, **kwargs):
        # * Check if the current step is a multiple of the check interval
        # * if so, call the adaptive_freezer() method
        if state.global_step % self.check_interval == 0:
            self.refrigerator.adaptive_freezer()
            self.refrigerator.show_frozen_status()

    # * Synchronize the frozen_heads list with the new number of heads
    def sync_frozen_heads(self, model, new_nheads):
        for layer_idx, frozen_heads_list in self.refrigerator.frozen_heads.items():
            valid_frozen_heads = [
                h for h in frozen_heads_list if h < new_nheads]
            self.refrigerator.frozen_heads[layer_idx] = valid_frozen_heads


# %%
# >>integrated: Multi-stage Training<<
embedding_model = BertModel.from_pretrained("/root/autodl-tmp/bert-mini/")
tokenizer = BertTokenizer.from_pretrained("/root/autodl-tmp/bert-mini/")


def tokenize(batch):
    tokens = tokenizer(
        batch['text'],
        truncation=True,
        padding='max_length',
        max_length=256,
        return_attention_mask=True
    )
    tokens['label'] = batch['label']
    return tokens


binary_args = TrainingArguments(
    # *custom
    output_dir="/root/autodl-tmp/KAN_results/stage1_depression",
    evaluation_strategy="steps",
    # save_strategy="no",
    max_steps=1000,
    # num_train_epochs=1,
    # save_steps=600,
    eval_steps=500,
    logging_steps=500,
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.05,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    # save_total_limit=3,
    warmup_steps=200,
    # linear / cosine / cosine with restart
    lr_scheduler_type="cosine_with_restarts",
)


class ContiguousTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        self._save(output_dir)

    def _save(self, output_dir):
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        super()._save(output_dir)

# %%


# ? Does order matter?
dataset_names = ['anxiety', 'ptsd', 'depression', 'bipolar', 'ocd']
datasets = {}

for name in dataset_names:
    dataset = load_dataset(
        'csv', data_files=f"/root/autodl-tmp/Datasets/{name}.csv")['train']
    dataset = dataset.map(tokenize, batched=True).remove_columns(['text'])
    dataset.set_format(type='torch', columns=[
                       'input_ids', 'attention_mask', 'label'])
    split_dataset = dataset.train_test_split(test_size=0.2)
    datasets[name] = (split_dataset['train'], split_dataset['test'])

# %%
# ? Tuning!
min_free_heads = 4
model = TextDiseasePredictor(
    d_model=32, nheads=8, num_classes=2, dropout=0.3, num_transformers=2)
# * ft
freezing_callback = FreezingCallback(freeze_threshold=5e-8, check_interval=10)

first = True
for disease in dataset_names:
    print("/n/n/n Begin training on dataset:", disease)
    train_data, val_data = datasets[disease]

    if hasattr(freezing_callback, 'refrigerator'):
        freezing_callback.refrigerator.model = model

    if first:
        trainer = ContiguousTrainer(
            model=model,
            args=binary_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            callbacks=[freezing_callback],
            compute_metrics=compute_metrics,
        )
        first = False
    else:
        trainer.train_dataset = train_data
        trainer.eval_dataset = val_data
        # trainer.args.max_steps += 500

        # * Reinitialize optimizer and lr_scheduler
        trainer.optimizer = None
        trainer.lr_scheduler = None

    trainer.train()

    freezing_callback.refrigerator.show_frozen_status()

    model = trainer.model

    frozen_heads = len(freezing_callback.refrigerator.frozen_heads[0])
    total_heads = model.transformers[0].self_attn.n_heads
    active_heads = total_heads - frozen_heads

    if active_heads < min_free_heads:
        new_nheads = min_free_heads - active_heads + total_heads
        model.update_attention_heads(new_nheads=new_nheads)
        freezing_callback.sync_frozen_heads(model, new_nheads)

# %%
# * Binary Save Best Model
trainer.save_model("/root/autodl-tmp/FK_12/stagewise_result0")


# %%
# >>Final Stage Fintune and adjust output format - binary 2 multi<<# %%

class MultiLabelWrapper(nn.Module):
    def __init__(self, pretrained_model, num_labels):
        super().__init__()
        self.pretrained_model = pretrained_model  # 获取预训练模型
        hidden_size = pretrained_model.transformers[0].d_model
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, labels=None, attention_mask=None):
        embedded_input = self.pretrained_model.embedding(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        embedded_input = self.pretrained_model.embedding_dim_reduction(
            embedded_input)

        x = embedded_input
        for transformer in self.pretrained_model.transformers:
            x = transformer(x, src_key_padding_mask=attention_mask)
        features = x.mean(dim=1)

        logits = self.classifier(features)
        loss = None
        if labels is not None:
            # criterion = FocalLoss(alpha=0.5, gamma=2.0)
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(logits, labels.float())

        return {"loss": loss, "logits": logits}


# %%
def finetune_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions

    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    probs = torch.sigmoid(logits).detach().cpu().numpy()

    # dynamic threshold
    optimal_thresholds = calculate_optimal_thresholds(labels, probs)
    print(f"Optimal thresholds for each label: {optimal_thresholds}")

    preds = np.zeros_like(probs, dtype=int)
    for i, threshold in enumerate(optimal_thresholds):
        preds[:, i] = (probs[:, i] > threshold).astype(int)

    label_names = ['depression', 'bipolar', 'ptsd', 'ocd', 'anxiety']

    # label-wise evaluation
    per_label_metrics = {}
    for i, label in enumerate(label_names):
        precision = precision_score(labels[:, i], preds[:, i], zero_division=0)
        recall = recall_score(labels[:, i], preds[:, i], zero_division=0)
        f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
        per_label_metrics[f'{label}_f1'] = f1

    print("\nComparing Predictions with True Labels for a few samples:")
    for i in range(20):
        print(f"\nSample {i}:")
        print(f"  True labels: {labels[i]}")
        print(f"  Predicted labels: {preds[i]}")
        exact_match = torch.all(labels[i] == preds[i])
        print(f"  Exact match: {exact_match.item()}")

        # metrics for this sample
        accuracy = accuracy_score(labels[i], preds[i])
        precision = precision_score(labels[i], preds[i])
        recall = recall_score(labels[i], preds[i])
        f1 = f1_score(labels[i], preds[i])

        print(f"  Accuracy: {accuracy}")
        print(f"  Precision: {precision}")
        print(f"  Recall: {recall}")
        print(f"  F1 Score: {f1}")
        print("-" * 30)

    # overall performance
    macro_f1 = f1_score(labels, preds, average='macro')
    micro_f1 = f1_score(labels, preds, average='micro')
    weighted_f1 = f1_score(labels, preds, average='weighted')
    em = accuracy_score(labels, preds)  # 精确匹配率（Exact Match）

    metrics = {
        **per_label_metrics,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'exact_match': em,
    }

    return metrics


# %%
# * New metrics DYNAMIC THRESHOLD<
def calculate_optimal_thresholds(labels, probs):
    thresholds = []
    for i in range(labels.shape[1]):
        precision, recall, threshold = precision_recall_curve(
            labels[:, i], probs[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = f1_scores.argmax()
        thresholds.append(threshold[optimal_idx])
    return thresholds


# %%
# * FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [batch, num_labels]
        # targets: [batch, num_labels]
        probs = torch.sigmoid(logits)  # [batch, num_labels]
        # BCE loss的基础项
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction=self.reduction)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - pt).pow(self.gamma)
        loss = self.alpha * focal_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# %%
tokenizer = BertTokenizer.from_pretrained("/root/autodl-tmp/bert-mini/")
df = pd.read_csv('/root/autodl-tmp/Datasets/multi_label.csv')
df = df.dropna(subset=['text'])
df = df[df['text'].str.strip() != '']

texts = df['text'].astype(str).tolist()
labels = df[["depression", "anxiety", "ocd", "bipolar", "ptsd"]].values

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=1502
)


class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True,
                                   max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = MultiLabelDataset(train_texts, train_labels, tokenizer)
val_dataset = MultiLabelDataset(val_texts, val_labels, tokenizer)

# %%
model0 = TextDiseasePredictor(
    d_model=112, nheads=28, num_classes=2, dropout=0.3, num_transformers=2)
model_weights = load_file(
    "/root/autodl-tmp/FK_12/stagewise_result0/model.safetensors")
model0.load_state_dict(model_weights)
print("model initialized successfully.")

multilabel_model = MultiLabelWrapper(model0, num_labels=5)
training_args = TrainingArguments(
    output_dir='./results',
    save_strategy='no',
    learning_rate=5e-5,
    max_steps=2000,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=100,
    weight_decay=0.2,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    metric_for_best_model="loss",
    greater_is_better=False,
)

finetune_trainer = ContiguousTrainer(
    model=multilabel_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=finetune_metrics
)

finetune_trainer.train()

# %%
finetune_trainer.save_model("/root/autodl-tmp/FK_12/finetune_result0")
print("model saved successcully")

# %%
# * logit lens

predictions = finetune_trainer.predict(val_dataset)

logits = predictions.predictions  # numpy数组
labels = predictions.label_ids    # numpy数组 (num_samples, num_labels)

# 将logits转换为DataFrame
label_names = ['depression', 'bipolar', 'ptsd', 'ocd', 'anxiety']
df_logits = pd.DataFrame(logits, columns=label_names)

# %%
plt.figure(figsize=(10, 6))
sns.kdeplot(df_logits["ocd"], fill=True, color="blue", alpha=0.6)
plt.title("Density Plot of OCD Logits", fontsize=16)
plt.xlabel("Logits", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.grid(True)
plt.show()


# %%
# * self-generated input samples
embedding_model = BertModel.from_pretrained("/root/autodl-tmp/bert-mini/")
tokenizer = BertTokenizer.from_pretrained("/root/autodl-tmp/bert-mini/")


def tokenize(batch):
    tokens = tokenizer(
        batch['text'],
        truncation=True,
        padding='max_length',
        max_length=256,
        return_attention_mask=True
    )
    tokens['label'] = batch['label']
    return tokens


# %%
multi_model = MultiLabelWrapper(TextDiseasePredictor(
    d_model=112, nheads=28, num_classes=2, dropout=0.3, num_transformers=2), num_labels=5)
model_weights = load_file(
    "/root/autodl-tmp/FK_12/finetune_result0/model.safetensors")
multi_model.load_state_dict(model_weights)
# multi_model = ContiguousTrainer.model
multi_model.eval()

# %%
# labels = df[["depression","anxiety","ocd","bipolar","ptsd"]].values
# * Implicit:
# Depression
input_dep = "On paper I have a great life. Beautiful, smart fiancée. Rent an amazing apartment in the downtown of a great city. Have a wonderful dog. A well paying job. Lots of friends. Great family. But I'm always fucking miserable. Mostly during the work week. I don't know what it is. It's like Friday-Sunday I'm feeling great, hang out with friends, do fun stuff with my SO, party, watch sports, whatever. As soon as Monday hits, pretty much every little thing irritates me beyond belief. Stupid things. Basically if my routine or my 'expectations' of my routine get disrupted, I just shut down and I'm annoyed and silent all day. I catch myself in this mood a lot. And then I think about how stupid it is that I'm upset for no reason other than maybe I had to eat lunch 30 minutes later, or I had to walk the dog in the afternoon when I didn't plan on it, or I had to give up our office for my SO to work in for an hour or two. Things that are inconsequential. And then I get mad at myself for being so upset about nothing to the point I can't pull myself out of it. I recognize I'm in these moods but I just can't pull myself out of it. I shouldn't even get moody at these things to begin with."

# OCD
input_ocd = "Why do I find the Spring so depressing? A lot of people talk about the Winter blues, but for me Spring by far (at least in recent years), is painfully depressing.  It's like the smell in the air, mixed with the temperature and longer/brighter daytime present this fake sense of happiness.  It's as if when I'm outside, things seem “too happy” and that scent in the air is gut-wrenchingly nostalgic of a past-time that can never be felt or experienced again. Because I'm not capable of it and I'm too worn-down to, anyway.  I actually remember when I was little (29 now) that I loved clear, sunny weather. I looked up the forecast almost obsessively ahead of time, banking on those days of where there are no clouds to block the sun. Not even partly cloudy.  It’s perplexing to think I was once like this since nowadays, I despise sunny weather. I genuinely feel better and more comfortable when it is cloudy with rain. Especially the eccentric types of weather where it looks as if nighttime has arrived too early, but instead it's just a storm brewing. Not to mention, I am at my peak mindset and performance late at night.  What the hell happened."

# Bipolar
input_bp = "I just want to be happy and to make my partner happy. I don't understand why I'm like this. I love my partner more than anything, yet I struggle to think clearly and communicate effectively. I'm terrified that they're planning against me or will become tired of my episodes and end our relationship, finding me unbearably difficult. I find it hard to reach out for help because I'm unsure how to express my feelings without making them feel accused or thinking I'm losing my sanity. I simply want to gain control over myself, my emotions, and my thoughts. All I desire is to feel happy, to stop causing misery for everyone around me, and to share in the joy that others seem to experience. I feel overwhelmed by fear, anger, and confusion."

# depression, anxiety, ptsd; true 1,1,0,0,1 -> pred 0,1,0,0,1
input_dp_axt_ptsd = "(31f) I hate my life I know it just comes with trauma that I have no idea how to compact, and I feel so behind. I live at home with my mom because of student loans from a degree I had to drop out of because her credit score wasn't good enough, and neither was mine. I now sit with 80k in student debt and only 20k would be gone if Biden finally wipes away student debt.. My mom right now has been more unstable than before. I get it, I'm overweight, I have mental health issues, need some sun and a better job, but it doesn't help when she berates and complains about it daily and comparing me to others. I barely eat as it is, and while she serves unhealthy food as well, she gets mad that I'm not eating healthy and moving like a fucking swan. I'm like 200 lbs full of anxiety, different kinds of odd combinations of grass and veggies in some green smoothies that tastes like eating someone's ass that hasn't showered for 3 years. Still gets mad that I eat unhealthy when she makes it and it's literally all we have. She gets mad that I don't spend time with her at all and prefer to hang out with my friends that are online. She tells me I look ugly and I should look better in clothes that look ugly on me as it is.  Literally, she treats me just like my older brother did minus the sexual abuse I endured for 14 fucking years (which ended when I was 26 by leaving to art school and finally having a way to make it end by severing ties with him (well he did it with me) Being yelled at because I get upset isn't a way to help someone unpack trauma nor help them get motivated about doing better. It's gotten so bad I can't focus on anything very well. I don't even have privacy to go and study to be a data analyst in Coursera because school is really expensive nowadays and i don't have the time to be able to go. I feel really stuck.  And I know many people are gonna say it's procrastination and I get it might be, but it stems from an overflowing  has never stopped. I can't afford therapy because that shit isn't covered, nor can I drive to one because I don't have a car nor do I have the money to pay for an Uber drive weekly along with whatever fee therapy comes with.bi also never have privacy so I can't do at home therapy. I have so little privacy my mom barges in and tries to talk to me even though I tell her I'm in a literal meeting. But if I try to set boundaries or do things myself I'm called an asshole... It's so much thrown at me I feel like I just freeze and just sit and do nothing because that's better than sitting with her and possibly be yelled and berated at for my weight for the umpth time even though she's heavy and diabetic herself. Yeah. My live sucks right now..."

# ocd & otsd; 00101
input_ocd_ptsd = "Hey. Thanks for reading. I am 22M from Europe and I was diagnosed with a significant mood disorder, obsessive tendencies, and high stress last year. I have also spent some time in a mental health care facility and was prescribed medications to manage my condition. The root of my struggles was a series of traumatic embarrassments and mistakes, most of which I caused myself. I've never been great at making decisions, even though I've read countless self-improvement books and other resources. Unfortunately, I've had a hard time retaining and applying what I learned, often acting impulsively or emotionally. To make things even more difficult, I was turned down by someone I deeply admired because of my behavior, which came across as awkward. She also discovered some things about me that I find embarrassing. I’ve been too open and vulnerable with people who didn’t deserve it, and that hurts deeply. I dropped out of high school because my biggest passion was pursuing a music career (rap). However, I've been considering finishing my last year of school and going to college. It's partly because I want a backup plan in case my music career doesn't take off financially, but I also believe it might help with my struggles. My low mood is the real obstacle—it’s keeping me from enjoying the things I love, trapping me in the past, and making me feel intense unease about what others might think of me. I never had this issue before. Is it possible to find joy again and become the athletic, quick-witted, confident, and happy person I was before my life took this turn? I also want to dedicate more time to the gym and writing lyrics, but I lack the drive to start. Can I pull myself out of this dark and challenging place and achieve my dreams? I hate feeling like I'm wasting my life, but I also can't simply “turn off” these feelings. Peace. 22 years old and struggling. Can I regain my confidence and happiness and find success in life?"

# Normal
input_00 = "I am feeling good."

#### -- Group Members' inputs -- ####
# Member_1's input (dont be afraid) (depression)
S_input = "I dream I was haunted by others in an abandoned school. I had to use my redundant brain to think of strategies to excape. But I failed and I fell. Now I just cannot tell the difference between reality and hallucination."
# Member_2's input (exaggerate a little) (depression)
G_input = "Some days I feel pressured, feeling like I need to do something, go somewhere, feeling like my time is so limited. Some days I don't wanna do anything, no showering, no self-care, no eating. But other days when nothing seems to matter, I play games survival games. I want to feel somewhat in control I guess."
# Member_3's input (relax) (anxiety)
M_input = "I feel like there's a constant weight pressing down on me, and even simple tasks feel overwhelming, while my mind keeps racing with worries I can't quiet."

# %%
# Tokenize input
inputs = tokenizer(input_00, return_tensors="pt",
                   truncation=True, padding=True, max_length=256)
input_ids = inputs["input_ids"].to('cuda')
attention_mask = inputs["attention_mask"].to('cuda')
stage3_model = multi_model.to('cuda')
# Forward pass with `output_attentions=True`
outputs = multi_model(input_ids=input_ids, attention_mask=attention_mask)

logits = outputs["logits"]
predictions = torch.argmax(logits, dim=-1)
probabilities = torch.sigmoid(logits)
print("Probabilities:", probabilities)


# %%
