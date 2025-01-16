"""
用于提取文本特征的编码器
   t1_text_encoder.py

Text Encoder 采用带有多头注意力的 Transformer 结构并使用 CLIP 预训练模型的参数。
具体结构是包括：
    一个用于扩增维度和位置编码的Embedding层;
    12个相同结构的带掩码的多头自注意力（Multi-Head Self-Attentiom, MHSA）层;
     最后一层 LayerNorm层

"""

import torch
from torch import nn
from transformers import CLIPTextModel


class Embed(nn.Module):
    def __init__(self, embed_dim=768, n_tokens=77, seq_len=49408):
        super().__init__()
        self.embedding = nn.Embedding(seq_len, embed_dim)
        self.pos_embedding = nn.Embedding(n_tokens, embed_dim)
        self.embed_dim = embed_dim
        self.n_tokens = n_tokens
        self.register_buffer('pos_ids', torch.arange(n_tokens).unsqueeze(0))

    def forward(self, input_ids):  # input_ids: (b, 77)  (b表示batch_size样本数量)
        # Token embedding: [b, 77] -> [b, 77, 768]
        embed = self.embedding(input_ids)
        # Positional embedding: [1, 77] -> [1, 77, 768]   (pos_ids 的 shape 是 [1, 77])
        pos_embed = self.pos_embedding(self.pos_ids)
        # [b, 77, 768]  (这里用到广播机制)
        return embed + pos_embed


class Attention(nn.Module):
    """
    添加注意力信息, 维度不变
    """

    def __init__(self, emb_dim=768, heads=12):
        super().__init__()
        self.wq = nn.Linear(emb_dim, emb_dim)
        self.wk = nn.Linear(emb_dim, emb_dim)
        self.wv = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)  # _proj: projection (投影)
        self.emb_dim = emb_dim  # 嵌入维度 (embedding dimension)
        self.heads = heads  # 多头注意力 (multi-head attention 的头数量)

    def get_mask(self, b, n_token):
        # 构造一个上三角的掩码矩阵，用于屏蔽后续的 token
        mask = torch.empty(b, n_token, n_token)  # 创建一个空的 [b, n_token, n_token] 张量
        mask.fill_(-float('inf'))  # 将所有元素填充为 -inf（屏蔽的默认值）
        # 取上三角部分，屏蔽后续 token，并添加一维以兼容
        mask.triu_(1).unsqueeze(1)  # 对角线及以下的位置为0
        return mask

    def forward(self, x):
        # in (b, 77, 768)  == (batch_size, 序列长度n_tok, 嵌入维度)
        b, n_token, _ = x.shape  # 获取批大小 b 和序列长度 n_token
        q: torch.Tensor = self.wq(x) / 8
        k: torch.Tensor = self.wk(x)
        v: torch.Tensor = self.wv(x)
        # 拆分为12个头, 并转换维度 (主要是emb_dim768拆成12份
        # q,k,v [b, 77, 768]->[b, 77, 12, 64]->[b, 12, 77, 64]->[b*12, 77, 64]
        q = (q.reshape(b, n_token, self.heads, self.emb_dim // self.heads)
             .transpose(1, 2).reshape(b * self.heads, n_token, self.emb_dim // self.heads))
        k = (k.reshape(b, n_token, self.heads, self.emb_dim // self.heads)
             .transpose(1, 2).reshape(b * self.heads, n_token, self.emb_dim // self.heads))
        v = (v.reshape(b, n_token, self.heads, self.emb_dim // self.heads)
             .transpose(1, 2).reshape(b * self.heads, n_token, self.emb_dim // self.heads))
        # 计算q,k乘积, qk关系矩阵 # bmm: batch matrix-matrix product
        # bmm仅支持三维: (batch_size, m, n), b个AB^T, 返回的是batch_size个结果矩阵
        attn = torch.bmm(q, k.transpose(1, 2))  # [b*12, 77, 77]
        # 调整形状以便添加掩码
        attn = attn.reshape(b, self.heads, n_token, n_token)  # [b, 12, 77, 77]
        # 添加掩码，屏蔽无效位置(未来token)
        # 广播 [b, 12, 77, 77] + [b, 1, 77, 77] -> [b, 12, 77, 77]
        attn = attn + self.get_mask(b, n_token).to(attn.device)
        # 恢复注意力得分的形状
        attn = attn.reshape(b * self.heads, n_token, n_token)  # [b*12, 77, 77]
        # 对最后一个维度（每个 token 的权重）进行归一化,  ,被mask的项经softmax由-inf->0
        attn = attn.softmax(dim=-1)  # [b*12, 77, 77]形状不变
        # 计算与v的乘积, attn=softmax(qk/sqrt(dk)) v
        # 一共batch_size*12个头个注意力矩阵: [77, 77] * [77, 64] -> [77, 64]
        attn = torch.bmm(attn, v)  # [b*12, 77, 64]
        # 调整形状以合并多个头的输出 (最后两维12*64=>768)
        # [b*12, 77, 64]->[b, 12, 77, 64]->[b, 77, 12, 64]->[b, 77, 768]
        attn = (attn.reshape(b, self.heads, n_token, self.emb_dim // self.heads)
                .transpose(1, 2).reshape(b, n_token, self.emb_dim))
        # 多头输出合并, 投影维度不变 [b, 77, 768]
        return self.out_proj(attn)


# class QuickGELU(nn.Module):
#     """
#     自定义的激活函数, 类似于标准的 GELU (Gaussian Error Linear Unit)
#     但计算更快，且有近似的效果。
#     """
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return (x * 1.702).sigmoid() * x


class ClipEncoder(nn.Module):
    """
    多模态表达信息嵌入, 维度不变 (尝试学习深层表达能力)

    ClipEncoder是 CLIP 模型中的编码器模块之一，用于将输入数据（如文本或图像）转换为多模态的嵌入表示。
    """

    def __init__(self, embed_dim=768, expand_dim=3072):
        super().__init__()
        self.seq1 = nn.Sequential(
                nn.LayerNorm(embed_dim),  # 对输入特征的最后一维进行归一化
                Attention())  # 添加注意力信息, 维度不变
        self.seq2 = nn.Sequential(
                nn.LayerNorm(embed_dim),  # 对输入特征的最后一维进行归一化
                nn.Linear(embed_dim, expand_dim))  # fcl, 将最后一个维度768->3072
        self.seq3 = nn.Linear(expand_dim, embed_dim)  # fcl, 将最后一个维度3072->768

    def forward(self, x):
        # 残差连接1
        x = x + self.seq1(x)
        # 残差连接2
        res = x
        x = self.seq2(x)  # [2,77,768]->[2,77,3072]
        x = x * (x * 1.702).sigmoid()  # QuickGELU(),  # 激活x-> (x * 1.702).sigmoid() * x
        return res + self.seq3(x)  # [2,77,3072]->[2,77,768]

        # 这里解释下,最后为什么要先扩展,然后激活后再缩回:
        # 这种设计主要来源于 Transformer 架构中常用的 Feed-Forward Neural Network (FFN) 子模块
        # 扩展是为了临时增加表达能力, 激活函数在更高维空间中能捕获到更多的非线性关系。
        # 扩展维度后，神经网络可以学习更复杂、更丰富的特征;
        # 再缩回来, 主要是考虑成本, 同时也为了能加上初始x残差


class TextEncoder(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()

        # seq = [Embed()] + [ClipEncoder() for _ in range(12)] + [nn.LayerNorm(embed_dim)]
        # self.seq = nn.Sequential(*seq)

        # 1+12+1 共14层
        self.seq = nn.Sequential(
                Embed(),
                ClipEncoder(),
                ClipEncoder(),
                ClipEncoder(),
                ClipEncoder(),
                ClipEncoder(),
                ClipEncoder(),
                ClipEncoder(),
                ClipEncoder(),
                ClipEncoder(),
                ClipEncoder(),
                ClipEncoder(),
                ClipEncoder(),
                nn.LayerNorm(embed_dim),  # 768
        )

    def forward(self, x):
        out = self.seq(x)
        return out


def load_pretrained_text_encoder(textEncoder: TextEncoder):
    params = CLIPTextModel.from_pretrained('./pretrained-params', subfolder='text_encoder')
    # 第一层: 词编码器 & 位置编码器的参数加载
    embed: Embed = textEncoder.seq[0]
    embed.embedding.load_state_dict(
            params.text_model.embeddings.token_embedding.state_dict())
    embed.pos_embedding.load_state_dict(
            params.text_model.embeddings.position_embedding.state_dict())

    # 12层的ClipEncoder
    for i in range(12):
        clipEncoder: ClipEncoder = textEncoder.seq[i + 1]
        # 第1层LayerNorm
        clipEncoder.seq1[0].load_state_dict(
                params.text_model.encoder.layers[i].layer_norm1.state_dict())
        # 注意力q矩阵
        clipEncoder.seq1[1].wq.load_state_dict(
                params.text_model.encoder.layers[i].self_attn.q_proj.state_dict())
        # 注意力k矩阵
        clipEncoder.seq1[1].wk.load_state_dict(
                params.text_model.encoder.layers[i].self_attn.k_proj.state_dict())
        # 注意力v矩阵
        clipEncoder.seq1[1].wv.load_state_dict(
                params.text_model.encoder.layers[i].self_attn.v_proj.state_dict())
        # 注意力out
        clipEncoder.seq1[1].out_proj.load_state_dict(
                params.text_model.encoder.layers[i].self_attn.out_proj.state_dict())

        # 第2层LayerNorm
        clipEncoder.seq2[0].load_state_dict(
                params.text_model.encoder.layers[i].layer_norm2.state_dict())
        # mlp第1层fc
        clipEncoder.seq2[1].load_state_dict(
                params.text_model.encoder.layers[i].mlp.fc1.state_dict())

        # mlp第2层fc
        clipEncoder.seq3.load_state_dict(
                params.text_model.encoder.layers[i].mlp.fc2.state_dict())

    # 最后一层, 输出layerNorm
    textEncoder.seq[13].load_state_dict(params.text_model.final_layer_norm.state_dict())

    return textEncoder
