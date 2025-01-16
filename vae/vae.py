"""
用于提取视觉特征的编码器 VAE
  t2_vae.py

Image Encoder 使用的模型是VAE（Variational Autoencoder），
由 ResNet 块结构和 少量的 自注意力（Self-Attention）层堆叠构成 的一个Encoder
和一个 Decoder 组成。
其中，所有标准化层和激活函数是 GroupNorm 标准化层 和 SiLU 激活函数。

关于encoder的输出是(1,8,64,64) 采样噪声(1,4,64,64).

VAE 的设计规范：这是标准的 Variational Autoencoder 实现方式，
必须包含均值和对数方差来实现重参数化技巧。
划分方式（前4和后4）通常由开发者在模型设计时决定，并在训练代码中隐式实现。
例如，训练过程中可能通过 slice 操作对编码器的输出进行均值和对数方差的拆分。
"""

import torch
from diffusers import AutoencoderKL
from torch import nn


class ResNetBlock(nn.Module):
    """
    残差块
    在两轮 " 组归一化、激活、卷积" 操作后, 拼接一次残差.
    作用: 如果网络过浅，无法提取复杂特征，模型性能有限, 但是深层网络可能难以训练，甚至性能退化
        这种残差结构让网络在保留输入特征的同时，能够逐层学习到额外的特征变换，
        使得深层网络既能提取丰富的特征，又不会出现退化现象
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.seq = nn.Sequential(
                # 将dim_in个通道分成 32 组, eps避免除零, affine=True: 允许学习可训练的缩放和平移参数
                nn.GroupNorm(num_groups=32, num_channels=dim_in, eps=1e-6, affine=True),
                nn.SiLU(),  # 激活函数, 改进 ReLU 的效果
                # padding=1: 边界填充使得输出尺寸与输入保持一致
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
                # 将dim_out个通道分成 32 组, eps避免除零, affine=True: 允许学习可训练的缩放和平移参数
                nn.GroupNorm(num_groups=32, num_channels=dim_out, eps=1e-6, affine=True),
                nn.SiLU(),
                # padding=1: 边界填充使得输出尺寸与输入保持一致
                nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1))

        # 计算残差x+f(x)时, 如果in,out维度不同, 得将残差统一成out的shape
        self.conv_shortcut = nn.Identity() if dim_in == dim_out \
            else nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1)  # 统一shape

    def forward(self, x):
        res = self.conv_shortcut(x)
        return res + self.seq(x)


class Attention(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.norm = nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6, affine=True)
        self.wq = torch.nn.Linear(embed_dim, embed_dim)
        self.wk = torch.nn.Linear(embed_dim, embed_dim)
        self.wv = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (b, 512, 64, 64) 对应(batch_size, channel, height, width)
        b, c, h, w = x.shape
        res = x
        x = self.norm(x)
        # [b, 512, 64, 64] -> [b, 512, 4096] -> [b, 4096, 512]
        x = x.flatten(start_dim=2).transpose(1, 2)
        # q,k,v (b, 4096, 512)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        # [b, 4096, 512] * [b, 512, 4096] -> [b, 4096, 4096]
        # 这里的 0.044194173824159216 = 1 / (512**0.5)
        attn = q.bmm(k.transpose(1, 2)) * 0.044194173824159216

        attn = torch.softmax(attn, dim=2)
        # [b, 4096, 4096] * [b, 4096, 512] -> [b, 4096, 512]
        attn = attn.bmm(v)
        # 线性投影,维度不变 [b, 4096, 512]
        attn = self.out_proj(attn)
        # [b, 4096, 512] -> [b, 512, 4096] -> [b, 512, 64, 64]
        attn = attn.transpose(1, 2).reshape(b, c, h, w)
        attn = attn + res
        return attn


class Pad(nn.Module):
    """
    增加:底下一行与右侧一列0
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.pad(x, (0, 1, 0, 1),  # (左，右，上，下)：右侧+1列，下面+1行
                                 # x的最后两个维度看作是一个mn行列矩阵, 对应高宽
                                 mode='constant',  # 填充值为常数
                                 value=0)  # 常数值为0


class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        # 编码器定义
        self.encoder = nn.Sequential(
                # in 输入卷积层：输入通道数为3，输出通道数为128，卷积核大小为3x3，步长为1，填充为1
                nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),

                # down1 (第1阶段下采样)
                nn.Sequential(
                        ResNetBlock(128, 128),  # 维度不变
                        ResNetBlock(128, 128),  # 维度不变
                        nn.Sequential(
                                Pad(),  # 自定义填充层：右侧与下方, 增加一行一列的0
                                # 下采样，步长为2 (Padding用刚刚自定义的)
                                torch.nn.Conv2d(128, 128, 3, stride=2, padding=0),
                        ),
                ),
                # down2 (第2阶段下采样)
                torch.nn.Sequential(
                        ResNetBlock(128, 256),  # 通道数从128变为256
                        ResNetBlock(256, 256),  # 维度不变
                        torch.nn.Sequential(
                                Pad(),  # 自定义填充层：右侧与下方, 增加一行一列的0
                                # 下采样，步长为2 (Padding用刚刚自定义的)
                                nn.Conv2d(256, 256, 3, stride=2, padding=0),
                        ),
                ),
                # down3 (第3阶段下采样)
                nn.Sequential(
                        ResNetBlock(256, 512),  # 通道数从256变为512
                        ResNetBlock(512, 512),  # 维度不变
                        nn.Sequential(
                                Pad(),  # 自定义填充层：右侧与下方, 增加一行一列的0
                                # 下采样，步长为2 (Padding用刚刚自定义的)
                                nn.Conv2d(512, 512, 3, stride=2, padding=0),
                        ),
                ),
                # 编码器最后阶段，进一步处理高维特征
                nn.Sequential(
                        ResNetBlock(512, 512),  # 维度不变
                        ResNetBlock(512, 512),  # 维度不变
                ),
                # mid (中间处理阶段)
                nn.Sequential(
                        ResNetBlock(512, 512),  # 维度不变
                        Attention(),  # 注意力机制，增强重要特征
                        ResNetBlock(512, 512),  # 维度不变
                ),
                # encoder out (输出阶段)
                nn.Sequential(
                        nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6),  # 分组归一化
                        nn.SiLU(),
                        # 卷积层，输出通道数为8
                        nn.Conv2d(512, 8, 3, padding=1),
                ),
                # 正态分布层 (用于生成正态分布参数的最后一层)   # 1×1 通道数不变
                nn.Conv2d(8, 8, 1))

        # 解码器定义
        self.decoder = nn.Sequential(

                # 正态分布层 (用于从正态分布参数恢复特征的卷积层)
                nn.Conv2d(4, 4, 1),

                # in (输入阶段，初步处理特征)  通道数增加: 4->512
                nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1),

                # middle (中间处理阶段)
                nn.Sequential(
                        ResNetBlock(512, 512),  # 维度不变
                        Attention(),  # 注意力机制
                        ResNetBlock(512, 512)),  # 维度不变

                # up1 (第1阶段上采样)
                nn.Sequential(
                        ResNetBlock(512, 512),  # 维度不变
                        ResNetBlock(512, 512),  # 维度不变
                        ResNetBlock(512, 512),  # 维度不变
                        # 将输入特征图的宽度和高度放大为原来的 2 倍
                        # (mode='nearest' 表示使用最近邻插值法进行上采样)
                        nn.Upsample(scale_factor=2.0, mode='nearest'),  # 上采样，尺寸扩大2倍
                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                ),
                # up2 (第2阶段上采样)
                nn.Sequential(
                        ResNetBlock(512, 512),  # 维度不变
                        ResNetBlock(512, 512),  # 维度不变
                        ResNetBlock(512, 512),  # 维度不变
                        nn.Upsample(scale_factor=2.0, mode='nearest'),  # 上采样，尺寸扩大2倍
                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                ),
                # up3 (第3阶段上采样)
                nn.Sequential(
                        ResNetBlock(512, 256),  # 通道数从512降至256
                        ResNetBlock(256, 256),  # 维度不变
                        ResNetBlock(256, 256),  # 维度不变
                        nn.Upsample(scale_factor=2.0, mode='nearest'),  # 上采样，尺寸扩大2倍
                        nn.Conv2d(256, 256, kernel_size=3, padding=1),
                ),
                # 输出准备
                nn.Sequential(
                        ResNetBlock(256, 128),  # 通道数从256降至128
                        ResNetBlock(128, 128),  # 维度不变
                        ResNetBlock(128, 128),  # 维度不变
                ),
                # decoder out (最终的解码器输出)
                nn.Sequential(
                        # 组归一化
                        nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6),
                        nn.SiLU(),
                        # 卷积层，输出通道数为3，与encoder的输入一致
                        nn.Conv2d(128, 3, 3, padding=1),
                ))

    def sample(self, h):
        # eg: h -> [1, 8, 64, 64] 包含了均值（mean）和对数方差（logvar）的信息。
        # [1, 8, 64, 64] 切片取[1, 4(前4), 64, 64]
        mean = h[:, :4]  # 对Tensor的前两个维度进行切片操作, 每个维度切片中m:n表示[m,n-1]
        # [1, 8, 64, 64] 切片取[1, 4(后4), 64, 64]
        logvar = h[:, 4:]  # 对数的方差
        std = logvar.exp() ** 0.5  # exp后开方, 得到->标准差
        # [1, 4, 64, 64]  构造一个高维正态分布  N(mean, std^2)
        noise = torch.randn(mean.shape, device=mean.device)
        # 重参数化 (利用公式z = μ + σ·N(0,1) )
        # 将标准正态分布的随机噪声 noise 映射到分布 N(mean, std^2)
        noise = mean + std * noise
        return noise

    def forward(self, x):
        # x : [1, 3, 512, 512]-> [1, 8, 64, 64]
        h = self.encoder(x)
        # [1, 8, 64, 64] 里一半存均值信息,一半存对数方差, 都是[1, 4, 64, 64]
        # 生成一个分布然后采样 -> [1, 4, 64, 64]
        h = self.sample(h)
        # [1, 4, 64, 64] -> [1, 3, 512, 512]
        h = self.decoder(h)
        return h


def load_res(res_block: ResNetBlock, param):
    res_block.seq[0].load_state_dict(param.norm1.state_dict())
    res_block.seq[2].load_state_dict(param.conv1.state_dict())
    res_block.seq[3].load_state_dict(param.norm2.state_dict())
    res_block.seq[5].load_state_dict(param.conv2.state_dict())
    if isinstance(res_block.conv_shortcut, nn.Conv2d):
        res_block.conv_shortcut.load_state_dict(param.conv_shortcut.state_dict())


def load_attn(attention: Attention, param):
    attention.norm.load_state_dict(param.group_norm.state_dict())
    attention.wq.load_state_dict(param.to_q.state_dict())
    attention.wk.load_state_dict(param.to_k.state_dict())
    attention.wv.load_state_dict(param.to_v.state_dict())
    attention.out_proj.load_state_dict(param.to_out[0].state_dict())


def load_pretrained_vae(vae: VAE):
    params: AutoencoderKL = AutoencoderKL.from_pretrained('./pretrained-params/', subfolder='vae')
    # encoder
    # encoder.in 输入层
    vae.encoder[0].load_state_dict(params.encoder.conv_in.state_dict())
    # encoder.3个down_sampler块,以及输出块, 这些块中前两个seq均是resnet块
    for i in range(4):
        load_res(vae.encoder[i + 1][0], params.encoder.down_blocks[i].resnets[0])
        load_res(vae.encoder[i + 1][1], params.encoder.down_blocks[i].resnets[1])
        if i != 3:  # 前3个下采样块的第三个seq的第二个seq(用于下采样的cnn)
            vae.encoder[i + 1][2][1].load_state_dict(
                    params.encoder.down_blocks[i].downsamplers[0].conv.state_dict())
    # encoder.mid
    load_res(vae.encoder[5][0], params.encoder.mid_block.resnets[0])
    load_res(vae.encoder[5][2], params.encoder.mid_block.resnets[1])
    load_attn(vae.encoder[5][1], params.encoder.mid_block.attentions[0])
    # encoder.out
    vae.encoder[6][0].load_state_dict(params.encoder.conv_norm_out.state_dict())
    vae.encoder[6][2].load_state_dict(params.encoder.conv_out.state_dict())
    # encoder.正态分布层
    vae.encoder[7].load_state_dict(params.quant_conv.state_dict())

    # decoder
    # decoder.正态分布层
    vae.decoder[0].load_state_dict(params.post_quant_conv.state_dict())
    # decoder.in
    vae.decoder[1].load_state_dict(params.decoder.conv_in.state_dict())
    # decoder.mid
    load_res(vae.decoder[2][0], params.decoder.mid_block.resnets[0])
    load_res(vae.decoder[2][2], params.decoder.mid_block.resnets[1])
    load_attn(vae.decoder[2][1], params.decoder.mid_block.attentions[0])
    # decoder.up
    for i in range(4):
        load_res(vae.decoder[i + 3][0], params.decoder.up_blocks[i].resnets[0])
        load_res(vae.decoder[i + 3][1], params.decoder.up_blocks[i].resnets[1])
        load_res(vae.decoder[i + 3][2], params.decoder.up_blocks[i].resnets[2])
        if i != 3:
            vae.decoder[i + 3][4].load_state_dict(
                    params.decoder.up_blocks[i].upsamplers[0].conv.state_dict())
    # decoder.out
    vae.decoder[7][0].load_state_dict(params.decoder.conv_norm_out.state_dict())
    vae.decoder[7][2].load_state_dict(params.decoder.conv_out.state_dict())
    return vae
