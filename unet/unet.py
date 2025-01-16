import os
import re

import torch
from diffusers import UNet2DConditionModel
from torch import nn


class ResNetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, time_emb_dim=1280):
        super().__init__()

        self.time_emb = nn.Sequential(
                nn.SiLU(),
                torch.nn.Linear(time_emb_dim, dim_out),
                nn.Unflatten(dim=1, unflattened_size=(dim_out, 1, 1)),
        )
        self.seq1 = nn.Sequential(
                nn.GroupNorm(num_groups=32, num_channels=dim_in, eps=1e-5, affine=True),
                nn.SiLU(),
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1))
        self.seq2 = nn.Sequential(
                nn.GroupNorm(num_groups=32, num_channels=dim_out, eps=1e-5, affine=True),
                nn.SiLU(),
                nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1))
        # 残差对齐, 仅改变通道
        self.conv_shortcut = nn.Identity() if dim_in == dim_out \
            else nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1)

    def forward(self, x, time):
        # x:[1, 320, 32, 32] # 假设dim_out=640!=dim_in=320
        res = self.conv_shortcut(x)  # -> res[1, 640, 32, 32]
        time = self.time_emb(time)  # [1, 1280] -> [1, 640, 1, 1]

        x = self.seq1(x) + time  # [1, 320, 32, 32] -> [1, 640, 32, 32] 广播
        x = res + self.seq2(x)  # [1, 640, 32, 32] 维度不变
        return x


class CrossAttention(nn.Module):
    """
    图&文 交叉注意力
    """

    def __init__(self, dim_q=320, dim_kv=768, heads=8):
        super().__init__()

        self.dim_q = dim_q  # 320
        self.heads = heads  # 8
        self.wq = nn.Linear(dim_q, dim_q, bias=False)  # mlp(320,320)
        self.wk = nn.Linear(dim_kv, dim_q, bias=False)  # mlp(768,320)
        self.wv = nn.Linear(dim_kv, dim_q, bias=False)  # mlp(768,320)
        self.out_proj = nn.Linear(dim_q, dim_q)  # mlp(320,320)

    def multi_head_reshape(self, x):
        b, lens, dim = x.shape  # [1, 4096, 320]
        # [1, 4096, 320] -> [1, 4096, 8, 40]-> [1, 8, 4096, 40]-> [8, 4096, 40]
        x = (x.reshape(b, lens, self.heads, dim // self.heads).transpose(1, 2)
             .reshape(b * self.heads, lens, dim // self.heads))
        return x

    def multi_head_reshape_inverse(self, x):
        b, lens, dim = x.shape  # [8, 4096, 40]
        # [8, 4096, 40] -> [1, 8, 4096, 40]-> [1, 4096, 8, 40]->[1, 4096, 320]
        x = (x.reshape(b // self.heads, self.heads, lens, dim).transpose(1, 2)
             .reshape(b // self.heads, lens, dim * self.heads))
        return x

    def forward(self, q, kv):
        # q:[1, 4096, 320], kv: [1, 77, 768], wq: mlp(320,320), wk,wv: mlp(768,320)
        q = self.wq(q)  # [1, 4096, 320] -> [1, 4096, 320]
        k = self.wk(kv)  # [1, 77, 768] -> [1, 77, 320]
        v = self.wv(kv)  # [1, 77, 768] -> [1, 77, 320]
        q = self.multi_head_reshape(q)  # [1, 4096, 320] -> [8, 4096, 40]
        k = self.multi_head_reshape(k)  # [1, 77, 320] -> [8, 77, 40]
        v = self.multi_head_reshape(v)  # [1, 77, 320] -> [8, 77, 40]
        # [8, 4096, 40] * [8, 40, 77] -> [8, 4096, 77]
        attn = q.bmm(k.transpose(1, 2)) * ((self.dim_q // self.heads) ** (-0.5))
        attn = attn.softmax(dim=-1)
        attn = attn.bmm(v)  # [8, 4096, 77] * [8, 77, 40] -> [8, 4096, 40]
        attn = self.multi_head_reshape_inverse(attn)  # [8, 4096, 40] -> [1, 4096, 320]
        attn = self.out_proj(attn)  # [1, 4096, 320] -> [1, 4096, 320]
        return attn


class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # in
        self.norm_in = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
        self.cnn_in = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        # attn
        self.norm_attn0 = nn.LayerNorm(dim, elementwise_affine=True)
        self.attn1 = CrossAttention(dim, dim)
        self.norm_attn1 = nn.LayerNorm(dim, elementwise_affine=True)
        self.attn2 = CrossAttention(dim, 768)
        # act
        self.norm_act = nn.LayerNorm(dim, elementwise_affine=True)
        self.fc0 = nn.Linear(dim, dim * 8)
        self.act = nn.GELU()
        self.fc1 = nn.Linear(dim * 4, dim)
        # out
        self.cnn_out = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, q, kv):
        # q:[1, 320, 64, 64], kv:[1, 77, 768]
        b, _, h, w = q.shape
        res1 = q
        # ----in----
        q = self.cnn_in(self.norm_in(q))  # [1, 320, 64, 64] 维度不变
        # [1, 320, 64, 64] -> [1, 64, 64, 320] -> [1, 4096, 320]
        q = q.permute(0, 2, 3, 1).reshape(b, h * w, self.dim)
        # ----attn----
        # [1, 4096, 320] 维度不变
        q = self.attn1(q=self.norm_attn0(q), kv=self.norm_attn0(q)) + q
        q = self.attn2(q=self.norm_attn1(q), kv=kv) + q
        # ----act----
        res2 = q  # [1, 4096, 320]
        q = self.fc0(self.norm_act(q))  # [1, 4096, 320] -> [1, 4096, 2560]
        d = q.shape[2] // 2  # d:(1280)
        # 矩阵一分为二,逐点相乘 -> [1, 4096, 1280] 维度不变
        q = q[:, :, :d] * self.act(q[:, :, d:])
        q = self.fc1(q) + res2  # fc1([1, 4096, 1280]) -> [1, 4096, 320]
        # ----out----
        # [1, 4096, 320] -> [1, 64, 64, 320] -> [1, 320, 64, 64]
        # ( contiguous() 将“非连续”的张量转换为物理上连续存储的新张量,安全 )
        q = q.reshape(b, h, w, self.dim).permute(0, 3, 1, 2).contiguous()
        q = self.cnn_out(q) + res1  # [1, 320, 64, 64] 维度不变
        return q


class DownBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.trans_block1 = TransformerBlock(dim_out)
        self.res_block1 = ResNetBlock(dim_in, dim_out)

        self.trans_block2 = TransformerBlock(dim_out)
        self.res_block2 = ResNetBlock(dim_out, dim_out)

        self.out = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=2, padding=1)

    def forward(self, vae_out, text_out, time):
        # dim_in:320, dim_out:640
        # vae:(1, 320, 32, 32),text:(1, 77, 768),time:(1, 1280)
        outs = []
        vae_out = self.res_block1(vae_out, time)  # (1, 320, 32, 32)->(1, 64, 32, 32)
        # q:vae_out(1, 64, 32, 32), kv:text_out(1, 77, 768)
        vae_out = self.trans_block1(vae_out, text_out)  # ->[1, 320, 64, 64]
        outs.append(vae_out)

        vae_out = self.res_block2(vae_out, time)
        vae_out = self.trans_block2(vae_out, text_out)
        outs.append(vae_out)

        vae_out = self.out(vae_out)  # [1, 640, 16, 16]
        outs.append(vae_out)

        return vae_out, outs


class UpBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_prev, add_up):
        super().__init__()

        self.res_block1 = ResNetBlock(dim_out + dim_prev, dim_out)
        self.res_block2 = ResNetBlock(dim_out + dim_out, dim_out)
        self.res_block3 = ResNetBlock(dim_in + dim_out, dim_out)

        self.trans_block1 = TransformerBlock(dim_out)
        self.trans_block2 = TransformerBlock(dim_out)
        self.trans_block3 = TransformerBlock(dim_out)

        self.out = torch.nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1),
        ) if add_up else nn.Identity()

    def forward(self, vae_out, text_out, time, out_down):
        vae_out = self.res_block1(torch.cat([vae_out, out_down.pop()], dim=1), time)
        vae_out = self.trans_block1(vae_out, text_out)

        vae_out = self.res_block2(torch.cat([vae_out, out_down.pop()], dim=1), time)
        vae_out = self.trans_block2(vae_out, text_out)

        vae_out = self.res_block3(torch.cat([vae_out, out_down.pop()], dim=1), time)
        vae_out = self.trans_block3(vae_out, text_out)

        vae_out = self.out(vae_out)
        return vae_out


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # in
        self.in_vae = torch.nn.Conv2d(4, 320, kernel_size=3, padding=1)
        self.in_time = torch.nn.Sequential(
                torch.nn.Linear(320, 1280),
                torch.nn.SiLU(),
                torch.nn.Linear(1280, 1280),
        )
        # down
        self.down_block1 = DownBlock(320, 320)
        self.down_block2 = DownBlock(320, 640)
        self.down_block3 = DownBlock(640, 1280)

        self.down_res1 = ResNetBlock(1280, 1280)
        self.down_res2 = ResNetBlock(1280, 1280)
        # mid
        self.mid_res1 = ResNetBlock(1280, 1280)
        self.mid_trans = TransformerBlock(1280)
        self.mid_res2 = ResNetBlock(1280, 1280)
        # up
        self.up_res1 = ResNetBlock(2560, 1280)
        self.up_res2 = ResNetBlock(2560, 1280)
        self.up_res3 = ResNetBlock(2560, 1280)

        self.up_in = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='nearest'),
                torch.nn.Conv2d(1280, 1280, kernel_size=3, padding=1),
        )

        self.up_block1 = UpBlock(640, 1280, 1280, True)
        self.up_block2 = UpBlock(320, 640, 1280, True)
        self.up_block3 = UpBlock(320, 320, 640, False)
        # out
        self.out = torch.nn.Sequential(
                torch.nn.GroupNorm(num_channels=320, num_groups=32, eps=1e-5),
                torch.nn.SiLU(),
                torch.nn.Conv2d(320, 4, kernel_size=3, padding=1),
        )
        # self.load_pretrained()

    def get_time_embed(self, t):
        # -9.210340371976184 = -math.log(10000)
        e = torch.arange(160) * -9.210340371976184 / 160
        e = e.exp().to(t.device) * t
        # [160+160] -> [320] -> [1, 320]
        e = torch.cat([e.cos(), e.sin()]).unsqueeze(dim=0)
        return e

    def forward(self, vae_out, text_out, time):
        # vae_out -> [1, 4, 64, 64]
        # out_encoder -> [1, 77, 768]
        # time -> [1]
        # ----in----
        # [1, 4, 64, 64] -> [1, 320, 64, 64]
        vae_out = self.in_vae(vae_out)
        # [1] -> [1, 320]
        time = self.get_time_embed(time)
        # [1, 320] -> [1, 1280]
        time = self.in_time(time)

        # ----down----
        # [1, 320, 64, 64]
        # [1, 320, 64, 64]
        # [1, 320, 64, 64]
        # [1, 320, 32, 32]
        # [1, 640, 32, 32]
        # [1, 640, 32, 32]
        # [1, 640, 16, 16]
        # [1, 1280, 16, 16]
        # [1, 1280, 16, 16]
        # [1, 1280, 8, 8]
        # [1, 1280, 8, 8]
        # [1, 1280, 8, 8]
        out_down = [vae_out]
        # [1, 320, 64, 64],[1, 77, 768],[1, 1280] -> [1, 320, 32, 32]
        # out -> [1, 320, 64, 64],[1, 320, 64, 64][1, 320, 32, 32]
        vae_out, out = self.down_block1(vae_out=vae_out, text_out=text_out, time=time)
        out_down.extend(out)
        # [1, 320, 32, 32],[1, 77, 768],[1, 1280] -> [1, 640, 16, 16]
        # out -> [1, 640, 32, 32],[1, 640, 32, 32],[1, 640, 16, 16]
        vae_out, out = self.down_block2(vae_out=vae_out, text_out=text_out, time=time)
        out_down.extend(out)
        # [1, 640, 16, 16],[1, 77, 768],[1, 1280] -> [1, 1280, 8, 8]
        # out -> [1, 1280, 16, 16],[1, 1280, 16, 16],[1, 1280, 8, 8]
        vae_out, out = self.down_block3(vae_out=vae_out, text_out=text_out, time=time)
        out_down.extend(out)
        # [1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        vae_out = self.down_res1(vae_out, time)
        out_down.append(vae_out)
        # [1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        vae_out = self.down_res2(vae_out, time)
        out_down.append(vae_out)

        # ----mid----
        # [1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        vae_out = self.mid_res1(vae_out, time)
        # [1, 1280, 8, 8],[1, 77, 768] -> [1, 1280, 8, 8]
        vae_out = self.mid_trans(vae_out, text_out)
        # [1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        vae_out = self.mid_res2(vae_out, time)

        # ----up----
        # [1, 1280+1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        vae_out = self.up_res1(torch.cat([vae_out, out_down.pop()], dim=1), time)
        # [1, 1280+1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        vae_out = self.up_res2(torch.cat([vae_out, out_down.pop()], dim=1), time)
        # [1, 1280+1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        vae_out = self.up_res3(torch.cat([vae_out, out_down.pop()], dim=1), time)
        # [1, 1280, 8, 8] -> [1, 1280, 16, 16]
        vae_out = self.up_in(vae_out)
        # [1, 1280, 16, 16],[1, 77, 768],[1, 1280] -> [1, 1280, 32, 32]
        # out_down -> [1, 640, 16, 16],[1, 1280, 16, 16],[1, 1280, 16, 16]
        vae_out = self.up_block1(vae_out=vae_out, text_out=text_out, time=time, out_down=out_down)
        # [1, 1280, 32, 32],[1, 77, 768],[1, 1280] -> [1, 640, 64, 64]
        # out_down -> [1, 320, 32, 32],[1, 640, 32, 32],[1, 640, 32, 32]
        vae_out = self.up_block2(vae_out=vae_out, text_out=text_out, time=time, out_down=out_down)
        # [1, 640, 64, 64],[1, 77, 768],[1, 1280] -> [1, 320, 64, 64]
        # out_down -> [1, 320, 64, 64],[1, 320, 64, 64],[1, 320, 64, 64]
        vae_out = self.up_block3(vae_out=vae_out, text_out=text_out, time=time, out_down=out_down)
        # ----out----
        # [1, 320, 64, 64] -> [1, 4, 64, 64]
        vae_out = self.out(vae_out)
        return vae_out


# transformer块参数加载
def __load_transformer_block(model: TransformerBlock, param):
    model.norm_in.load_state_dict(param.norm.state_dict())
    model.cnn_in.load_state_dict(param.proj_in.state_dict())

    model.attn1.wq.load_state_dict(param.transformer_blocks[0].attn1.to_q.state_dict())
    model.attn1.wk.load_state_dict(param.transformer_blocks[0].attn1.to_k.state_dict())
    model.attn1.wv.load_state_dict(param.transformer_blocks[0].attn1.to_v.state_dict())
    model.attn1.out_proj.load_state_dict(param.transformer_blocks[0].attn1.to_out[0].state_dict())

    model.attn2.wq.load_state_dict(param.transformer_blocks[0].attn2.to_q.state_dict())
    model.attn2.wk.load_state_dict(param.transformer_blocks[0].attn2.to_k.state_dict())
    model.attn2.wv.load_state_dict(param.transformer_blocks[0].attn2.to_v.state_dict())
    model.attn2.out_proj.load_state_dict(param.transformer_blocks[0].attn2.to_out[0].state_dict())

    model.fc0.load_state_dict(param.transformer_blocks[0].ff.net[0].proj.state_dict())
    model.fc1.load_state_dict(param.transformer_blocks[0].ff.net[2].state_dict())

    model.norm_attn0.load_state_dict(param.transformer_blocks[0].norm1.state_dict())
    model.norm_attn1.load_state_dict(param.transformer_blocks[0].norm2.state_dict())
    model.norm_act.load_state_dict(param.transformer_blocks[0].norm3.state_dict())

    model.cnn_out.load_state_dict(param.proj_out.state_dict())


# resnet 块参数加载
def load_res_block(model: ResNetBlock, param):
    model.time_emb[1].load_state_dict(param.time_emb_proj.state_dict())

    model.seq1[0].load_state_dict(param.norm1.state_dict())
    model.seq1[2].load_state_dict(param.conv1.state_dict())

    model.seq2[0].load_state_dict(param.norm2.state_dict())
    model.seq2[2].load_state_dict(param.conv2.state_dict())

    if isinstance(model.conv_shortcut, nn.Conv2d):
        model.conv_shortcut.load_state_dict(param.conv_shortcut.state_dict())


# 下采样块参数加载
def load_down_block(model: DownBlock, param):
    __load_transformer_block(model.trans_block1, param.attentions[0])
    __load_transformer_block(model.trans_block2, param.attentions[1])

    load_res_block(model.res_block1, param.resnets[0])
    load_res_block(model.res_block2, param.resnets[1])
    model.out.load_state_dict(param.downsamplers[0].conv.state_dict())


# 上采样块参数加载
def load_up_block(model: UpBlock, param):
    __load_transformer_block(model.trans_block1, param.attentions[0])
    __load_transformer_block(model.trans_block2, param.attentions[1])
    __load_transformer_block(model.trans_block3, param.attentions[2])

    load_res_block(model.res_block1, param.resnets[0])
    load_res_block(model.res_block2, param.resnets[1])
    load_res_block(model.res_block3, param.resnets[2])
    if isinstance(model.out, nn.Sequential):
        model.out[1].load_state_dict(param.upsamplers[0].conv.state_dict())


# unet整体参数加载
def load_pretrained_unet(model: UNet):
    params = UNet2DConditionModel.from_pretrained('./pretrained-params/', subfolder='unet')
    # in
    model.in_vae.load_state_dict(params.conv_in.state_dict())
    model.in_time[0].load_state_dict(params.time_embedding.linear_1.state_dict())
    model.in_time[2].load_state_dict(params.time_embedding.linear_2.state_dict())
    # down
    load_down_block(model.down_block1, params.down_blocks[0])
    load_down_block(model.down_block2, params.down_blocks[1])
    load_down_block(model.down_block3, params.down_blocks[2])
    load_res_block(model.down_res1, params.down_blocks[3].resnets[0])
    load_res_block(model.down_res2, params.down_blocks[3].resnets[1])
    # mid
    __load_transformer_block(model.mid_trans, params.mid_block.attentions[0])
    load_res_block(model.mid_res1, params.mid_block.resnets[0])
    load_res_block(model.mid_res2, params.mid_block.resnets[1])
    # up
    load_res_block(model.up_res1, params.up_blocks[0].resnets[0])
    load_res_block(model.up_res2, params.up_blocks[0].resnets[1])
    load_res_block(model.up_res3, params.up_blocks[0].resnets[2])
    model.up_in[1].load_state_dict(params.up_blocks[0].upsamplers[0].conv.state_dict())
    load_up_block(model.up_block1, params.up_blocks[1])
    load_up_block(model.up_block2, params.up_blocks[2])
    load_up_block(model.up_block3, params.up_blocks[3])
    # out
    model.out[0].load_state_dict(params.conv_norm_out.state_dict())
    model.out[2].load_state_dict(params.conv_out.state_dict())
    return model
