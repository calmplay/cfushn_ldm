# -*- coding: utf-8 -*-
# @Time    : 2025/1/16 14:10
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import os

import numpy as np
import torch
from PIL import Image
from diffusers import DiffusionPipeline, PNDMScheduler
from transformers import CLIPTokenizer

from textcoder.text_encoder import load_pretrained_text_encoder, TextEncoder
from unet.unet import UNet
from utils import load_checkpoint, to_snake_case, cy_log
from vae.vae import load_pretrained_vae, VAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# scheduler和tokenizer (使用diffusers库的)
pipeline = DiffusionPipeline.from_pretrained('./pretrained-params/', safety_checker=None,
                                             local_files_only=True)
scheduler: PNDMScheduler = pipeline.scheduler
tokenizer: CLIPTokenizer = pipeline.tokenizer
del pipeline

# 模型加载
text_encoder = load_pretrained_text_encoder(TextEncoder())
vision_encoder = load_pretrained_vae(VAE())
unet, _, epoch, loss = (load_checkpoint(UNet()))

text_encoder.requires_grad_(False)
vision_encoder.requires_grad_(False)
unet.requires_grad_(False)

text_encoder.eval()
vision_encoder.eval()
unet.eval()

text_encoder.to(device)
vision_encoder.to(device)
unet.to(device)


def generate(text):
    # 词编码 [1, 77]
    pos = tokenizer(text, padding='max_length', max_length=77,
                    truncation=True, return_tensors='pt').input_ids.to(device)
    neg = tokenizer('', padding='max_length', max_length=77,
                    truncation=True, return_tensors='pt').input_ids.to(device)
    pos_out = text_encoder(pos)  # (1, 77, 768)
    neg_out = text_encoder(neg)  # -
    text_out = torch.cat((neg_out, pos_out), dim=0)  # (2, 77, 768)
    # 全噪声图
    vae_out = torch.randn(1, 4, 64, 64, device=device)
    # 生成时间步
    scheduler.set_timesteps(50, device=device)

    for time in scheduler.timesteps:
        noise = torch.cat((vae_out, vae_out), dim=0)
        noise = scheduler.scale_model_input(noise, time)
        # 预测噪声分布
        # print('text out', text_out.shape)
        pred_noise = unet(vae_out=noise, text_out=text_out, time=time)
        # 降噪
        pred_noise = pred_noise[0] + 7.5 * (pred_noise[1] - pred_noise[0])
        # 继续添加噪声
        vae_out = scheduler.step(pred_noise, time, vae_out).prev_sample

    # 从压缩图恢复成图片
    vae_out = 1 / 0.18215 * vae_out
    image = vision_encoder.decoder(vae_out)
    # 转换并保存
    image = image.cpu()
    image = (image + 1) / 2
    image = image.clamp(0, 1)
    image = image.permute(0, 2, 3, 1)
    image = image.numpy()[0]
    image = Image.fromarray(np.uint8(image * 255))
    # 如果指定文件夹不存在,则创建
    if not os.path.exists(f'./output/out_ckpt_{epoch}_loss_{loss}'):
        os.makedirs(f'./output/out_ckpt_{epoch}_loss_{loss}')
    desc = to_snake_case(text)
    image.save(f'./output/out_ckpt_{epoch}_loss_{loss}/{desc}.jpg')


if __name__ == '__main__':
    texts = [
        'a drawing of a star with a jewel in the center',
        'a drawing of a woman in a red cape',
        'a drawing of a dragon sitting on its hind legs',
        'a drawing of a blue sea turtle holding a rock',
        'a blue and white bird with its wings spread',
        'a blue and white stuffed animal sitting on top of a white surface',
        'a teddy bear sitting on a desk',
    ]  # 'the spider-man hanging upside down from a ceiling'
    images = []
    for i, text in enumerate(texts):
        generate(text)
        cy_log(f'text: {text}, finished')
