# -*- coding: utf-8 -*-
# @Time    : 2025/1/16 14:10
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm
import io

import torch
from PIL import Image
from datasets import load_dataset
from diffusers import DiffusionPipeline, PNDMScheduler
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer

from textcoder.text_encoder import TextEncoder, load_pretrained_text_encoder
from unet.unet import UNet, load_pretrained_unet
from utils import save_checkpoint, load_checkpoint, cy_log
from vae.vae import VAE, load_pretrained_vae

train_mode = 0  # 0:从头开始训练; 1:加载预训练权重; 2:加载最近的checkpoint权重
batch_size = 1
start_epoch = 0
epochs = 400

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# scheduler和tokenizer (使用diffusers库的)
pipeline = DiffusionPipeline.from_pretrained('./pretrained-params/', safety_checker=None,
                                             local_files_only=True)
scheduler: PNDMScheduler = pipeline.scheduler
tokenizer: CLIPTokenizer = pipeline.tokenizer
del pipeline
cy_log('Device :', device)
cy_log('Scheduler settings: ', scheduler)
cy_log('Tokenizer settings: ', tokenizer)

# 数据处理与加载
# # load from Hugging Face
# dataset = load_dataset(path='lansinuote/diffsion_from_scratch', split='train')
# load from local
dataset = load_dataset('parquet', data_files='./data/train.parquet', split='train')
# (ps: 加载本地数据集,如果加载的是标准格式文件（如 csv, json, parquet 等），可以直接使用格式名称作为 path)

# 图像增强模块
compose = transforms.Compose([
    # 最小边的长度会变为 512，保持原始长宽比
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
    # 按照短边裁剪出正方形区域，裁剪结果的宽高都是 512×512
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    # 对所有通道使用相同的均值和标准差
    transforms.Normalize([0.5], [0.5])
])


# 将文本&图像pair处理
def pair_text_image_process(data):
    """
    # dateset结构: [{'image': {'bytes':bytes, 'path': str}, 'text': str}]
    # process后-> [{'pixel_values': Tensor, 'input_ids': Tensor}], 列表长度=总样本数量
    """
    # pixel_values = [compose(i) for i in data['image']]
    # pixel_values = [compose(Image.open(io.BytesIO(i['bytes']))) for i in data['image']]
    # pixel_values = [compose(Image.open(io.BytesIO(i['bytes'])).convert('RGB'))
    #                 for i in data['image']]
    pixel_values = [compose(Image.open(io.BytesIO(i['bytes']))) for i in data['image']]
    input_ids = tokenizer.batch_encode_plus(
            data['text'], padding='max_length', truncation=True, max_length=77).input_ids
    return {'pixel_values': pixel_values, 'input_ids': input_ids}


def collate_fn(data):
    pixel_values = [pair['pixel_values'] for pair in data]
    input_ids = [pair['input_ids'] for pair in data]
    pixel_values = torch.stack(pixel_values).to(device)
    input_ids = torch.stack(input_ids).to(device)
    return {'pixel_values': pixel_values, 'input_ids': input_ids}


dataloader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size,
                        num_workers=0)

# 模型与网络初始化
text_encoder: TextEncoder = load_pretrained_text_encoder(TextEncoder())
vae: VAE = load_pretrained_vae(VAE())
unet: UNet = UNet()  # 后面根据 train_mode决定如何初始化权重

text_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(True)

text_encoder.eval()
vae.eval()
unet.train()

text_encoder.to(device)
vae.to(device)
unet.to(device)

# 优化器, 损失函数
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01,
                              eps=1e-8)
criterion = torch.nn.MSELoss().to(device)

# 通过 autocast 和 GradScaler 实现支持混合精度训练
scaler = GradScaler()

# 根据train_mode初始化unet
if train_mode == 1:
    unet = load_pretrained_unet(unet)
elif train_mode == 2:
    unet, optimizer, last_epoch, _ = load_checkpoint(unet, optimizer)
    start_epoch = last_epoch + 1


# 定义loss
def loss_fn(data_pair):
    img = data_pair['pixel_values']
    token = data_pair['input_ids']
    with torch.no_grad():
        # 文字编码 [1, 77] -> [1, 77, 768]
        text_code = text_encoder(token)
        # 抽取图像特征图 [1, 3, 512, 512] -> [1, 4, 64, 64]
        out_vae = vae.encoder(img)
        out_vae = vae.sample(out_vae)
        out_vae = out_vae * 0.18215  # 0.18215 = vae.config.scaling_factor

    # 添加噪声  (1000 = scheduler.num_train_time_steps)
    noise = torch.randn_like(out_vae)
    noise_step = torch.randint(0, 1000, (1,)).to(device)
    out_vae = scheduler.add_noise(out_vae, noise, torch.IntTensor(noise_step))

    # 根据文字信息,把特征图中的噪声计算出来
    noise_pred = unet(out_vae=out_vae,
                      out_encoder=text_code,
                      time=noise_step)  # [1, 4, 64, 64]

    return criterion(noise_pred, noise)


# train
loss_recorder = []
cy_log('start training ...')

for epoch in range(start_epoch, epochs):
    epoch_loss = 0.
    for step, data in enumerate(dataloader):
        loss = loss_fn(data)
        loss.backward()
        epoch_loss += loss.item()
        # 每4步更新一次
        if (epoch * len(dataloader) + step) % 4 == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        if step % 10 == 0:
            cy_log(f' > step: {step}  loss: {loss.item():.6f} loss_sum: {epoch_loss:.6f}')

    loss_recorder.append((epoch, epoch_loss))
    cy_log(f'epoch: {epoch:03}  loss: {epoch_loss:.8f}')
    cy_log('--------------------loss records:--------------------')
    for record in loss_recorder:
        cy_log(record)
    cy_log('------------------------------------------------------')

    # 每20个epoch保存一次
    if (epoch + 1) % 20 == 0:
        cy_log(f'save check point (epoch:{epoch})...')
        save_checkpoint(unet, optimizer, epoch, epoch_loss)
        cy_log('save successfully.\n')

cy_log('end training.')
