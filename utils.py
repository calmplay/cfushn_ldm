# -*- coding: utf-8 -*-
# @Time    : 2025/1/16 13:47
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import os
import re

import torch

from unet.unet import UNet


def cy_log(*args):
    print('cfushn >>> ', *args)  # 这里必须使用*解包, 否则args会以列表形式输出


def save_checkpoint(model, optimizer, epoch, loss):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    folder = '/home/cy/workdir/cfushn_ldm/model_save'
    torch.save(state, os.path.join(folder, f"checkpoint_{epoch}.pth"))


def load_checkpoint(unet: UNet, optimizer=None, checkpoint_path: str = None):
    """
    读取模型的 checkpoint 文件并加载权重、优化器状态及其他训练信息。

    Args:
        unet (torch.nn.Module): 需要加载权重的模型实例。
        optimizer (torch.optim.Optimizer, optional): 如果需要恢复优化器状态，请传入优化器实例。
        checkpoint_path (str): 默认None,会加载特定目录中最后一个checkpoint

    Returns:
        dict: 包含额外信息的字典，例如 epoch 和损失。
    """

    if checkpoint_path is None:
        folder = '/home/cy/workdir/cfushn_ldm/model_save'
        files = os.listdir(folder)
        matched_files = []
        for file in files:
            match = re.match("checkpoint_(\d+).pth.tar", file)
            if match:
                # 提取文件名中的 epoch 值并保存
                epoch = int(match.group(1))  # 假设第一个捕获组是数字
                matched_files.append((epoch, file))
        # 如果没有匹配文件
        if not matched_files:
            raise FileNotFoundError(f"there has no matched models in '{folder}'")
        # 按 epoch 值排序并返回最后一个文件的完整路径
        matched_files.sort(key=lambda x: x[0])  # 按 epoch 升序排序
        latest_file = matched_files[-1][1]  # 获取最后一个文件名
        checkpoint_path = os.path.join(folder, latest_file)

    # 加载 checkpoint 文件
    check_point = torch.load(checkpoint_path)

    # 加载模型权重
    if unet is None:
        unet = UNet()
    unet.load_state_dict(check_point["state_dict"])
    unet.train()

    # 如果提供了优化器实例，加载优化器状态
    optimizer is None or optimizer.load_state_dict(check_point["optimizer"])

    last_epoch = check_point["epoch"]
    loss = check_point["loss"]
    # 返回其他训练信息
    print(f"-------------load checkpoint: epoch:{last_epoch},loss:{loss}---------------")
    return unet, optimizer, last_epoch, loss


def to_snake_case(text: str) -> str:
    """
    将一段提示语转化为用下划线连接的字符串。

    Args:
        text (str): 输入的提示语。

    Returns:
        str: 转换后的字符串。
    """
    import re

    # strip()去除首尾空格,并将多个空格替换为单个空格
    cleaned_prompt = re.sub(r'\s+', ' ', text.strip())

    # 替换空格和特殊字符为下划线，并将字符串转换为小写
    snake_case = re.sub(r'[^a-zA-Z0-9]+', '_', cleaned_prompt).lower()

    # 确保没有多余的下划线（strip('_')去除首尾下划线）
    snake_case = re.sub(r'_+', '_', snake_case).strip('_')

    return snake_case
