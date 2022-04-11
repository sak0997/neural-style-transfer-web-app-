import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from transformer_net import TransformerNet
from vgg import Vgg16
import streamlit as st

# we will use the conecpt of caching here that is once a user has used a particular model instead of loading
# it again and again everytime they use it we will cache the model.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем модель


@st.cache
def load_model(model_path):

    with torch.no_grad():
        style_model = TransformerNet()  # transformer_net.py содержит модель стиля
        state_dict = torch.load(model_path)
        # Удалить сохраненные устаревшие ключи running_* в InstanceNorm из контрольной точки
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        return style_model

# Содержимое изображение и модель стиля, которую мы загрузили
# with load_model function


@st.cache
def stylize(style_model, content_image, output_image):

    # if the content image is a path then
    if type(content_image) == "str":
        content_image = utils.load_image(
            content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)

    # Для обработки одного изображения как пакета
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = style_model(content_image).cpu()

    # Выходное изображение здесь путь к выходному изображению
    img = utils.save_image(output_image, output[0])
    return img


if __name__ == "__main__":
    main()
