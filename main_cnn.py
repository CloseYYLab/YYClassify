import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from cam import GradCAM, show_cam_on_image, center_crop_img
from model import Model


def plot_cam(model, device, val_loader):
    # model = Model(cfg='vgg.yaml', ch=3, nc=5, auc=False).to('cuda')
    # ckpt = torch.load('E:\classification\Your_net\\run\exp2\save_weights\Your_net.pt',map_location='cpu')
    # model.load_state_dict(ckpt)
    target_layers = [model.classifer.conv]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t_trans = transforms.Compose([
        transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)],
                             std=[1 / s for s in std]),
        transforms.ToPILImage(),
    ])
    # load image
    data_iter = iter(val_loader)
    images = next(data_iter)
    image = images[0][5]

    img = t_trans(image)
    # img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = image
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0).to('cuda')

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    target_category = 0  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()
