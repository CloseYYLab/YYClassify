import json
import sys

import matplotlib.pyplot as plt
import torch
from model import Model
import argparse
import os
from torchvision import transforms
from datasets import predDataset
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import json
import time
from util import *


def args_parser():
    parser = argparse.ArgumentParser('Your_net')
    parser.add_argument('--pre_root', default='E:\classification\\flower_photos\\val\\daisy', help='data file')
    parser.add_argument('--img_size', default=224, help='data file')
    parser.add_argument('--batch_size', default=1, help='batch—size')
    parser.add_argument('--plot', default=False, help='plot')

    parser.add_argument('--num_class', default=5, help='class number')
    parser.add_argument('--weights', type=str, default='E:\classification\Your_net\\run\exp2\save_weights\Your_net.pt',
                        help='class number')

    parser.add_argument('--save', default=True, help='class number')
    parser.add_argument('--cfg', default='vgg.yaml', help='class number')

    opt = parser.parse_args()
    return opt


IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes


def main(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t_trans = transforms.Compose([
        transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)],
                             std=[1 / s for s in std]),
        transforms.ToPILImage(),

    ])

    with open('class_indices.json', 'r') as f:
        class_indict = json.load(f)

    model = Model(cfg=opt.cfg, ch=3, nc=opt.num_class, auc=False).to(device)
    ckpt = torch.load(opt.weights, map_location='cpu')
    model.load_state_dict(ckpt)

    predsets = predDataset(opt, opt.pre_root)
    predataloader = DataLoader(predsets, batch_size=1, shuffle=False)

    pre_bar = tqdm(predataloader, file=sys.stdout)

    for data, image_name in pre_bar:
        image = data.to(device)

        start_time = time.time()
        out = model(image)
        end_time = time.time()

        t = end_time - start_time
        pro, pred_cls = torch.max(out, dim=1)

        print('{} The image is {}'.format(image_name, class_indict[str(pred_cls.item())]),
              '\t pro is {}'.format(round(pro.item(), 3)),
              'Inference time is {}ms'.format(t * 1000),

              )

        # 保存预测结果
        if opt:
            img = t_trans(image.squeeze())
            plt.imshow(img)
            plt.title(class_indict[str(pred_cls.item())])
            plt.savefig('')
            save_path = makefile()
            os.makedirs(save_path + '/save_weights', exist_ok=True)
        # 保存热力图


if __name__ == '__main__':
    opt = args_parser()
    main(opt)
