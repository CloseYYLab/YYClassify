import json
import sys

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
def args_parser():
    parser = argparse.ArgumentParser('Your_net')
    parser.add_argument('--pre_root', default='E:\classification\\flower_photos\\val', help='data file')
    parser.add_argument('--img_size', default=224, help='data file')
    parser.add_argument('--batch_size', default=1, help='batch—size')
    parser.add_argument('--plot', default=False, help='plot')

    parser.add_argument('--num_class', default=5, help='class number')
    parser.add_argument('--weights', type=str, default='E:\classification\Your_net\\run\exp2\save_weights\Your_net.pt', help='class number')
    parser.add_argument('--cfg', default='s.yaml', help='class number')

    opt = parser.parse_args()
    return opt


IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes


def main(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('class_indices.json','r') as f:
        class_indict = json.load(f)

    data_transform = transforms.Compose(
        [transforms.Resize(384),
         transforms.CenterCrop(opt.img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    model = Model(cfg=opt.cfg, ch=3, nc=opt.num_class, auc=False)
    ckpt = torch.load(opt.weights,map_location='cpu')
    model.load_state_dict(ckpt)




    if  os.path.isdir(opt.pre_root):
        predsets = predDataset(opt, opt.pre_root)
        predataloader = DataLoader(predsets, batch_size=1, shuffle=False)

        pre_bar = tqdm(predataloader, file=sys.stdout)
        for data , image_name in pre_bar:
            image = data

            out = model(image)

            pred_cls = torch.max(out, dim=1)[1]

            print('{} The image is {}'.format(image_name, class_indict[str(pred_cls.item())]),
                  '\t pro is {}'.format(round(pro.item(), 3)),
                  'Inference time is {}ms'.format(t * 1000),

                  )
    else:
        img_path = opt.pre_root

        image = Image.open(img_path)

        image = data_transform(image)

        start_time = time.time()

        output = torch.squeeze(model(image.to(device))).cpu()
        end_time = time.time()

        predict = torch.softmax(output, dim=0)
        pro, predict_cls = torch.max(predict, dim=0)
        t = end_time - start_time

        print('{} The image is {}'.format(img_path, class_indict[str(predict_cls.item())]),
              '\t pro is {}'.format(round(pro.item(), 3)),
              'Inference time is {}ms'.format(t * 1000),

              )
if __name__ == '__main__':
    opt = args_parser()
    main(opt)
