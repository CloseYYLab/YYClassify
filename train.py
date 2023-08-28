import sys

import torch
import os
import math

from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets
import argparse
from model import Model
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from util import *
from datasets import ImageDataset


def args_parser():
    parser = argparse.ArgumentParser('Your_net')
    parser.add_argument('--root_dir', default='E:\classification\\flower_photos', help='data file')
    parser.add_argument('--batch_size', default=32, help='batch—size')
    parser.add_argument('--data_size', default=224, help='batch—size')
    parser.add_argument('--epochs', default=5, help='epoch')
    parser.add_argument('--optim', default='Adam', help='choose one optim method, SGD Adam ')
    parser.add_argument('--lr', default=0.0001, help='learning rate')
    parser.add_argument('--lrf', default=0.1, help='cos')
    parser.add_argument('--save_path', default='save_weights', help='save weights')
    parser.add_argument('--num_classes', default=5, help='class number')
    parser.add_argument('--auc', default=False, help='class number')
    parser.add_argument('--cfg', default='vgg.yaml', help='class number')

    parser.add_argument('--weights', type=str, default='', help='class number')
    parser.add_argument('--resume', type=bool, default=False, help='continue last epoch train')

    opt = parser.parse_args()
    return opt


def main(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))

    train_set = ImageDataset(opt, root=opt.root_dir + '/train/', istrain=True)
    val_set = ImageDataset(opt, root=opt.root_dir + '/val/', istrain=False)
    # 创建递增文件夹
    save_path = makefile()
    os.makedirs(save_path + '/save_weights', exist_ok=True)

    cla_dict = dict((key, val) for key, val in enumerate(os.listdir(opt.root_dir + '/train')))
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=len(cla_dict))
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    num_train, num_val = len(train_set), len(val_set)
    model_name = 'Your_net'
    model = Model(cfg=opt.cfg, ch=3, nc=opt.num_classes, auc=opt.auc).to(device)
    print(model)
    best_acc = 0.0

    Sw = SummaryWriter('./runs/' + model_name)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False)
    train_step = len(train_loader)

    if opt.weights != '':
        ckpt = torch.load(opt.weights, map_location='cpu')
        model.load_state_dict(ckpt)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    loss_function = torch.nn.CrossEntropyLoss()

    lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - opt.lrf) + opt.lrf  # cos
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(opt.epochs):
        from tqdm import tqdm
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        model.train()
        for i, data, in enumerate(train_bar):
            train_img, train_lable = data[0], data[1]
            train_img, train_lable = train_img.to(device), train_lable.to(device)

            optimizer.zero_grad()
            output = model(train_img)
            if isinstance(output, list):
                loss_l = loss_function(output[-1], train_lable)
                loss_m = loss_function(output[-2], train_lable)
                loss_s = loss_function(output[-3], train_lable)
                loss = loss_l + loss_m + loss_s
            else:
                loss = loss_function(output, train_lable)
            loss.backward()
            optimizer.step()

            running_loss += loss

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     opt.epochs,
                                                                     loss)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            acc = 0
            val_bar = tqdm(val_loader, file=sys.stdout, colour='red')
            for i, data in enumerate(val_bar):
                val_img, val_lable = data[0], data[1]
                val_img, val_lable = val_img.to(device), val_lable.to(device)

                output = model(val_img)
                pre = torch.max(output, dim=1)[1]
                acc += torch.eq(pre, val_lable).sum().item()

        val_acc = acc / num_val
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_step, val_acc))

        tags = ["loss", "accuracy", "learning_rate"]
        Sw.add_scalar(tags[0], running_loss / train_step, epoch)  # 标题，x轴平均损失，y轴epoch
        Sw.add_scalar(tags[1], acc, epoch)
        Sw.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path + '/save_weights/{}.pt'.format(model_name))
    print('Finished train and val')


if __name__ == '__main__':
    opt = args_parser()
    print(vars(opt))
    main(opt)
