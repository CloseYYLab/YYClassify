import glob

import torch
from torchvision import transforms
import os
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self,
                 opt,
                 istrain: bool,
                 root: str,

                 ):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.root = root

        """ declare data augmentation """
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # 448:600
        # 384:510
        # 768:
        if istrain:
            # transforms.RandomApply([RandAugment(n=2, m=3, img_size=data_size)], p=0.1)
            # RandAugment(n=2, m=3, img_size=sub_data_size)
            self.transforms = transforms.Compose([
                transforms.Resize((340, 340)),
                transforms.RandomCrop((opt.data_size, opt.data_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-10, 10)),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((340, 340)),
                transforms.CenterCrop((opt.data_size, opt.data_size)),
                transforms.ToTensor(),
                normalize
            ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)

    def getDataInfo(self, root):
        data_infos = []
        folders = os.listdir(root)
        folders.sort()  # sort by alphabet
        print("[dataset] class number:", len(folders))
        for class_id, folder in enumerate(folders):
            files = os.listdir(root + folder)
            for file in files:
                data_path = root + folder + "/" + file
                data_infos.append({"path": data_path, "label": class_id})
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):

        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        # read image by PIL.
        image_name = image_path.split('.')[:-4]
        img = Image.open(image_path).convert('RGB')

        img = self.transforms(img)

        # return img, sub_imgs, label, sub_boundarys
        return img, label, image_name


class predDataset(torch.utils.data.Dataset):
    def __init__(self,
                 opt,
                 root: str,
                 ):

        self.root = root
        self.imagelist = glob.glob(self.root)

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, index):
        image_path = self.imagelist[index]
        image_name = image_path.split('/')[-1]
        img = Image.open(image_path).convert('RGB')

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize( mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])

        img = self.transforms(img)

        return img, image_name


