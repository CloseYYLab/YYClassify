import json
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

def makefile():
    c = 99999
    path = ''
    for i in range(1, c):
        path = './run/exp{}'.format(i)
        if (os.path.exists(path)):
            continue
        else:
            os.makedirs(path)
            break
    return path


def plot_images(images, true_labels, predicted_labels,epoch,save_path, save=True):

    num_images = images.shape[0]
    num_rows = num_images // 4  # 计算所需的行数

    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(16, 16))
    with open('class_indices.json', 'r') as f:
        class_dict = json.load(f)

    os.makedirs(save_path + '/result', exist_ok=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t_trans = transforms.Compose([
        transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)],
                             std=[1 / s for s in std]),
        transforms.ToPILImage(),

    ])

    # 遍历图像数据并绘制
    for i, (image, true_label, predicted_label) in enumerate(zip(images, true_labels, predicted_labels)):
        row = i // 4
        col = i % 4
        ax = axes[row, col]

        image = t_trans(image.cpu())

        ax.imshow(image)  # 绘制图像
        ax.axis('off')  # 关闭坐标轴
        # 设置标题并使用不同颜色区分正确和错误分类
        if true_label == predicted_label:
            ax.set_title(f'True: {class_dict[str(true_label.item())]}\nPredicted: {class_dict[str(predicted_label.item())]}',
                         color='green')
        else:
            ax.set_title(f'True: {class_dict[str(true_label.item())]}\nPredicted: {class_dict[str(predicted_label.item())]}',
                         color='red')
        fig.subplots_adjust(top=0.92)

    plt.savefig(save_path + '/result/{}.png'.format(epoch))

    plt.show()
