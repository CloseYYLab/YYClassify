import json
import os
import matplotlib.pyplot as plt


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


def plot_images(images, true_labels, predicted_labels,save = True):
    num_images = len(images)
    num_rows = (num_images + 7) // 8  # 计算所需的行数

    fig, axes = plt.subplots(nrows=num_rows, ncols=8, figsize=(16, 2 * num_rows))
    with open('class.json','r') as f:
        class_dict = json.load(f)

    # 遍历图像数据并绘制
    for i, (image, true_label, predicted_label) in enumerate(zip(images, true_labels, predicted_labels)):
        row = i // 8
        col = i % 8
        ax = axes[row, col]
        ax.imshow(image)  # 绘制图像
        ax.axis('off')  # 关闭坐标轴

        # 设置标题并使用不同颜色区分正确和错误分类
        if true_label == predicted_label:
            ax.set_title(f'True: {class_dict[str(true_label)]}\nPredicted: {class_dict[str(predicted_label)]}', color='green')
        else:
            ax.set_title(f'True: {class_dict[str(true_label)]}\nPredicted: {class_dict[str(predicted_label)]}', color='red')
    if save:
        save_file = 'run/detect'
        os.makedirs(save_file,exist_ok=True)
        plt.savefig('run/detect/pred.png')

    plt.tight_layout()
    plt.show()
