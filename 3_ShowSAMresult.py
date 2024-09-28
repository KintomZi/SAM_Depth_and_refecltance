import sys

import numpy as np  # 导入NumPy库
import matplotlib
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库
import pickle  # 导入pickle库用于读取二进制文件
import os

matplotlib.use('TkAgg')


def load_anns(aXx, anns, tuMingDu):
    if len(anns) == 0:
        return
    if tuMingDu < 0 or tuMingDu > 1:
        raise ValueError('透明度tuMingDu 必须在0到1之间!')
    # 根据 'area' 大小对注释进行降序排序，'area' 大的在前
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    # 关闭自动缩放，保持当前图像的比例和范围
    # aXx.set_autoscale_on(False)

    # 创建一个全白的图像，其中 alpha 通道（透明度）设为 0 (完全透明)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0  # 设置透明度为0，完全透明

    for ann in sorted_anns:
        m = ann['segmentation']  # 获取分割掩码
        # color_mask = np.concatenate([np.random.random(3), [0.50]])  # 随机生成颜色并设置透明度
        color_mask = np.append(np.random.random(3), tuMingDu)
        img[m] = color_mask  # 应用颜色掩码

    aXx.imshow(img)  # 在子图上显示生成的图像


def SAMimg_ShoworSave(mask, background, tuoMingDu, saveImg: str = None):
    # 创建一个画布，包含一个子图，大小为12x6英寸
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(background)  # 背景图
    load_anns(ax, mask, tuoMingDu)  # 由背景图渲染上mask

    # ax.set_title('Feature-SAM')  # 设置子图标题
    ax.axis('off')  # 隐藏坐标轴

    if saveImg is None:
        plt.show()

    else:
        # bbox_inches='tight'紧贴图像, pad_inches=0.1留空距离
        plt.savefig(saveImg, bbox_inches='tight', pad_inches=0, dpi=1080)  # 保存图像到指定路径


# 主函数入口
if __name__ == '__main__':
    # 选择 特征存放文件夹
    featureFolder = '/media/huiwei/date/dataALL/ALS dataset/H3D dataset/Mar18_val/depth_0'
    objectName = os.path.basename(os.path.dirname(featureFolder))
    featureName = os.path.basename(featureFolder).split('_')[0]

    for root, dirs, files in os.walk(featureFolder):
        # 一般 背景图片和pkl文件 存在同级文件夹内
        for file in files:
            if '.pkl' in file and 'sam' in file.lower():
                print(f'正在执行：{file}')
                maskName = file.split(f'{featureName}_SAM')[0]
                maskPath = os.path.join(root, file)
                with open(maskPath, 'rb') as f:  # 'rb' 表示读取二进制文件
                    masks_F = pickle.load(f)

                tempBG = os.path.join(root, maskName + '.png')
                for file_bg in files:
                    if maskName in file_bg and '.png' in file_bg.lower():
                        if 'rgb' in file_bg.lower():
                            tempBG = os.path.join(root, file_bg)
                print(f'背景图像为:{os.path.basename(tempBG)}')
                imgBG = plt.imread(tempBG)

                savePath = os.path.join(root, file.split('.')[0] + '.png')
                SAMimg_ShoworSave(masks_F, background=imgBG, tuoMingDu=0.9, saveImg=savePath)
                print('执行完毕\n')

    #################################################################################
