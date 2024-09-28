import sys
from array import array

import numpy as np  # 导入NumPy库
import matplotlib
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库
import pickle  # 导入pickle库用于读取二进制文件
import os
from scipy.ndimage import distance_transform_edt  # 用来处理图像的空值替换
import csv

matplotlib.use('TkAgg')


def maskMerge(mask_pkl, saveNullMask: bool = False):
    print('\n执行maskMerge,其中class默认从 1 开始，0 代表没有mask的像素')
    maskLabelArray = np.full(mask_pkl[0]['segmentation'].shape, 0, dtype=np.single)
    for class_nums in range(len(mask_pkl)):
        maskLabelArray[mask_pkl[class_nums]['segmentation']] = int(class_nums + 1)
    print(f'== min_class:{maskLabelArray.min()} == max_class:{maskLabelArray.max()} ==')
    if saveNullMask:
        print('maskMerge执行完毕')
        return maskLabelArray
    else:
        if maskLabelArray.min() == 0:
            print('执行将 0 替换为最邻近值')
            _, nearest_indices = distance_transform_edt(maskLabelArray == 0, return_indices=True)
            # tuple(nearest_indices)转换为二维索引，用最接近的非零元素的值替换零
            maskLabelArray = maskLabelArray[tuple(nearest_indices)]
            print('maskMerge执行完毕')
        return maskLabelArray


# 主函数入口
if __name__ == '__main__':
    # 选择 特征存放文件夹(可选择)
    featureFolder = '/media/huiwei/date/dataALL/ALS dataset/H3D dataset/Mar18_val/depth_0'
    # 选择 mask 对象（可选择）
    # 相对于 特征存放文件夹 的路径
    mask_name = 'split_6/Mar18_val_5_depth_SAM.pkl'

    feature_name = os.path.basename(featureFolder).split('_')[0]  # 获取特征名称
    GM_location = os.path.join(featureFolder, 'GM_parameter.pkl')  # 获取 格网 的对象文件
    maskLocation = os.path.join(featureFolder, mask_name)  # 获取 mask 文件

    with open(maskLocation, 'rb') as f:  # 'rb' 表示读取二进制文件，读取mask文件
        print(f'\n开始读取 {os.path.basename(maskLocation)} 文件')
        masks_file = pickle.load(f)

    labelArray = maskMerge(masks_file)  # 将 mask类别 数据合并到一个数组中

    '''
    plt.figure(figsize=(10, 10))  # 设置图像大小
    plt.imshow(labelArray, cmap='hsv')  # 使用 jet colormap 显示图像
    plt.colorbar()  # 添加颜色条
    plt.show()  # 显示图像
    '''

    with open(GM_location, 'rb') as f:  # 'rb' 表示读取二进制文件，读取格网对象文件
        print(f'\n开始读取 {os.path.basename(GM_location)} 文件')
        GM = pickle.load(f)

    featureLable = np.full(GM.Pt.shape[0], -1)  # 创建 写入Label 的一维数组
    duiqi_start = (1000, 2000)  # 选择 mask对应完整格网的位置，以mask的左上角为准!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print('\n将maskMerge后的数组，写入点云（其中 -1 表示无mask输入）')
    for i in range(labelArray.shape[0]):
        for j in range(labelArray.shape[1]):
            y_id = i + duiqi_start[0]
            x_id = j + duiqi_start[1]
            pixPt = GM.GPtSet[y_id, x_id]
            for ppt in pixPt:
                featureLable[ppt] = labelArray[i, j]

    truthClass = GM.Label
    mask_dict = {}  # 创建mask的类别字典，存放其中真值类别
    print('\n遍历写入的mask类别数据，进行mask的每个真值的存入')
    for index, value in enumerate(featureLable):
        # 如果整数不在字典中，初始化为空列表
        if value not in mask_dict:
            mask_dict[value] = []
        # 将对应的浮点数值添加到列表中
        mask_dict[value].append(truthClass[index])

    print('\n遍历mask类别字典，分析其中的真值组成与占比，同时写入并保存CSV文件')
    tClass, _ = np.unique(truthClass, return_counts=True)
    csvSaveFolder = os.path.dirname(maskLocation)
    csvSaveName = os.path.basename(maskLocation).split('.')[0] + '.csv'
    # 打开CSV文件进行写入
    with open(os.path.join(csvSaveFolder, csvSaveName), mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 写入表头
        header = ['ID'] + [f'{tc}' for tc in tClass] + ['Total']
        writer.writerow(header)

        # 写入数据
        for key, submask in mask_dict.items():
            mask_class, mcCounts = np.unique(submask, return_counts=True)
            row = [key]  # 初始化行，以ID开头
            for tc in tClass:
                if tc not in mask_class:
                    row.append(0)  # 如果类别不存在，添加0
                else:
                    suoying = np.where(mask_class == tc)[0]
                    percentZhanbi = (mcCounts[suoying] / len(submask)).item()
                    row.append(f'{percentZhanbi:.4f}')  # 保留四位小数
            row.append(len(submask))  # 添加Total
            writer.writerow(row)  # 写入行

    print(f"数据已保存到 {csvSaveName}")


