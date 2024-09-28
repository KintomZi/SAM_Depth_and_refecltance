import numpy as np
import matplotlib
from GridManager import GridManager
from datetime import datetime  # 导入datetime模块
import os
import pickle  # 导入pickle库用于读取二进制文件
import sys

matplotlib.use('TkAgg')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    objectLocation = r'/media/huiwei/date/dataALL/ALS dataset/H3D dataset/Mar18_val.npy'

    objectFolder = os.path.dirname(objectLocation)  # 获取文件所在的文件夹路径
    objectName = os.path.basename(objectLocation).split('.')[0]  # 获取文件名

    resultFolder = os.path.join(objectFolder, objectName)
    os.makedirs(resultFolder, exist_ok=True)  # 以文件名创建文件夹

    print('开始处理点云转换为正射栅格...')
    data = np.load(objectLocation)
    # 调用 x, y, z 数据
    x = data['x']  # 访问 x 列
    y = data['y']  # 访问 y 列
    z = data['z']  # 访问 z 列
    red = data['red']
    green = data['green']
    blue = data['blue']
    reflectance = data['reflectance']
    classification = data['classification']

    # 将 x, y, z 合并为一个二维数组，每行是一个点的 (x, y, z) 坐标
    xyz = np.column_stack([x, y, z])
    rgb = np.column_stack([red, green, blue])

    GM = GridManager(R=0.05)
    GM.Pt2Grids(xyz, RGB=rgb, pc_label=classification)
    GM.Pixel_loadFeature(feature=z)  # !!!!!!!!!!同时需要设置文件夹名称(featureName)
    GM.GridsSplit(50, edgeReserve=True)

    print('\n点云处理完毕,开始形成图像并保存...')
    featureName = 'depth'  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    current_time = datetime.now().strftime('_%Y%m%d_%H%M%S')
    featureFolder = os.path.join(resultFolder, featureName + current_time)
    os.makedirs(featureFolder, exist_ok=True)  # 创建特征文件夹

    with open(os.path.join(featureFolder, 'GM_parameter.pkl'), 'wb') as f:  # 'wb' 表示写入二进制文件
        pickle.dump(GM, f)

    # GM.ShowVisGrid()  # 是否进行特征映射的多colormap显示
    # GM.ShowGrid_orSave(GM.GFeature, nullMethod=1)  # 是否进行图像显示(或保存) GFeature  G_RGB
    # 是否进行分割后图像显示(或保存)"

    imgRGB = os.path.join(featureFolder, objectName + '_RGB.png')
    # 检查文件夹是否存在，实现 整幅 RGB图像 的保存
    if GM.RGB is not None and not os.path.exists(imgRGB):
        print(f"\n整幅RGB文件开始已创建: {os.path.basename(imgRGB)}")
        GM.ShowGrid_orSave(GM.G_RGB, nullMethod=1, SavePath=imgRGB)  #

    imgFeature = os.path.join(featureFolder, objectName + f'_{featureName}.png')
    # 检查文件夹是否存在，实现 整幅 特征图像 的保存
    if not os.path.exists(imgFeature):
        print(f"\n整幅Feature文件开始创建: {os.path.basename(imgFeature)}")
        GM.ShowGrid_orSave(GM.GFeature, nullMethod=1, SavePath=imgFeature)  #

    splitAll = os.path.join(featureFolder, f'split_{GM.GSplitNums}')
    os.makedirs(splitAll, exist_ok=True)  # 以创建 分割后结果 文件夹

    for i in range(GM.GSplitNums):
        print(f'\n分割后的 第{i}幅图像')
        if GM.RGB is not None:
            print('== RGB ==')
            subNameRGB = objectName + '_' + str(i) + f'_RGB.png'
            savePathRGB = os.path.join(splitAll, subNameRGB)
            GM.ShowGrid_orSave(GM.G_RGBsplit[i], 3, savePathRGB)  # GFeatureSplit  G_RGBsplit
        print('== Feature ==')
        subName = objectName + '_' + str(i) + f'_{featureName}.png'
        savePath = os.path.join(splitAll, subName)
        GM.ShowGrid_orSave(GM.GFeatureSplit[i], 3, savePath)  # GFeatureSplit  G_RGBsplit

    print('\n!!!ProcessPoint进程结束!!!')
