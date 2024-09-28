# Main task: Given point clouds of reasonable size (e.g. 20m*10m*5m);
#            Build grids with predefined grid size
# Input:  Original 3D point clouds (Large files should be divided into small tiles before)

# 主要任务:给定合理大小的点云(如20m*10m*5m)；构建网格与预定义的网格大小
# 输入:原始3D点云(大文件之前要分成小块)
import copy
import os
import numpy as np
import open3d as o3d
import math
# from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import distance_transform_edt  # 用来处理图像的空值替换
from sklearn.preprocessing import QuantileTransformer  # 用于处理图像的灰度分布


def featureDistribution(feature: np, num_intervals: int = 10):
    """
    将特征用区间展示出来

    Args:
        feature: 一维特征数组
        num_intervals: 区间数目，默认为10

    Returns:
        None
    """
    # 获取数组的最大值
    maxValue = np.max(feature)
    # 获取数组的最小值
    minValue = np.min(feature)
    # 计算最大值与最小值之间的差值
    value_range = maxValue - minValue
    # 计算每个区间的宽度
    interval_width = value_range / num_intervals
    # 创建数组来保存每个区间的计数
    interval_counts = np.zeros(num_intervals, dtype=int)
    # 计算每个区间内 feature 的数目
    for value in feature:
        # 计算该值所属的区间索引
        interval_index = int((value - minValue) / interval_width)
        # 特殊处理最大值以防超出范围
        if interval_index == num_intervals:
            interval_index -= 1
        # 增加对应区间的计数
        interval_counts[interval_index] += 1
    print('\n')
    # 打印每个区间的计数
    for i in range(num_intervals):
        print(
            f"区间 {i + 1} ({minValue + i * interval_width:.2f} to {minValue + (i + 1) * interval_width:.2f}): {interval_counts[i]} 个")
    raise 'Feature展示完毕，请注释掉并开始特征拉伸'


def completeImgSplit(img_path, sideSplitPixel, yushu_chuli: bool):
    """
    对整幅图像进行裁剪并保存，适用于全图的拉伸后的裁剪，不同于局部裁剪后拉伸

    Args:
        img_path: 裁剪的文件路径
        sideSplitPixel: 裁剪的边长像素数
        yushu_chuli: 是否进行多余图块的合并

    Returns:
          None
    """

    imageName = os.path.basename(img_path)  # 文件名称
    image = plt.imread(img_path)  # 使用 imread() 加载图像

    imgSplitFolder = os.path.join(os.path.dirname(img_path), imageName.split('.')[0] + '_Split')
    os.makedirs(imgSplitFolder, exist_ok=True)  # 创建特征文件夹
    # 获取图像的边长像素数
    img_H, img_W = image.shape[:2]  # 提取高度和宽度
    wSplit, wRemainder = divmod(img_W, sideSplitPixel)  # 底（商、余数）
    hSplit, hRemainder = divmod(img_H, sideSplitPixel)  # 高（商、余数）

    splitImgAll = []
    for i in range(hSplit + 1):  # 第i行
        for j in range(wSplit + 1):  # 第j列
            # 计算行和列的起始与结束位置
            row_start = i * sideSplitPixel
            col_start = j * sideSplitPixel

            # 处理到右侧边缘和底部边缘的情况
            row_end = min(row_start + sideSplitPixel, img_H)
            col_end = min(col_start + sideSplitPixel, img_W)

            # 如果边缘保留且没有余数则跳过额外的分割块
            if (i == hSplit and hRemainder == 0) and (j == wSplit and wRemainder == 0):
                continue

            subGSplit = image[row_start:row_end, col_start:col_end]
            splitImgAll.append(subGSplit)

    if yushu_chuli is False:
        for i in range(len(splitImgAll)):
            # 保存分割后的图像块
            split_image_name = f"{imageName.split('.')[0]}_Split{i}.png"  # 生成新文件名
            print(f'不合并多余块，正在保存{split_image_name}')
            saveImg = os.path.join(str(imgSplitFolder), split_image_name)
            plt.imsave(saveImg, splitImgAll[i])  # 保存图像
    else:
        kuan = wSplit + 1  # 列数

        temp = list(splitImgAll)  # 复制一个临时列表用于操作，避免在同一个数组上操作
        # 处理多余列向左合并
        for ii in range(hSplit + 1):
            if wRemainder != 0:  # 只有当有余数时才处理
                last_lie = ii * kuan + wSplit  # 最后一列的块
                if splitImgAll[last_lie].shape[1] != sideSplitPixel:  # 如果宽度不足
                    temp[last_lie - 1] = np.hstack((temp[last_lie - 1], temp[last_lie]))  # 向左合并
        # 处理多余行向上合并
        for jj in range(wSplit + 1):
            if hRemainder != 0:  # 只有当有余数时才处理
                last_hang = hSplit * kuan + jj  # 最后一行的块
                if temp[last_hang].shape[0] != sideSplitPixel:  # 如果高度不足
                    temp[last_hang - kuan] = np.vstack((temp[last_hang - kuan], temp[last_hang]))  # 向上合并

        saveNums = 0
        for ss in range(len(temp)):
            if temp[ss].shape[0] >= sideSplitPixel and temp[ss].shape[1] >= sideSplitPixel:
                # 保存分割后的图像块
                split_image_name = f"{imageName.split('.')[0]}_Split{saveNums}.png"  # 生成新文件名
                print(f'合并多余小块，正在保存{split_image_name}')
                saveImg = os.path.join(str(imgSplitFolder), split_image_name)
                plt.imsave(saveImg, temp[ss])  # 保存图像
                saveNums = saveNums + 1


class GridManager(object):
    def __init__(self, R=0.05):
        """
        Init paramters
        """
        self.Pt = None  # 原始点云数组[n×3]
        self.feature = None  # 原始特征数组[n×1]
        self.Label = None  # 标签数组[n×1]
        self.RGB = None  # RGB数组[n×3]

        self.Pt_non_negative = None  # 将点云整体移动确保非负
        self.feature_non_negative = None  # 将特征整体移动确保非负

        self.GSize = R  # 像素块的边长，空间分辨率
        # W*H structure
        self.GWidth = 0  # x == width =col 图片宽度像素数
        self.GHeight = 0  # y == height = row 图片高度像素数

        # !!!值得注意的是，矩阵是[行×列]，所以索引使用要为[self.idx[点序, 1], self.idx[点序, 0]]
        self.idx = None  # 建立的空间体素索引[x_id,y_id,z_id]，
        # 各个网格的点集索引
        self.GPtSet = defaultdict(list)  # 以建立的二维网格中，存放 在其内部的点的(多个)索引,网格序号从0开始
        # 各个网格的最上层点索引
        self.GPtSurface = defaultdict(list)  # 以建立的二维网格中，存放 在其内部的点的特征的(单个)索引,网格序号从0开始

        self.GFeature = None  # float image，图像的像素矩阵[W×H],存放线性值特征用于显示
        self.G_RGB = None  # float color-image 图像的像素矩阵[W×H×3],存放RGB值用于显示

        self.GSplitNums = None  # 分割图像数目
        self.GFeatureSplit = None  # 特征分割后的所有矩阵
        self.G_RGBsplit = None  # 图像分割后的所有矩阵

        self.UsedF = None
        #
        # self.pcd = o3d.geometry.PointCloud()
        # self.H_Line_set = o3d.geometry.LineSet()  # Horizontal lines
        # self.V_Line_set = o3d.geometry.LineSet()  # Vertical lines

    def Pt2Grids(self, points, RGB=None, pc_label=None):
        """
            对点云进行正射下的栅格化操作

        Args:
            points: 二维数组(n×3)，点云[x,y,z]
            RGB: 二维数组(n×3)，点云的RGB通道[Red,Green,Blue]
            pc_label: 标签

        Returns:
            None
        """

        self.Pt = points
        self.Label = pc_label
        # 计算一个点云数组的在第 0 轴上（即每列）最大/最小坐标值
        maxP = np.amax(points, 0)[0:3]
        minP = np.amin(points, 0)[0:3]
        self.Pt_non_negative = self.Pt - minP  # 将点云的最小值平移至0，从而使点云恒大于等于0

        loadRGB = False
        if RGB is not None:
            if isinstance(RGB, np.ndarray) and RGB.ndim == 2:
                loadRGB = True
                self.RGB = RGB
            else:
                raise ValueError('RGB格式错误')

        W = int(np.ceil((maxP[0] - minP[0]) / self.GSize))  # x == width 图片宽度像素数
        H = int(np.ceil((maxP[1] - minP[1]) / self.GSize))  # y == h 图片高度像素数
        self.GWidth = W
        self.GHeight = H
        Rs = (self.GSize, self.GSize, self.GSize)
        # 每个元素除以Rs数组中的对应元素，然后进行向下取整操作,从而生成空间体素索引[x_id,y_id,z_id]
        self.idx = np.floor((self.Pt_non_negative / Rs)).astype(np.ushort)

        # 处理边界：例如确保在边界上的点被正确包含
        # 在这里假设包含边界，将最外围的边界点收到次外围栅格里
        self.idx[self.idx[:, 0] >= W, 0] = W - 1
        self.idx[self.idx[:, 1] >= H, 1] = H - 1

        # 创建一个形状为 (W, H) 的特征矩阵，并将矩阵中每个元素都初始化为-1
        self.GFeature = np.full((H, W), -1, dtype=np.single)
        # 创建一个形状为 (W, H, 3) 的RGB矩阵，并将矩阵中每个元素都初始化为[-1,-1,-1]
        self.G_RGB = np.full((H, W, 3), -1, dtype=np.float32)

        for p in range(points.shape[0]):
            self.GPtSet[(self.idx[p, 1], self.idx[p, 0])].append(p)  # 将各个点存入对应栅格中
            # 如果当前的坐标Z值大于对应栅格里面的坐标Z值，说明位于最上层
            if self.Pt_non_negative[p, 2] > self.GFeature[self.idx[p, 1], self.idx[p, 0]]:
                self.GPtSurface[(self.idx[p, 1], self.idx[p, 0])].append(p)  # 传入最上层点的索引
                if loadRGB:
                    self.G_RGB[self.idx[p, 1], self.idx[p, 0]] = RGB[p, :]  # 传入最上层点的RGB

    def Pixel_loadFeature(self, feature: np = None):
        """
        根据每个网格最顶层的点的索引将一维特征值写入GFeature中用于显示

        Args:
            feature: 一维特征,不输入则 默认用坐标Z值

        Returns:
            None
        """
        if feature is None:  # 默认特征为归一化Z坐标值
            self.feature = self.Pt_non_negative[:, 2]
            self.feature_non_negative = self.feature - self.feature.min()  # 将特征的最小值平移至0，从而使点云恒大于等于0

        elif isinstance(feature, np.ndarray) and feature.ndim == 1:  # 一维特征归一化
            if np.all(feature == 0):
                raise ValueError("feature数组中的所有元素都是零")
            else:
                self.feature = feature
                feature_non_negative = feature - feature.min()  # 将特征的最小值平移至0，从而使点云恒大于等于0
                self.feature_non_negative = feature_non_negative
        else:
            raise ValueError('特征格式错误，只处理一维特征')  # 处理其他可能的情况，或者抛出错误

        # 将最上层点的特征赋值到对应的网格中
        for (y_id, x_id), point_idx in self.GPtSurface.items():
            self.GFeature[y_id, x_id] = self.feature_non_negative[point_idx[0]]

    def GridsSplit(self, sideLength, edgeReserve: bool = False):  # overlapPercent=0
        # 创建一个形状为 (H, W) 的特征矩阵，并将矩阵中每个元素都初始化为-1
        trans = np.full((self.GHeight, self.GWidth), -1, dtype=np.single)
        sidePixelNums = math.ceil(sideLength / self.GSize)  # 获取分割图幅的边长像素数,向上取整
        wSplit, wRemainder = divmod(self.GWidth, sidePixelNums)  # 底（商、余数）
        hSplit, hRemainder = divmod(self.GHeight, sidePixelNums)  # 高（商、余数）

        GFS = []
        GrgbS = []
        for i in range(hSplit + 1):  # 第i行
            for j in range(wSplit + 1):  # 第j列
                # 计算行和列的起始与结束位置
                row_start = i * sidePixelNums
                col_start = j * sidePixelNums

                # 处理到右侧边缘和底部边缘的情况
                row_end = min(row_start + sidePixelNums, self.GHeight)
                col_end = min(col_start + sidePixelNums, self.GWidth)

                # 如果边缘保留且没有余数则跳过额外的分割块
                if (i == hSplit and hRemainder == 0) and (j == wSplit and wRemainder == 0):
                    continue

                subGSplit = trans[row_start:row_end, col_start:col_end]
                GFS.append(subGSplit)
                if self.G_RGB is not None:
                    subRGB = self.G_RGB[row_start:row_end, col_start:col_end, :]
                    GrgbS.append(subRGB)

        for (y_id, x_id), point_idx in self.GPtSurface.items():
            hangNum, hangYu = divmod(y_id, sidePixelNums)  # 行块数,0开始（商、余数）
            lie_Num, lie_Yu = divmod(x_id, sidePixelNums)  # 列块数,0开始（商、余数）
            splitnums = hangNum * (wSplit + 1) + lie_Num  # 第几块
            GFS[splitnums][hangYu, lie_Yu] = self.feature_non_negative[point_idx[0]]

        self.GFeatureSplit = []  # 创建列表，存放分割后的Feature矩阵
        self.G_RGBsplit = []  # 创建列表，存放分割后的RGB矩阵

        if edgeReserve is False:
            self.GFeatureSplit = GFS
            self.GSplitNums = len(self.GFeatureSplit)  # 总块数
        else:
            kuan = wSplit + 1  # 列数

            # 合并

            def chuLiYuKuai(muBiaoArray, addArray):
                temp = list(muBiaoArray)  # 复制一个临时列表用于操作，避免在同一个数组上操作
                # 处理多余列向左合并
                for i in range(hSplit + 1):
                    if wRemainder != 0:  # 只有当有余数时才处理
                        s = i * kuan + wSplit  # 最后一列的块
                        if muBiaoArray[s].shape[1] != sidePixelNums:  # 如果宽度不足
                            temp[s - 1] = np.hstack((temp[s - 1], temp[s]))  # 向左合并
                # 处理多余行向上合并
                for j in range(wSplit + 1):
                    if hRemainder != 0:  # 只有当有余数时才处理
                        s = hSplit * kuan + j  # 最后一行的块
                        if temp[s].shape[0] != sidePixelNums:  # 如果高度不足
                            temp[s - kuan] = np.vstack((temp[s - kuan], temp[s]))  # 向上合并
                for s in range(len(temp)):
                    if temp[s].shape[0] >= sidePixelNums and temp[s].shape[1] >= sidePixelNums:
                        addArray.append(temp[s])

            chuLiYuKuai(GFS, self.GFeatureSplit)
            self.GSplitNums = len(self.GFeatureSplit)  # 总块数
            if self.G_RGB is not None:
                chuLiYuKuai(GrgbS, self.G_RGBsplit)

    ####################################################################################################################
    # 空值替换
    def replace_null_toMin(self, array, replaceValue: int = None):
        """
        将空值(-1)进行替换replaceValue

        Args:
            array:输入像素矩阵，可二维的线性特征，也可三维的RGB特征
            replaceValue:替换值，默认为array的除(-1)外最小值

        Returns:
            取代空值(-1)处理后的像素矩阵

        """
        # 获取空值索引
        arrayto2D = None
        arrayOutput = array.copy()
        if isinstance(array, np.ndarray) and array.ndim == 2:
            print(f'二维数据[H×W]进行 空值(-1)进行替换f{replaceValue}')
            arrayto2D = array.copy()
        # 一维特征归一化
        elif isinstance(array, np.ndarray) and array.ndim == 3:
            print(f'三维数据[H×W×3]通过转二维数据[H×W]进行 空值(-1)进行替换f{replaceValue}')
            arrayto2D = np.sum(array**3, axis=2) / 3
        else:
            raise ValueError('要替换的数据存在错误')

        if replaceValue is None:
            # 将数组展平成一维数组
            arr_flattened = array.flatten()
            # 找到最小值
            min_value = np.min(arr_flattened)
            # 排除最小值后找到第二小的值
            second_smallest = np.min(arr_flattened[arr_flattened > min_value])
            replaceValue = second_smallest

        # 使用 np.where 获取符合条件的索引
        indices = np.where(arrayto2D == -1)
        if isinstance(array, np.ndarray) and array.ndim == 2:
            arrayOutput[tuple(indices)] = replaceValue
        elif isinstance(array, np.ndarray) and array.ndim == 3:
            arrayOutput[tuple(indices)] = [replaceValue, replaceValue, replaceValue]
        return arrayOutput

    def replace_null_localAverage(self, array, localRange: int = 1, replaceValue: float = None):
        """
        对每个空值(-1)的周围像素范围进行处理，若周围全为空值，二维数据中填充replaceValue，
        RGB数据则填充[replaceValue, replaceValue, replaceValue]，否则填充所有非空像素的平均值。

        Args:
            array (numpy.ndarray): 输入的像素矩阵，可以是二维的灰度图像或三维的 RGB 图像。
            localRange (int): 判定周围像素的范围，默认为 1。范围越大，计算的平均值越平滑。
            replaceValue (optional): 用于替换所有邻域像素为 -1 的区域的默认值。如果为 None，二维数据会自动计算全局第二小(即除-1外)的值作为替换值；三维数据默认使用 [0, 0, 0]。

        Returns:
            numpy.ndarray: 处理后的像素矩阵，其中原始空值(-1)位置被替换为计算出的值。
        """
        # 获取空值索引
        arrayto2D = None

        arrayOutput = array.copy()
        if isinstance(array, np.ndarray) and array.ndim == 2:
            print('二维数据[H×W]进行周边平均像素替换')
            arrayto2D = array.copy()
            if replaceValue is None:
                # 将数组展平成一维数组
                arr_flattened = array.flatten()
                # 找到最小值
                min_value = np.min(arr_flattened)
                # 排除最小值后找到第二小的值
                replaceValue = np.min(arr_flattened[arr_flattened > min_value])
        elif isinstance(array, np.ndarray) and array.ndim == 3:
            print('三维数据[H×W×3]通过转二维数据[H×W]进行周边平均像素替换')
            arrayto2D = np.sum(array**3, axis=2) / 3
            replaceValue = np.array([0, 0, 0])
        else:
            raise ValueError('要替换的数据存在错误')

        # 使用 np.where 获取符合条件的索引
        indices = np.where(arrayto2D == -1)

        # 遍历所有符合条件的元素
        for i in range(len(indices[0])):
            row, col = indices[0][i], indices[1][i]  # 获取当前像素的行row，列col
            localPixlLength = localRange * 2 + 1  # 获取周围像素的边长数目
            localValue = None
            if isinstance(array, np.ndarray) and array.ndim == 2:
                localValue = 0
            elif isinstance(array, np.ndarray) and array.ndim == 3:
                localValue = np.array([0, 0, 0])
            localNums = 0
            for m in range(localPixlLength):
                for n in range(localPixlLength):
                    rowlocal = row - localRange + m
                    collocal = col - localRange + n
                    if 0 <= rowlocal < array.shape[0] and 0 <= collocal < array.shape[1]:  # 不超过索引范围
                        if rowlocal != row and collocal != col:  # 不选取中心点
                            if arrayto2D[rowlocal, collocal] != -1:  # 只采集非-1的值
                                localNums = localNums + 1
                                localValue = localValue + array[rowlocal, collocal]

            if isinstance(array, np.ndarray) and array.ndim == 2:
                if localNums == 0:  # 确保被除数非零
                    arrayOutput[row, col] = replaceValue
                else:
                    arrayOutput[row, col] = localValue / localNums
            elif isinstance(array, np.ndarray) and array.ndim == 3:
                if localNums == 0:  # 确保被除数非零
                    arrayOutput[row, col, :] = replaceValue
                else:
                    arrayOutput[row, col, :] = localValue / localNums
        return arrayOutput

    def replace_null_withNearest(self, array):
        """
        【将二维数组中 所有的空值(-1) 替换为其 最近的非零元素的值 】【将三维数组中 通过立方和除三 转化为二维数组，再执行前面的操作】

        Args:
            array:元素组

        Returns:
            替换所有的空值(-1)后的二维数组

        """

        arrayto2D = None
        # 计算数组中值为 非零 的点到最近 零点的距离，并同时返回最近非零值点的索引
        # distance, nearest_indices = distance_transform_edt(array, return_indices=True)
        # 计算数组中值为 0 的点到最近非零值点的距离，并同时返回最近非零值点的索引
        # distance, nearest_indices = distance_transform_edt(arrayto2D == 0, return_indices=True)
        # nearest_indices是（2×（array.shape））
        # nearest_indices[0][i, j] 是前景像素 (i, j) 最近的背景像素的行索引，nearest_indices[1][i, j] 是列索引。
        if isinstance(array, np.ndarray) and array.ndim == 2:
            print('二维数据[H×W]进行 最邻近替换')
            arrayto2D = array
        # 一维特征归一化
        elif isinstance(array, np.ndarray) and array.ndim == 3:
            print('三维数据[H×W×3]通过转二维数据[H×W]进行 最邻近替换')
            arrayto2D = np.sum(array**3, axis=2) / 3
        else:
            raise ValueError('要替换的数据存在错误')

        _, nearest_indices = distance_transform_edt(arrayto2D == -1, return_indices=True)

        # tuple(nearest_indices)转换为二维索引，用最接近的非零元素的值替换零
        filled_array = array[tuple(nearest_indices)]
        return filled_array

    ####################################################################################################################
    # 一维数据拉伸处理

    def dataStretch(self, nullProcessResult, stretchMethod: int = 1):
        stretchJieGuo = None
        print('Start Stretch')
        if stretchMethod == 1:  # 简单归一化
            # "np.clip(feature, minValue, maxValue)"
            # "使用np.clip将小于minValue的元素替换为minValue，将大于maxValue的元素替换为maxValue" 65520
            "每幅图像的是否拉伸以及拉伸的范围都是不一样的，depth一般是不拉伸的"
            "先用featureDistribution观察数据分布，调试拉伸的范围，设置完范围后featureDistribution必须注释掉"
            # featureDistribution(feature_non_negative, 100)
            jiequMin = 65520
            jiequMax = nullProcessResult.max()
            stretchJieGuo = np.clip(nullProcessResult, jiequMin, jiequMax)
            print(f'截取的范围为：{jiequMin}~{jiequMax}')
            # featureDistribution(feature_non_negative, 10)
        elif stretchMethod == 2:  # 使用 QuantileTransformer 进行正态分布拉伸
            transformer_normal = QuantileTransformer(output_distribution='normal')
            stretchJieGuo = transformer_normal.fit_transform(nullProcessResult)
        return stretchJieGuo

    ####################################################################################################################
    # 加载图像数据

    def PrepareImg(self, showObject: np, nullMethod: int = 1):
        """
        处理二维或三维图像矩阵，支持不同的空值处理方法，最后返回 特征映射'plasma'数据 或 RGB数据。

        Args:
            showObject (np.ndarray): 输入的二维数组，可以是二维的特征矩阵或三维的 RGB 图像矩阵。
            nullMethod (int): 指定处理空值的方法。可以选择以下值:
                              【1 - 使用局部平均替换空值，周围全是空值就替换为图幅最小值（默认最小值，可调）】
                              【2 - 使用最近邻值替换空值】
                              【3 - 将空值替换为数组最小值（默认最小值，可调）】
                              默认值为 1。

        Returns:
            np.ndarray: 处理后的图像矩阵。如果是二维输入，则返回灰度图；如果是三维输入，则返回 RGB 图像。

        Raises:
            ValueError: 当 `ShowObject` 不是二维或三维的 NumPy 数组，或 `ShowMethod` 不在有效范围内时，抛出此异常。
        """

        # 检查 ShowObject 是否为二维或三维的 NumPy 数组
        if isinstance(showObject, np.ndarray) and (showObject.ndim == 2 or showObject.ndim == 3):
            print('开始栅格转图像...')
        else:
            raise ValueError('ShowObject存在错误')

        vis_map = None  # 初始化变量用于存储处理后的图像矩阵

        # 根据 ShowMethod 选择不同的空值处理方法
        if nullMethod == 1:
            # 使用局部平均替换空值
            vis_map = self.replace_null_localAverage(showObject, replaceValue=None)
        elif nullMethod == 2:
            # 使用最近邻值替换空值
            vis_map = self.replace_null_withNearest(showObject)
        elif nullMethod == 3:
            # 将空值替换为最小值
            vis_map = self.replace_null_toMin(showObject, replaceValue=0)
        else:
            # 如果 ShowMethod 超出有效范围，则抛出错误
            raise ValueError('输入格式不对 或 输入范围超限')

        imageTX = None  # 初始化变量用于存储最终显示的图像

        if isinstance(showObject, np.ndarray) and (showObject.ndim == 2):  # 如果输入的是二维数组，则应用灰度 colormap 映射
            print(f'min:{vis_map.min()},max:{vis_map.max()}')
            # ！！！！！！！！！！！！！！！！不是所有特征都需要拉伸
            # vis_map = self.dataStretch(vis_map, stretchMethod=2)
            # 对处理后的图像进行归一化
            vis_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min())
            # 使用 colormap 进行映射
            imageTX = mpl.colormaps['plasma'](vis_map)  # 'plasma效果最好'  'viridis效果其次'  'Greys'  'binary'
        elif isinstance(showObject, np.ndarray) and (showObject.ndim == 3):  # 如果输入的是三维 RGB 图像，则直接使用处理后的矩阵
            imageTX = vis_map

        return imageTX

        ####################################################################################################################
        # 显示模块

    def ShowGrid_orSave(self, showObject: np, nullMethod: int = 1, SavePath: str = None):
        """
        显示或保存二维网格图像，支持不同的特征映射'gray'和显示方法。

        Args:
            showObject (np.ndarray): 输入的二维数组，用于显示或处理。通常为特征矩阵或RGB图像矩阵。
            nullMethod (int): 指定处理空值的显示方法。可以选择以下值:
                              【1 - 使用局部平均替换空值，周围全是空值就替换为图幅最小值（默认最小值，可调）】
                              【2 - 使用最近邻值替换空值】
                              【3 - 将空值替换为数组最小值（默认最小值，可调）】
                              默认值为 1。
            SavePath (str): 指定保存图像的路径。如果提供了此参数，图像将被保存到该路径，而不会在屏幕上显示。
                            如果此参数为 None，图像将直接显示在屏幕上。默认值为 None。

        Returns:
            None: 该函数没有返回值。图像要么被保存到指定路径，要么显示在屏幕上。

        Raises:
            ValueError: 当 `ShowObject` 不是 'Feature' 或 'RGB'，或者 `ShowMethod` 不是有效值时，抛出此异常。
        """
        imageTX = self.PrepareImg(showObject, nullMethod)

        # 如果指定了保存路径，则保存图像
        if SavePath is not None:
            plt.imsave(SavePath, imageTX)
        else:
            # 如果没有指定保存路径，则显示图像
            plt.imshow(imageTX)
            plt.show()

    def ShowVisGrid(self, ShowMethod: int = 1):
        """
        实现图像的多 colormap(特征映射)显示，映射colormap=gray

        Args:
            ShowMethod (int): 指定处理空值的显示方法。可以选择以下值:
                              【1 - 使用局部平均替换空值】
                              【2 - 使用最近邻值替换空值】
                              【3 - 将空值替换为零】
                              默认值为 1。

        Returns:
            None

        """

        vis_map = None
        if ShowMethod == 1:
            vis_map = self.replace_null_localAverage(self.GFeature, replaceValue=None)
        elif ShowMethod == 2:
            vis_map = self.replace_null_withNearest(self.GFeature)
        elif ShowMethod == 3:
            vis_map = self.replace_null_toMin(self.GFeature, replaceValue=None)
        else:
            raise ValueError('输入格式不对 或 输入范围超限')
        # 将 self.G 归一化到 [0, 255] 范围，并转换为无符号 8 位整数类型
        vis_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min())
        # 定义要显示的colormap列表
        colormaps = ['viridis', 'plasma', 'rainbow', 'binary', 'gray', 'gist_gray']
        # 创建一个 2 行 3 列的子图，图像大小为 12x8 英寸
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for ax, cmap in zip(axes.flat, colormaps):
            ax.imshow(vis_map, cmap=cmap)
            ax.set_title(cmap)
            # 关闭当前子图的坐标轴
            ax.axis('off')

        # 调整子图间距，使其更紧凑
        plt.tight_layout()
        plt.show()

        ####################################################################################################################
        # 没用上的

    ####################################################################################################################
    # 其他
    def GetIJ(self, vid):
        if vid > (self.GWidth * self.GHeight - 1):
            vid = self.GWidth * self.GHeight - 1  # Out of bound
        if vid < 0:
            vid = 0
        j = math.floor(vid / self.GWidth)
        i = math.floor(vid - j * self.GWidth)
        return i, j

    def rotate_point_cloud(self, points, axis, angle):
        """
        Rotates a 3D point cloud around a specific axis by a given angle.

        Parameters:
        points (numpy.ndarray): A numpy array of shape (N, 3) representing the point cloud.
        axis (str): The axis to rotate around ('x', 'y', or 'z').
        angle (float): The angle to rotate by, in radians.

        Returns:
        numpy.ndarray: The rotated point cloud.
        """
        # Define rotation matrices
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        if axis == 'x':
            rotation_matrix = np.array([[1, 0, 0],
                                        [0, cos_angle, -sin_angle],
                                        [0, sin_angle, cos_angle]])
        elif axis == 'y':
            rotation_matrix = np.array([[cos_angle, 0, sin_angle],
                                        [0, 1, 0],
                                        [-sin_angle, 0, cos_angle]])
        elif axis == 'z':
            rotation_matrix = np.array([[cos_angle, -sin_angle, 0],
                                        [sin_angle, cos_angle, 0],
                                        [0, 0, 1]])
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

        # Rotate points
        rotated_points = np.dot(points, rotation_matrix.T)

        return rotated_points

    def GetGrid_3_Neighbours_Down(self, vid):
        '''
        3 neighbours, lower layer
        '''
        i, j = self.GetIJ(vid)
        down = self.GetIJ(i - 1, j)  # left right upper down
        downL = self.GetIJ(i - 1, j - 1)
        downR = self.GetIJ(i - 1, j + 1)
        return [down, downL, downR]

    def PrepareLines(self):
        H_Dir = np.arange(0.0, self.GSize * (self.GWidth + 1), self.GSize)
        H_Vertices = np.ndarray([2 * len(H_Dir), 3])
        H_lines = np.ndarray([len(H_Dir), 2])
        for i in range(len(H_Dir)):
            H_Vertices[i] = [H_Dir[i], 0, 0]
            H_Vertices[i + len(H_Dir)] = [H_Dir[i], self.GSize * self.GHeight, 0]
            H_lines[i] = [i, i + len(H_Dir)]
        self.H_Line_set.points = o3d.utility.Vector3dVector(H_Vertices)
        self.H_Line_set.lines = o3d.utility.Vector2iVector(H_lines)
        ##
        V_Dir = np.arange(0.0, self.GSize * (self.GHeight + 1), self.GSize)
        V_Vertices = np.ndarray([2 * len(V_Dir), 3])
        V_lines = np.ndarray([len(V_Dir), 2])
        for i in range(len(V_Dir)):
            V_Vertices[i] = [0, V_Dir[i], 0]
            V_Vertices[i + len(V_Dir)] = [self.GSize * self.GWidth, V_Dir[i], 0]
            V_lines[i] = [i, i + len(V_Dir)]
        self.V_Line_set.points = o3d.utility.Vector3dVector(V_Vertices)
        self.V_Line_set.lines = o3d.utility.Vector2iVector(V_lines)

    def PrepareClouds(self, pFeature):
        self.pcd.points = o3d.utility.Vector3dVector(self.Pt)
        I_p = 1.0 - copy.deepcopy(pFeature)
        cm = plt.get_cmap("rainbow")
        intensity = np.array(I_p) / max(I_p)
        intensity = cm(intensity)[:, 0: 3]
        self.pcd.colors = o3d.utility.Vector3dVector(intensity)

    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, m * 0.35)))
        # ax[1].set_title('Generated Masks')
        # ax[1].axis('off')
        # plt.show()


if __name__ == "__main__":
    print('该文件是方法库')
    completeImgSplit(
        '/media/huiwei/date/dataALL/ALS dataset/H3D dataset/Mar18_val/class_20240924_102056/Mar18_val_class.png',
        1000,
        True)
