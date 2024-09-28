import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import pickle
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    print('开始加载SAM模型参数...')
    # 指定 SAM 模型的检查点文件名
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    # 指定使用的模型类型
    model_type = "vit_h"
    # 指定使用的设备为 CUDA（即 GPU）
    device = "cuda"
    # 从 sam_model_registry 中获取指定模型类型的模型，并加载相应的检查点
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # 将模型移动到指定的设备上
    sam.to(device=device)
    # 创建一个 SamAutomaticMaskGenerator 实例，用于自动生成掩码
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,  # 指定使用的模型
        points_per_side=64,  # 每个侧面的点数

        pred_iou_thresh=0.75,  # 默认0.88；越接近 1.0，意味着仅保留高质量的掩码。降低此值将生成更多但质量较低的掩码。
        stability_score_thresh=0.75,  # 默认0.95；值越高，模型生成的掩码形状越稳定。

        stability_score_offset=0.8,  # 计算稳定性分数时，偏移截止值的量。默认值为1.0
        crop_n_layers=0,  # 裁剪层数
        crop_n_points_downscale_factor=0,  # 掩码的最小区域。小于该区域的掩码将被忽略，以减少生成不必要的小掩码。
        min_mask_region_area=100,  # 最小掩码区域面积（需要 OpenCV 进行后处理）
    )

    # 选择 特征存放文件夹
    featureFolder = '/media/huiwei/date/dataALL/ALS dataset/H3D dataset/Mar18_val/depth_0'

    print('模型加载完毕，执行SAM')
    for root, dirs, files in os.walk(featureFolder):
        for file in files:
            if '.png' in file and 'rgb' not in file.lower() and 'sam' not in file.lower():  # 将文件名转换为小写后进行比较
                print(f'(SAM)正在执行：{file}')

                file_path = os.path.join(root, file)  # 获取文件的完整路径
                img = plt.imread(file_path)
                # 如果图像是浮点类型，将其转换为uint8类型
                # 这个步骤将图像数据从0-1的浮点数范围转换为0-255的uint8范围
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (255 * img).astype(np.uint8)
                # 使用掩码生成器生成深度图像的掩码
                masks_Feature = mask_generator.generate(img[:, :, :-1])  # 核心耗时
                # 根据掩码的面积对掩码进行降序排序
                masks_Feature = sorted(masks_Feature, key=(lambda x: x['area']), reverse=True)

                masksname = file.split('.')[0] + '_SAM.pkl'
                masksName = os.path.join(root, masksname)  # 获取文件的完整路径
                with open(masksName, 'wb') as f:  # 'wb' 表示写入二进制文件
                    pickle.dump(masks_Feature, f)
