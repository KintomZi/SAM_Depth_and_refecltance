运行此项目所使用的环境(不是最低要求，仅作为参考)：pytorch2.4.1+cu121,python3.9,

只运行了H3D数据集中的Mar18_var三维数据

1是将点云进行处理，使三维信息转换为俯视下的平面图片。图片信息可以有RGB 和指定Feature(高度Z,reflectance)的映射.根据SAM选择的colormap是plasma。当然可以处理指定信息，但需要对方法库中的相关参数改动。里面还对图像裁剪进行操作

2是将1生成的png图片导入到SAM大模型中生成Mask,并保存pkl作为中间文件

3是对Mask进行可视化，选择RGB图片作为背景，没有RGB就是使用Feature图片作为背景。同时保存下来。

4是Mask进行合并（因为SAM处理后的Mask是图片的二进制文件），最后对应到开始的对象文件，将Mask覆盖的像素作为类别信息写入到点云之中。同时生成图幅中每个Mask里面Truth的百分占比。

GridManager是存储该项目中的函数与方法。里面有许多方法有可选参数，从而适应不同情况


（左图）=>>RGB图像              （右图）=>>Ground truth图像(class number=11) 

<img src="https://github.com/user-attachments/assets/0c79cf25-c85f-49fe-94b4-ce556dec2753" alt="RGB图像" width="500"/> <img src="https://github.com/user-attachments/assets/2475cccd-641e-42b1-97dc-1d97b8c6a93b" alt="Ground truth图像" width="500"/>


Depth图像(映射到 plasma 中)

<img src="https://github.com/user-attachments/assets/c9006ef5-2cc9-4d23-9b9d-0fe28cef9702" alt="Depth图像" width="500"/>

Reflectance图像(映射到 plasma 中)，但原图数据跨度过大导致对比度太差，故进行的一些拉伸(左：手动Clip,右：自动正态分布)

<img src="https://github.com/user-attachments/assets/8aa049b2-08c3-40b8-8898-88876400c44a" alt="ref_clip" width="500"/> <img src="https://github.com/user-attachments/assets/32b8b6b6-19c0-42b5-bfa7-1d1086d6f0a6" alt="ref_nor" width="500"/> 


Depth to SAM的结果(背景图片：RGB)

<img src="https://github.com/user-attachments/assets/86680dec-d25a-49ba-8e43-130f3b98fc2f" alt="Depth to SAM" width="500"/>

reflectance_clip to SAM(背景图片：RGB)  reflectance_normal to SAM(背景图片：RGB)

<img src="https://github.com/user-attachments/assets/9cafccf2-5c5c-42e7-9fd7-5d0037ba7773" alt="reflectance_clip to SAM" width="500"/>  <img src="https://github.com/user-attachments/assets/650125f0-3271-4424-8d9d-4489388b5390" alt="reflectance_normal to SAM" width="500"/>





