import scipy.io
import torch
from torchvision import transforms, datasets,models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
# from vggNet import vgg
from mobilenetv3_model_cbam_psdp import mobilenet_v3_large as create_model
import time
# 修改系统的图片显示
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

data_transform = transforms.Compose(
    [transforms.Resize((128, 128)),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# load image

# img = Image.open("/home/cqmu/lqn/Test6_mobilenetv3/Test6_mobilenet/26_25/25_medical50_5.bmp")
img = Image.open("/home/cqmu/lqn/data_set/medical_photos/70/69_medical615.bmp")
img = img.convert('RGB')
# [N, C, H, W]C

img = data_transform(img)

# expand batch dimension

img = torch.unsqueeze(img, dim=0)
# print(img.shape)  #[N,C,H,W]-->[1,3,128,128]


# create model
# load new model
# model = models.vgg16(pretrained=False) # pretrained=True: 同时加载预训练模型和预训练模型的参数
model = create_model(num_classes=130)
startTime = time.time()
# load model weights
# model_weight_path = "/home/cqmu/lqn/Test6_mobilenet/weights/10.7 mobilenetv3_model-81.pth"
model_weight_path = "/home/cqmu/lqn/Test6_mobilenetv3/Test6_mobilenet/2024.3.25 21:53 weights_/model-97.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()
# print(model)
# # torch.save(model,"/home/cqmu/harddisk/lqn/Test6_mobilenet/mobilenetv2model.png")
# # forwd 输出特征图
# # print(img.shape)  #[1,3,128,128]
out_put = model(img)
print("查看的卷积层数:",len(out_put))
# # #
# # # 遍历所要查看的卷积层
j = 121
for feature_map in out_put:
#    # [N, C, H, W] -> [C, H, W]
#    # print(feature_map.shape)
    im = np.squeeze(feature_map.detach().numpy())
    # print(im.shape)   #(160,4,4)
#    # [C, H, W] -> [H, W, C] = = [height, weight, numbers_feature_map]
    im = np.transpose(im, [1, 2, 0])
    print("该层卷积层的特征图维度:")
    print(im.shape)    #(4,4,160)
# #     # feature_size=im.shape[2]
# # #     # show top 12 feature maps
# # #     # 遍历样本图在该层卷积中的16张特征图
# # #     plt.figure()
# # #     for i in range(16):
# # #         ax = plt.subplot(4, 4, i + 1)
# # #         # [H, W, N]: im[:, :a, i]第i张特征图的信息
# # #         plt.imshow(im[:, :, i])
# # #         # plt.imshow(im[:, :, i])
# # #     plt.show()
    # 把每一个卷积层存储为.mat矩阵
    file_name = 'data' + str(j) + '.mat'
    scipy.io.savemat(file_name, {'data'+str(j): im[:, :, :]})
    j=j+1
endTime = time.time()
needTime = endTime - startTime
print("程序运行时间：%.2f秒" % needTime)




