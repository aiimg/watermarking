import scipy.io
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model1 import mobilenet_v3_large as create_model
import time
# modify the system image display
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

data_transform = transforms.Compose(
    [transforms.Resize((128, 128)),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# load image
img = Image.open("/home/cqmu/lqn/data_set/medical_photos/70/69_medical.bmp")
img = img.convert('RGB')
# [N, C, H, W]C
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)
# print(img.shape)  #[N,C,H,W]-->[1,3,128,128]
# create model
# load new model
model = create_model(num_classes=130)
startTime = time.time()
# load model weights
model_weight_path = "/home/cqmu/lqn/mobilenetv3/weights/model_mobilenetv3.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()
out_put = model(img)
print("The number of convolution layers:",len(out_put))
j = 121
for feature_map in out_put:
    im = np.squeeze(feature_map.detach().numpy())
    im = np.transpose(im, [1, 2, 0])
    print("The feature map dimension of convolution layer:")
    print(im.shape)
    # store each convolution layer as a .mat matrix
    file_name = 'data' + str(j) + '.mat'
    scipy.io.savemat(file_name, {'data'+str(j): im[:, :, :]})
    j=j+1
endTime = time.time()
needTime = endTime - startTime
print("program run time: %.2f s" % needTime)




