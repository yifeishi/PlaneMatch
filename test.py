import os
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataset_planematch import *
from network_planematch import *
import util

config = util.get_args()
transformed_dataset = PlanarPatchDataset(csv_file=config.test_csv_path, root_dir=config.test_root_dir,transform=transforms.Compose([ToTensor()]))
dataloader = DataLoader(transformed_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

model = torch.load('./model.pkl')
model.cuda(config.gpu)

if not os.path.exists(config.feature_path):
    os.makedirs(config.feature_path)

count = 0
print('Start testing ......')
for i_batch, sample_batched in enumerate(dataloader):
    sample = transformed_dataset[i_batch]
    x1 = sample_batched['rgb_global_image1'].float()
    x2 = sample_batched['rgb_global_image2'].float()
    x3 = sample_batched['rgb_global_image3'].float()
    x4 = sample_batched['depth_global_image1'].float()
    x5 = sample_batched['depth_global_image2'].float()
    x6 = sample_batched['depth_global_image3'].float()
    x7 = sample_batched['normal_global_image1'].float()
    x8 = sample_batched['normal_global_image2'].float()
    x9 = sample_batched['normal_global_image3'].float()
    x10 = sample_batched['mask_global_image1'].float()
    x11 = sample_batched['mask_global_image2'].float()
    x12 = sample_batched['mask_global_image3'].float()
    x13 = sample_batched['rgb_local_image1'].float()
    x14 = sample_batched['rgb_local_image2'].float()
    x15 = sample_batched['rgb_local_image3'].float()
    x16 = sample_batched['depth_local_image1'].float()
    x17 = sample_batched['depth_local_image2'].float()
    x18 = sample_batched['depth_local_image3'].float()
    x19 = sample_batched['normal_local_image1'].float()
    x20 = sample_batched['normal_local_image2'].float()
    x21 = sample_batched['normal_local_image3'].float()
    x22 = sample_batched['mask_local_image1'].float()
    x23 = sample_batched['mask_local_image2'].float()
    x24 = sample_batched['mask_local_image3'].float()

    x1 = Variable(x1.cuda(config.gpu), requires_grad=True)
    x2 = Variable(x2.cuda(config.gpu), requires_grad=True)
    x3 = Variable(x3.cuda(config.gpu), requires_grad=True)
    x4 = Variable(x4.cuda(config.gpu), requires_grad=True)
    x5 = Variable(x5.cuda(config.gpu), requires_grad=True)
    x6 = Variable(x6.cuda(config.gpu), requires_grad=True)
    x7 = Variable(x7.cuda(config.gpu), requires_grad=True)
    x8 = Variable(x8.cuda(config.gpu), requires_grad=True)
    x9 = Variable(x9.cuda(config.gpu), requires_grad=True)
    x10 = Variable(x10.cuda(config.gpu), requires_grad=True)
    x11 = Variable(x11.cuda(config.gpu), requires_grad=True)
    x12 = Variable(x12.cuda(config.gpu), requires_grad=True)
    x13 = Variable(x13.cuda(config.gpu), requires_grad=True)
    x14 = Variable(x14.cuda(config.gpu), requires_grad=True)
    x15 = Variable(x15.cuda(config.gpu), requires_grad=True)
    x16 = Variable(x16.cuda(config.gpu), requires_grad=True)
    x17 = Variable(x17.cuda(config.gpu), requires_grad=True)
    x18 = Variable(x18.cuda(config.gpu), requires_grad=True)
    x19 = Variable(x19.cuda(config.gpu), requires_grad=True)
    x20 = Variable(x20.cuda(config.gpu), requires_grad=True)
    x21 = Variable(x21.cuda(config.gpu), requires_grad=True)
    x22 = Variable(x22.cuda(config.gpu), requires_grad=True)
    x23 = Variable(x23.cuda(config.gpu), requires_grad=True)
    x24 = Variable(x24.cuda(config.gpu), requires_grad=True)
    feature1 = model(x1, x4, x7, x10, x13, x16, x19, x22)
    feature2 = model(x2, x5, x8, x11, x14, x17, x20, x23)

    if count == 0:
        featureAll1 = feature1.data.cpu().numpy()
        featureAll2 = feature2.data.cpu().numpy()
    else:
        featureAll1 = np.vstack((featureAll1,feature1.data.cpu().numpy()))
        featureAll2 = np.vstack((featureAll2,feature2.data.cpu().numpy()))
    for i in range(0,feature1.data.cpu().numpy().shape[0]):
        dist1 = np.linalg.norm(feature1.data.cpu().numpy()[i]-feature2.data.cpu().numpy()[i])
        print('feature distance:',dist1)
    count = count + 1
    
print('Writing feature to file...')
np.savetxt(config.feature_path+"/feature1.txt",featureAll1, fmt="%f", delimiter="  ")
np.savetxt(config.feature_path+"/feature2.txt",featureAll2, fmt="%f", delimiter="  ")
np.savetxt(config.feature_path+"/feature3.txt",featureAll3, fmt="%f", delimiter="  ")
print('Finished...')