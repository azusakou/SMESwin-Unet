import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from skimage import transform as tf
import albumentations as A
import torchvision.transforms as transforms

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def transform_add(image, mask):
    image = np.array(image)
    mask = np.array(mask)

    trans = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.5),  # 使用随机大小的内核将运动模糊应用于输入图像。
            A.MedianBlur(blur_limit=3, p=0.5),  # 中值滤波
            A.Blur(blur_limit=3, p=0.2),  # 使用随机大小的内核模糊输入图像。
        ], p=1),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
        # 随机应用仿射变换：平移，缩放和旋转输入
        A.RandomBrightnessContrast(p=0.5),  # 随机明亮对比度
    ])
    #trans = A.Compose([A.HorizontalFlip(p=0.5),A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5)])
    trans_results = trans(image=image, mask=mask)
    return trans_results

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        trans_results =  transform_add(image, label)
        image = trans_results['image']
        label = trans_results['mask']
        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        image = transform_img(image)
        #label = torch.from_numpy(label)

        channel, x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            #image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            image = tf.resize(image, (3,self.output_size[0], self.output_size[1]), order=0)
            #label = tf.resize(label, (self.output_size[0], self.output_size[1], 3), order=0)

            image = torch.from_numpy(image.astype(np.float32))#.transpose(2, 0, 1))#.unsqueeze(0) transpose(2, 0, 1) # 转换为：（c,h,w)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class MoNuSeg_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name +'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']

        else:
            vol_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, vol_name + '.npz')
            label = np.load(data_path)['label']
            images = []
            repatches_image = np.zeros([9, 9, 1, 224, 224, 3], int)
            for h in range(9):
                for w in range(9):
                    data = np.load(os.path.join(self.data_dir, vol_name + '_h' + str(h) + '_w' + str(w) + '.npz'))['image']
                    images.append(data)
            c = -1
            for i in range(9):
                for k in range(9):
                    c += 1
                    repatches_image[i][k][0] = images[c]
            image = repatches_image

        sample = {'image': image, 'label': label}
        if self.split == "train":
            if self.transform:
                sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample