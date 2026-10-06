import os
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import custom_transforms as tr
import math


class RemoteData(data.Dataset):
    def __init__(self, base_dir='./data/', train=None, dataset='barley', crop_size=None,
                 val_full_img=False, split=None):
        super(RemoteData, self).__init__()
        self.dataset_dir = base_dir
        if split is None:
            split = 'train' if train is not False else 'test'
        if split not in {'train', 'val', 'test'}:
            raise ValueError("split must be one of {'train', 'val', 'test'}")
        self.split = split
        self.train = split == 'train'
        self.dataset = dataset
        self.val_full_img = val_full_img
        self.images = []
        self.labels = []
        self.names = []
        self.alphas = []
        
        if crop_size is None:
            crop_size = [512, 512]
        self.crop_size = crop_size
        
        self.image_dir = os.path.join(self.dataset_dir, self.dataset + '/images')
        self.label_dir = os.path.join(self.dataset_dir, self.dataset + '/labels')
        
        txt = os.path.join(self.dataset_dir, self.dataset, 'annotations', f'{self.split}.txt')

        with open(txt, "r") as f:
            self.filename_list = f.readlines()
            
        for filename in self.filename_list:
            # Load image and extract alpha channel if present
            image_path = os.path.join(self.image_dir, filename.strip() + '.png')
            image = Image.open(image_path)
            image = np.array(image)
            alpha = None
            if image.shape[2] == 4:
                alpha = image[..., 3]
            image = image[..., 0:3]
            
            # Load label
            label_path = os.path.join(self.label_dir, filename.strip() + '.png')
            label = Image.open(label_path)
            label = np.array(label)
            
            if self.val_full_img:
                self.images.append(image)
                self.labels.append(label)
                self.names.append(filename.strip())
                if alpha is not None:
                    self.alphas.append(alpha)
            else:
                if alpha is not None:
                    slide_crop(image, label, self.crop_size, self.images, self.labels,
                               alpha=alpha, alpha_patches=self.alphas, stride_rate=2/3)
                else:
                    slide_crop(image, label, self.crop_size, self.images, self.labels, stride_rate=2/3)
                    
        assert(len(self.images) == len(self.labels))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = {'image': self.images[index], 'label': self.labels[index]}
        sample = self.transform(sample)
        if self.val_full_img:
            sample['name'] = self.names[index]
        if self.alphas:
            sample['alpha'] = self.alphas[index]
        return sample

    def transform(self, sample):
        if self.train:
            composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.RandomVerticalFlip(),
                tr.ToTensor(add_edge=False),
            ])
        else:
            composed_transforms = transforms.Compose([
                tr.ToTensor(add_edge=False),
            ])
        return composed_transforms(sample)

    def __str__(self):
        return 'dataset:{} split:{}'.format(self.dataset, self.split)


def slide_crop(image, label, crop_size, image_patches, label_patches,
               stride_rate=1.0/2.0, alpha=None, alpha_patches=None):
    """images shape [h, w, c]"""
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    if len(label.shape) == 2:
        label = np.expand_dims(label, axis=2)
    if alpha is not None:
        alpha = np.expand_dims(alpha, axis=2)
        
    stride_rate = stride_rate
    h, w, c = image.shape
    H, W = crop_size
    stride_h = int(H * stride_rate)
    stride_w = int(W * stride_rate)
    assert h >= crop_size[0] and w >= crop_size[1]
    
    h_grids = int(math.ceil(1.0 * (h - H) / stride_h)) + 1
    w_grids = int(math.ceil(1.0 * (w - W) / stride_w)) + 1
    
    for idh in range(h_grids):
        for idw in range(w_grids):
            h0 = idh * stride_h
            w0 = idw * stride_w
            h1 = min(h0 + H, h)
            w1 = min(w0 + W, w)
            
            if h1 == h and w1 != w:
                crop_img = image[h - H:h, w0:w0 + W, :]
                crop_label = label[h - H:h, w0:w0 + W, :]
                if alpha is not None:
                    crop_alpha = alpha[h - H:h, w0:w0 + W, :]
            if w1 == w and h1 != h:
                crop_img = image[h0:h0 + H, w - W:w, :]
                crop_label = label[h0:h0 + H, w - W:w, :]
                if alpha is not None:
                    crop_alpha = alpha[h0:h0 + H, w - W:w, :]
            if h1 == h and w1 == w:
                crop_img = image[h - H:h, w - W:w, :]
                crop_label = label[h - H:h, w - W:w, :]
                if alpha is not None:
                    crop_alpha = alpha[h - H:h, w - W:w, :]
            if w1 != w and h1 != h:
                crop_img = image[h0:h0 + H, w0:w0 + W, :]
                crop_label = label[h0:h0 + H, w0:w0 + W, :]
                if alpha is not None:
                    crop_alpha = alpha[h0:h0 + H, w0:w0 + W, :]
                    
            crop_img = crop_img.squeeze()
            crop_label = crop_label.squeeze()
            
            if alpha is not None:
                crop_alpha = crop_alpha.squeeze()
                # Only keep patch if there is valid data in the alpha channel
                if np.any(crop_alpha > 0):
                    image_patches.append(crop_img)
                    label_patches.append(crop_label)
                    alpha_patches.append(crop_alpha)
            else:
                image_patches.append(crop_img)
                label_patches.append(crop_label)


def label_to_RGB(image, classes=4):
    RGB = np.zeros(shape=[image.shape[0], image.shape[1], 3], dtype=np.uint8)
    palette = [[255, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    for i in range(classes):
        index = image == i
        RGB[index] = np.array(palette[i])
    return RGB


def RGB_to_label(image=None, classes=4):
    palette = [[255, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    label = np.zeros(shape=[image.shape[0], image.shape[1]], dtype=np.uint8)
    for i in range(len(palette)):
        index = image == np.array(palette[i])
        index[..., 0][index[..., 1] == False] = False
        index[..., 0][index[..., 2] == False] = False
        label[index[..., 0]] = i
    return label


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # Adjusted to load barley dataset for testing
    remotedata_train = RemoteData(train=True, dataset='barley')
    dataloader = DataLoader(remotedata_train, batch_size=1, shuffle=False, num_workers=1)

    for ii, sample in enumerate(dataloader):
        im = sample['label'].numpy().astype(np.uint8)
        pic = sample['image'].numpy().astype(np.uint8)
        print(im.shape)
        im = np.squeeze(im, axis=0)
        pic = np.squeeze(pic, axis=0)
        print(im.shape)
        im = np.transpose(im, axes=[1, 2, 0])[:, :, 0:3]
        pic = np.transpose(pic, axes=[1, 2, 0])[:, :, 0:3]
        print(im.shape)
        im = np.squeeze(im, axis=2)
        im = label_to_RGB(im)
        plt.imshow(pic)
        plt.show()
        plt.imshow(im)
        plt.show()
        if ii == 10:
            break
