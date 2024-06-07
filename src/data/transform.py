import random

import torchvision.transforms as T


class TrainTransform:
    def __init__(self, mean, std, size, aug_prob=0.4, re_prob=0.2):
        self.aug_prob = aug_prob
        self.random_crop =  T.Compose(
            T.RandomResizedCrop((256, 256), scale=(0.3, 0.8), ratio=(0.75, 1.3333333333333333)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        )
        self.random_erase = T.Compose(
            T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.RandomErasing(p=re_prob, scale=(0.05, 0.7), ratio=(0.3, 3.3)),
            T.Normalize(mean=mean, std=std)
        )
        self.plain_transform = T.Compose(
            T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        )

    def __call__(self, img):
        prob = random.random()
        if prob > self.aug_prob:
            return self.plain_transform(img)

        W, H = img.size
        if (W * H) < 1000:
            return self.plain_transform(img)

        if  prob <= (self.aug_prob/2):
            return self.random_crop(img)
        else:
            return self.random_erase(img)



class TestTransform:
    def __init__(self, mean, std, size):
        self.transform = T.Compose(
            [
                T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ]
        )

    def __call__(self, img):
        self.transform(img)