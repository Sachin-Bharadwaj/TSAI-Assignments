import torchvision
import torch.utils.data as data
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_CIFAR10dataset(root="./data", train_flag=True, download_flag=True):
    dataset = torchvision.datasets.CIFAR10(
        root=root, train=train_flag, download=download_flag
    )
    return dataset


def get_CIFAR10_musigma():
    return ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


def get_transforms(basic=True, tfms_list=None):
    if basic:
        return A.Compose(
            [
                A.Normalize(
                    mean=get_CIFAR10_musigma()[0], std=get_CIFAR10_musigma()[1]
                ),
                ToTensorV2(),
            ]
        )
    elif tfms_list is None:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    rotate_limit=15, scale_limit=0.1, shift_limit=0.0625, p=0.5
                ),
                A.CoarseDropout(
                    max_holes=1,
                    max_height=16,
                    max_width=16,
                    min_holes=1,
                    min_height=16,
                    min_width=16,
                    fill_value=get_CIFAR10_musigma()[0],
                    mask_fill_value=None,
                    p=0.5,
                ),
                A.Normalize(
                    mean=get_CIFAR10_musigma()[0], std=get_CIFAR10_musigma()[1]
                ),
                ToTensorV2()
            ]
        )
    else:
        return A.Compose(
            tfms_list +
            [A.Normalize(
                mean=get_CIFAR10_musigma()[0], std=get_CIFAR10_musigma()[1]
            )] +
            [ToTensorV2()]
        )


def get_dataloader(dataset, bs_cuda=512, bs_cpu=64, device="cuda"):
    dataloader_args = (
        dict(shuffle=True, batch_size=bs_cuda)
        if device == "cuda"
        else dict(shuffle=True, batch_size=bs_cpu)
    )
    return data.DataLoader(dataset, **dataloader_args)


class CIFAR10_dataset(data.Dataset):
    def __init__(self, data, targets, transforms=None):
        self.data = data
        self.targets = targets
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        X = self.data[item]
        y = self.targets[item]
        if self.transforms is not None:
            X = self.transforms(image=X)["image"]
        return X, y
