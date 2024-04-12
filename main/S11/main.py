import sys
sys.path.append("../../")
import cv2
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchsummary import summary
import albumentations as A
from torch_lr_finder import LRFinder

from dataloader import get_CIFAR10dataset, CIFAR10_dataset, get_transforms, get_dataloader, get_CIFAR10_musigma
from models.resnet import ResNet18, ResNet34
from train import train_epoch, test_epoch
from utils import show_samples, plot_loss, plot_acc, get_misclassified, get_denormalized_imgs, get_gradcam_img, show_gradcam_plots, show_misclassified_imgs

from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    # set global config
    SEED = 1
    # CUDA?
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("CUDA Available?", device)
    # For reproducibility
    torch.manual_seed(SEED)
    if device == 'cuda':
        torch.cuda.manual_seed(SEED)

    # Get the data
    train_data = get_CIFAR10dataset(root="../../data", train_flag=True, download_flag=True)
    test_data = get_CIFAR10dataset(root="../../data", train_flag=False, download_flag=True)

    # create CIFAR datasets + transforms
    bs = 512
    train_tfms_list = [
        A.PadIfNeeded(min_height=40, min_width=40, border_mode=0, value=get_CIFAR10_musigma()[0], p=1.0),
        # border_mode=0 needs fill value
        A.RandomCrop(height=32, width=32, p=1.0),
        A.HorizontalFlip(p=0.5),
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
        )
    ]

    train_tfms = get_transforms(basic=False, tfms_list=train_tfms_list)
    test_tfms = get_transforms(basic=True, tfms_list=None)

    train_ds = CIFAR10_dataset(data=train_data.data, targets=train_data.targets, transforms=train_tfms)
    test_ds = CIFAR10_dataset(data=test_data.data, targets=test_data.targets, transforms=test_tfms)

    train_dl = get_dataloader(train_ds, bs_cuda=bs, bs_cpu=64, device=device)
    test_dl = get_dataloader(test_ds, bs_cuda=bs, bs_cpu=64, device=device)

    # visualize some samples
    # get some random training images
    dataiter = iter(train_dl)
    images, labels = next(dataiter)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # show images
    show_samples(torchvision.utils.make_grid(images[:4]))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    # Build the network
    model = ResNet18().to(device)

    # Run LR finder
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-7)
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_dl, end_lr=100, num_iter=100)
    lr_finder.plot()  # to inspect the loss-learning rate graph
    lr_finder.reset()  # to reset the model and optimizer to their initial state

    # Train the model
    EPOCHS = 20
    criterion = nn.CrossEntropyLoss()
    criterion_test = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=1e-7)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=1e-3, epochs=EPOCHS,
                                              steps_per_epoch=len(train_dl), pct_start=0.2, div_factor=10)

    train_stats = {}
    test_stats = {}
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train_epoch(model, device, train_dl, optimizer, criterion, epoch, train_stats)
        scheduler.step()
        test_epoch(model, device, test_dl, criterion_test, test_stats)

    # plot train and test loss/acc
    plot_loss(train_stats, test_stats)
    plot_acc(train_stats, test_stats)

    # show mis-classified images
    misclassified = {}
    get_misclassified(model, device, test_dl, misclassified, num_samples=10)
    show_misclassified_imgs(misclassified, classes, nmax=10)

    # grad-cam on mis-classified images
    input_tensor = misclassified['data']
    original_imgs = get_denormalized_imgs(input_tensor)
    wrong_preds = misclassified['pred']
    correct_labels = misclassified['target']
    target_layers = [model.layer3[-1]]
    grayscale_cams, cam_image = get_gradcam_img(model, target_layers, input_tensor, original_imgs, wrong_preds)
    show_gradcam_plots(grayscale_cams, cam_image, original_imgs, classes, \
                       wrong_preds, correct_labels, resize=(1024, 1024), figsize=(20, 20))