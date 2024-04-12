from matplotlib import pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2 as cv2

def get_CIFAR10_musigma():
    return ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

def show_samples(img):
    img = img / 2 + 0.5    # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_loss(train_stats, test_stats):
    train_loss = train_stats['loss']
    test_loss = test_stats['loss']
    epochs = np.arange(0, len(train_loss))

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(epochs, train_loss, 'r', label='train_loss')
    ax[1].plot(epochs, test_loss, 'k', label='test_loss')
    ax[0].legend()
    ax[0].set_xlabel('epochs')
    ax[1].legend()
    ax[1].set_xlabel('epochs')


def plot_acc(train_stats, test_stats):
    train_acc = train_stats['acc']
    test_acc = test_stats['acc']
    epochs = np.arange(0, len(train_acc))

    plt.figure()
    plt.plot(epochs, train_acc, 'r', label='train_acc')
    plt.plot(epochs, test_acc, 'k', label='test_acc')
    plt.legend()
    plt.xlabel('epochs')
    plt.legend()

def get_misclassified(model, device, test_loader, misclassified, num_samples=10):
    model.eval()
    with torch.no_grad():
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(
                    dim=1, keepdim=True
                ).squeeze()
        idx = torch.where(pred != target)[0][0:num_samples]
        misclassified['data'] = data[idx].cpu()
        misclassified['pred'] = pred[idx].cpu()
        misclassified['target'] = target[idx].cpu()

def get_denormalized_imgs(input):
    '''
    input: normalized images, <B,C,H,W>
    '''
    mu, sigma = get_CIFAR10_musigma()
    mu = np.array(mu)
    sigma = np.array(sigma)

    # de-normalize images
    imgs = input
    npimgs = imgs.numpy()
    # de-normalize the normalized image
    npimgs = sigma[None, :, None, None] * npimgs
    npimgs = npimgs + mu[None, :, None, None]
    npimgs = np.clip(npimgs, 0, 1)
    imgs = np.transpose(npimgs, axes=(0, 2, 3, 1))
    return imgs

def get_gradcam_img(model, target_layer, input_tensor, imgs, preds):
    '''
    model: your trained model obj
    target_layer: layer in the model where you want to esimate Grad-CAM
    input_tensor: input to model (usually normalized image) <1,C,H,W>
    imgs: original de-normalized images <B,C,H,W>
    preds: can be preds or ground truth (label for which Grad CAM needs to be computed)
    '''
    targets = [ClassifierOutputTarget(pr) for pr in preds]
    target_layers = [model.layer3[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(imgs, grayscale_cams[0, :], use_rgb=True)

    return grayscale_cams, cam_image

def show_gradcam_plots(grayscale_cams, cam_image, original_imgs, classes, \
                       preds, labels, resize=(32,32), figsize=(20,20)):
    '''
    grayscale_cams: output from get_gradcam_img(), grayscale gradcam image <B,C,H,W>
    cam_image: output from get_gradcam_img(), grayscale gradcam overlaid on input image (de-normalized) <B,C,H,W>
    original_imgs: de-normalized <B,C,H,W>
    classes: for classifier
    preds: from classifier
    label: ground truth
    '''
    nrows = grayscale_cams.shape[0]
    fig, axes = plt.subplots(nrows, 1, figsize=figsize)
    for i in range(grayscale_cams.shape[0]):
        cam = np.uint8(255 * grayscale_cams[i, :, :])
        cam = cv2.merge([cam, cam, cam])  # grayscale
        cam_image_ = cam_image[i]  # overlaid (gradCAM grayscale + input image)
        img_ = np.uint8(255 * original_imgs[i])  # original image
        # rescale for visibility
        cam = cv2.resize(cam, resize, interpolation=cv2.INTER_CUBIC)
        cam_image_ = cv2.resize(cam_image_, resize, interpolation=cv2.INTER_CUBIC)
        image_ = cv2.resize(img_, resize, interpolation=cv2.INTER_CUBIC)
        single_img_ = np.hstack((image_, cam, cam_image_))
        # set the labels
        pred_label = classes[preds[i]]
        true_label = classes[labels[i]]
        axes[i].imshow(single_img_)
        axes[i].set_xticks([], [])
        axes[i].set_yticks([], [])
        axes[i].set_title(f"Pred:{pred_label}, True:{true_label}")

def show_misclassified_imgs(misclassified, classes, nmax=10, figsize=(20,20)):
    imgs =misclassified['data'][0:nmax]
    # de-normalize the images
    imgs = get_denormalized_imgs(imgs)
    wrong_preds = misclassified['pred']
    correct_labels = misclassified['target']
    nrows = nmax
    fig, axes = plt.subplots(nrows, 1, figsize=figsize)
    for i in range(nmax):
        pred_label = classes[wrong_preds[i]]
        true_label = classes[correct_labels[i]]
        axes[i].imshow(imgs[i])
        axes[i].set_xticks([], [])
        axes[i].set_yticks([], [])
        axes[i].set_title(f"Pred:{pred_label}, True:{true_label}")
