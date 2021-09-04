import torch
import cv2
import random
import copy
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def calculate_accuracy(output, target):
    output = torch.sigmoid(output) >= 0.5
    target = target == 1.0
    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()


def get_transform():
    transforms = dict()
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    transforms['train'] = train_transform
    transforms['val'] = val_transform

    return transforms


def show_random(df, image_dir):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    for ax in axs.ravel():
        filename, label = df.sample().values[0]
        img = cv2.imread(f'{image_dir}/{filename}.tif', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(label)

    plt.show()


def visualize_augmentations(dataset, idx=None, samples=9, cols=3):
    # https://albumentations.ai/docs/examples/pytorch_classification/
    random.seed(42)
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 8))
    for i in range(samples):
        if idx is None:
            idx = random.randint(0, len(dataset))
        image, label = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(label)
    plt.tight_layout()
    plt.show()
