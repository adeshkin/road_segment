import torch
import cv2
import random
import pandas as pd
import copy
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn import metrics


def calculate_auc_score(output, target):
    fpr, tpr, thresholds = metrics.roc_curve(target, output, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


def calculate_accuracy(output, target):
    output = torch.sigmoid(output) >= 0.5
    target = target == 1.0
    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()


def get_transform():
    transforms = dict()
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=90, p=0.5),
        A.CoarseDropout(min_height=32, max_height=48, min_width=32, max_width=48, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
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


def visualize_augmentations(dataset, sample=None, samples=9, cols=3):
    # https://albumentations.ai/docs/examples/pytorch_classification/
    random.seed(42)
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 8))
    for i in range(samples):
        if sample is None:
            idx = random.randint(0, len(dataset))
        else:
            idx = sample
        image, label = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(label)
    plt.tight_layout()
    plt.show()


def read_data(params):
    df = pd.read_csv(params['train_filepath'])
    test_df = pd.read_csv(params['test_filepath'])

    image_ids = df['Image_ID'].tolist()
    labels = df['Target'].tolist()

    train_ids, valid_ids = train_test_split(image_ids, test_size=params['test_size'],
                                            random_state=42, stratify=labels)

    train_df = df[df['Image_ID'].isin(train_ids)]
    val_df = df[df['Image_ID'].isin(valid_ids)]

    return train_df, val_df, test_df
