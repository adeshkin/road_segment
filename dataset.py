import cv2
import torch


class RoadSegmentFolds(torch.utils.data.Dataset):
    def __init__(self, X, y, image_dir, transform=None):
        self.img_dir = image_dir
        self.image_ids = X.tolist()
        self.labels = y.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = cv2.imread(f'{self.img_dir}/{image_id}.tif', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image=img)["image"]

        return image, label

class RoadSegment(torch.utils.data.Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.img_dir = image_dir
        self.image_ids = self.df['Image_ID'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = cv2.imread(f'{self.img_dir}/{image_id}.tif', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.df[self.df['Image_ID'] == image_id]['Target'].item()

        if self.transform:
            image = self.transform(image=img)["image"]

        return image, label


class RoadSegmentTest(torch.utils.data.Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.img_dir = image_dir
        self.transform = transform
        self.image_ids = self.df['Image_ID'].tolist()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = cv2.imread(f'{self.img_dir}/{image_id}.tif', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=img)["image"]

        return image, image_id
