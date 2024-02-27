import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torchvision.models import resnet18, resnet34
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import cv2
import os
import copy
import random
from tqdm import tqdm
from PIL import Image


MU = [0.49404085, 0.4227429,  0.38548836]
STD = [0.28619801, 0.2488274,  0.23580867]


def train_epoch(model, train_dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    processed_data = 0

    for i, (img, points) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(points, outputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)
        processed_data += img.size(0)

    train_loss = running_loss / processed_data

    return train_loss


def fast_train_epoch(model, train_dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    processed_data = 0

    for i, (img, points) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(points, outputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)
        processed_data += img.size(0)

        if i == 5:
            break

    train_loss = running_loss / processed_data

    return train_loss


def validate(model, test_dataloader, criterion):
    model.eval()
    running_loss = 0.0
    processed_size = 0

    for i, (img, points) in enumerate(tqdm(test_dataloader)):

        with torch.no_grad():
            outputs = model(img)
            loss = criterion(outputs, points)

        running_loss += loss.item() * img.size(0)
        processed_size += img.size(0)

    val_loss = running_loss / processed_size

    return val_loss


def compute_mean_and_std(img_dir, names):
    imgs_path = [img_dir + '\\' + name for name in names][:10]

    rgb_values = np.concatenate(
        [Image.open(img).getdata() for img in imgs_path],
        axis=0
    ) / 255.

    mu_rgb = np.mean(rgb_values, axis=0)
    std_rgb = np.std(rgb_values, axis=0)

    return mu_rgb, std_rgb


def get_train_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(100, 100)),
        transforms.Normalize(mean=MU, std=STD)
    ])

    return transform


def get_test_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(100, 100)),
        transforms.Normalize(mean=MU, std=STD)
    ])

    return transform


class MyBeautifulModel(nn.Module):

    def __init__(self):
        super(MyBeautifulModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 4)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, 1)

        self.lin1 = nn.Linear(25600, 1000)
        self.lin2 = nn.Linear(1000, 500)
        self.lin3 = nn.Linear(500, 28)

    def forward(self, x):
        drop1 = nn.Dropout(0.1)
        drop2 = nn.Dropout(0.2)
        drop3 = nn.Dropout(0.3)
        drop4 = nn.Dropout(0.4)

        x = drop1(self.pool1(F.relu(self.conv1(x))))
        x = drop2(self.pool2(F.relu(self.conv2(x))))
        x = drop3(self.pool3(F.relu(self.conv3(x))))
        x = drop4(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = self.lin3(x)

        return x


class MyBeautifulModelv2(nn.Module):
    def __init__(self):
        super().__init__()
        md = resnet18(pretrained=True)
        # все, кроме avgpool и fc
        self.feature_extractor = nn.Sequential(
            md.conv1,
            md.bn1,
            md.relu,
            md.maxpool,
            md.layer1,
            md.layer2,
            md.layer3,
            md.layer4
        )
        self.regr = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.regr(x)

        return x


class TrainDataset(Dataset):
    def __init__(self, gt, img_dir, transform=None):
        super(Dataset).__init__()
        self.gt = gt
        self.img_names = list(gt.keys())
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        H, W, _ = img.shape
        true_points = copy.deepcopy(self.gt[img_name])

        if np.random.uniform(size=1)[0] < 0.2:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.merge((gray_img,gray_img,gray_img))

        img = self.transform(img)

        true_points[1::2] = true_points[1::2] / W * 100
        true_points[::2] = true_points[::2] / H * 100
        true_points = torch.FloatTensor(np.round(true_points))

        return img, true_points


class ValDataset(Dataset):
    def __init__(self, gt, img_dir, transform=None):
        super(Dataset).__init__()
        self.gt = gt
        self.img_names = list(gt.keys())
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        H, W, _ = img.shape
        img = self.transform(img)

        true_points = copy.deepcopy(self.gt[img_name])
        true_points[1::2] = true_points[1::2] / W * 100
        true_points[::2] = true_points[::2] / H * 100
        true_points = torch.FloatTensor(np.round(true_points))

        return img, true_points


class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        super(Dataset).__init__()
        self.img_names = os.listdir(img_dir)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        H, W, _ = img.shape
        img = self.transform(img)

        return img, img_name, H, W


def train_detector(train_gt, train_img_dir, fast_train=False):
    # split data on train and test
    shuffle_train_gt = sorted(copy.copy(list(train_gt.keys())), key=lambda x: random.random())
    train_pairs = {k: train_gt[k] for k in shuffle_train_gt[:-500]}
    test_pairs = {k: train_gt[k] for k in shuffle_train_gt[-500:]}
    # compute mean and std using first 10 images of train data
    train_names = list(train_pairs.keys())
    global MU
    global STD
    if MU is None and STD is None:
        MU, STD = compute_mean_and_std(train_img_dir, train_names)
    # determ train and test transforms
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()
    # datasets
    train_dataset = TrainDataset(train_pairs, train_img_dir, transform=train_transform)
    test_dataset = ValDataset(test_pairs, train_img_dir, transform=test_transform)
    # train params
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    # dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # determ model and criterion
    model = MyBeautifulModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    best_val_loss = float('inf')

    if fast_train:
        print(f"Fast Training one epoch 5 iterations:")
        train_loss = fast_train_epoch(model, train_dataloader, optimizer, criterion)
        scheduler.step()
        print("Training loss is:", train_loss)

        print(f"Validation fast train epoch:")
        val_loss = validate(model, test_dataloader, criterion)
        print("Validation loss is:", val_loss)

        # torch.save(model.state_dict(), 'facepoints_model.ckpt')
        return model

    for epoch in range(NUM_EPOCHS):

        print(f"Training {epoch + 1} epoch:")
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion)
        scheduler.step()
        print("Training loss is:", train_loss)

        print(f"Validation {epoch + 1} epoch:")
        val_loss = validate(model, test_dataloader, criterion)
        print("Validation loss is:", val_loss)

        if val_loss < best_val_loss:
          best_val_loss = val_loss
          torch.save(model.state_dict(), 'facepoints_model.ckpt')

    return model


def detect(model_filename, test_img_dir):
    model = MyBeautifulModel()
    model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')), strict=True)
    model.eval()

    output_scores = {}

    # test data
    test_transform = get_test_transforms()
    test_dataset = TestDataset(test_img_dir, test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0)

    for i, (img, img_name, H, W) in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            points = model(img)[0]
            points = points.detach().numpy()
            points[::2] = points[::2] / 100 * H.item()
            points[1::2] = points[1::2] / 100 * W.item()
            output_scores[img_name[0]] = points

    return output_scores

