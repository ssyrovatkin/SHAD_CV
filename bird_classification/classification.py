import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import cv2
import os
from tqdm import tqdm


MU = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
N_CLASSES = 50


def train_epoch(model, train_dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    processed_data = 0

    y_labels = []
    scores = []
    metrics = {'f1_score': None, 'precision': None, 'recall': None, 'accuracy': None}

    for i, (img, labels) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, 1)
        y_labels.extend(labels.cpu().numpy())
        scores.extend(preds.cpu().detach().numpy())

        running_loss += loss.item() * img.size(0)
        processed_data += img.size(0)

    train_loss = running_loss / processed_data
    # metrics['precision'] = precision_score(scores, y_labels, average='macro')
    # metrics['recall'] = recall_score(scores, y_labels, average='macro')
    metrics['f1_score'] = f1_score(scores, y_labels, average='macro')
    metrics['accuracy'] = accuracy_score(scores, y_labels)

    return train_loss, metrics


def fast_train_epoch(model, train_dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    processed_data = 0
    y_labels = []
    scores = []
    metrics = {'f1_score': None, 'precision': None, 'recall': None, 'accuracy': None}

    for i, (img, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)
        processed_data += img.size(0)

        preds = torch.argmax(outputs, 1)
        y_labels.extend(labels.cpu().numpy())
        scores.extend(preds.cpu().detach().numpy())

        if i == 5:
            break

    train_loss = running_loss / processed_data
    # metrics['precision'] = precision_score(scores, y_labels, average='macro')
    # metrics['recall'] = recall_score(scores, y_labels, average='macro')
    metrics['f1_score'] = f1_score(scores, y_labels, average='macro')
    metrics['accuracy'] = accuracy_score(scores, y_labels)

    return train_loss, metrics


def validate(model, test_dataloader, criterion):
    model.eval()
    running_loss = 0.0
    processed_size = 0
    y_labels = []
    scores = []
    metrics = {'f1_score': None, 'precision': None, 'recall': None, 'accuracy': None}

    for i, (img, labels) in enumerate(tqdm(test_dataloader)):

        with torch.no_grad():
            outputs = model(img)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * img.size(0)
        processed_size += img.size(0)

        preds = torch.argmax(outputs, 1)
        y_labels.extend(labels.cpu().numpy())
        scores.extend(preds.cpu().detach().numpy())

    val_loss = running_loss / processed_size
    # metrics['precision'] = precision_score(scores, y_labels, average='macro')
    # metrics['recall'] = recall_score(scores, y_labels, average='macro')
    metrics['f1_score'] = f1_score(scores, y_labels, average='macro')
    metrics['accuracy'] = accuracy_score(scores, y_labels)

    return val_loss, metrics


def get_train_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(224, 224)),
        transforms.Normalize(mean=MU, std=STD)
    ])

    return transform


def get_test_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(224, 224)),
        transforms.Normalize(mean=MU, std=STD)
    ])

    return transform


class MyBeautifulModel(nn.Module):
    def __init__(self):
        super().__init__()
        md = resnet34(pretrained=False)

        self.md_freezed = nn.Sequential(
            md.conv1,
            md.bn1,
            md.relu,
            md.maxpool,
            md.layer1,
            md.layer2
        )

        self.md_trained = nn.Sequential(
            md.layer3,
            md.layer4
        )

        for p in self.md_freezed.parameters():
            p.requires_grad = False

        self.avg_pool = nn.AvgPool2d(5, stride=1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(nn.Dropout(p=0.3),
                          nn.Linear(4608, N_CLASSES, bias=True))


    def forward(self, x):
        out = self.md_freezed(x)
        out = self.md_trained(out)
        out = self.avg_pool(out)
        out = self.flatten(out)
        out = self.classifier(out)
        out = torch.softmax(out, dim=-1)
        return out


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
        label = torch.LongTensor([self.gt[img_name]])[0]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        H, W, _ = img.shape

        img = self.transform(img)

        return img, label


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
        label = torch.LongTensor([self.gt[img_name]])[0]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        H, W, _ = img.shape
        img = self.transform(img)

        return img, label


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
        img = self.transform(img)

        return img, img_name


def train_classifier(train_gt, train_img_dir, fast_train=False):
    # split data on train and test
    names = list(train_gt.keys())
    labels = list(train_gt.values())

    X_train, X_test, _, _ = train_test_split(names, labels, test_size = 0.1, \
                                                        random_state = 42, stratify=labels)

    train_pairs = {k: train_gt[k] for k in X_train}
    test_pairs = {k: train_gt[k] for k in X_test}

    # determ train and test transforms
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()
    # datasets
    train_dataset = TrainDataset(train_pairs, train_img_dir, transform=train_transform)
    test_dataset = ValDataset(test_pairs, train_img_dir, transform=test_transform)

    # train params
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    # dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # determ model and criterion
    model = MyBeautifulModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    best_acc = 0

    if fast_train:
        print(f"Fast Training one epoch 5 iterations:")
        train_loss, metrics = fast_train_epoch(model, train_dataloader, optimizer, criterion)
        scheduler.step()
        print("Training loss is:", train_loss)
        print("Accuracy score is:", metrics['accuracy'])

        print(f"Validation fast train epoch:")
        val_loss, metrics = validate(model, test_dataloader, criterion)
        print("Validation loss is:", val_loss)
        print("Accuracy score is:", metrics['accuracy'])

        # torch.save(model.state_dict(), 'birds_model.ckpt')
        return model

    for epoch in range(NUM_EPOCHS):

        print(f"Training {epoch + 1} epoch:")
        train_loss, metrics = train_epoch(model, train_dataloader, optimizer, criterion)
        scheduler.step()
        print("Training loss is:", train_loss)
        print("Accuracy score is:", metrics['accuracy'])

        print(f"Validation {epoch + 1} epoch:")
        val_loss, metrics = validate(model, test_dataloader, criterion)
        print("Validation loss is:", val_loss)
        print("Accuracy score is:", metrics['accuracy'])

        if metrics['accuracy'] < best_acc:
          best_acc = metrics['accuracy']
          torch.save(model.state_dict(), 'birds_model.ckpt')

    return model


def classify(model_filename, test_img_dir):
    model = MyBeautifulModel()
    model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')), strict=True)
    model.eval()

    output_scores = {}

    # test data
    test_transform = get_test_transforms()
    test_dataset = TestDataset(test_img_dir, test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0)

    for i, (img, img_name) in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            label = torch.argmax(model(img), 1)
            label = label.detach().numpy()
            output_scores[img_name[0]] = label[0]

    return output_scores