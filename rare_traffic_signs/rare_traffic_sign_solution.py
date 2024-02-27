# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import cv2
import csv
import json
import tqdm
import math
import pickle
import typing
import random

import numpy as np
from sklearn.metrics import accuracy_score
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KNeighborsClassifier

import torch.nn.functional as F


CLASSES_CNT = 205


class DatasetRTSD(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения датасета.
    :param root_folders: список путей до папок с данными
    :param path_to_classes_json: путь до classes.json
    """
    def __init__(self, root_folders, path_to_classes_json) -> None:
        super(DatasetRTSD, self).__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
        self.samples = [] ### YOUR CODE HERE - список пар (путь до картинки, индекс класса)
        self.classes_to_samples = {self.class_to_idx[k]: [] for k in self.classes}  ### YOUR CODE HERE - cловарь из списков картинок для каждого класса, classes_to_samples[индекс класса] = [список чисел-позиций картинок в self.samples]
        for root in root_folders:
            for c in os.listdir(root):
                class_dir = os.path.join(root, c)
                paths = os.listdir(class_dir)
                for p in paths:
                    self.samples.append((root + '/' + c + '/' + p, self.class_to_idx[c]))
                    class_idx = self.class_to_idx[c]
                    self.classes_to_samples[class_idx].append(len(self.samples) - 1)

        self.transform = A.Compose([
                                    A.Resize(224, 224),
                                    A.Normalize(),
                                    ToTensorV2()
                                ]) ### YOUR CODE HERE - аугментации + нормализация + ToTensorV2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        ### YOUR CODE HERE
        img_path, class_idx = self.samples[index]
        img = cv2.imread(img_path)
        img = self.transform(image=img)['image']

        return img, img_path, class_idx

    @staticmethod
    def get_classes(path_to_classes_json):
        """
        Считывает из classes.json информацию о классах.
        :param path_to_classes_json: путь до classes.json
        """
        file = open(path_to_classes_json)
        data = json.load(file)
        class_to_idx = {k: data[k]['id'] for k in data.keys()}  ### YOUR CODE HERE - словарь, class_to_idx['название класса'] = индекс
        classes = list(data.keys())                             ### YOUR CODE HERE - массив, classes[индекс] = 'название класса'
        file.close()
        return classes, class_to_idx


class TestData(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения тестового датасета.
    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    def __init__(self, root, path_to_classes_json, annotations_file=None):
        super(TestData, self).__init__()
        self.root = root
        self.samples = os.listdir(root) ### YOUR CODE HERE - список путей до картинок
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)

        self.transform = A.Compose([
                                    A.Resize(224, 224),
                                    A.Normalize(),
                                    ToTensorV2()
                                ]) ### YOUR CODE HERE - преобразования: ресайз + нормализация + ToTensorV2
        self.targets = None
        if annotations_file is not None:
            self.targets = {}  ### YOUR CODE HERE - словарь, targets[путь до картинки] = индекс класса
            with open(annotations_file, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                fields = next(csvreader)
                for item in csvreader:
                    name, c = item
                    self.targets[name] = self.class_to_idx[c]


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        ### YOUR CODE HERE
        img_path = self.samples[index]
        img = cv2.imread(os.path.join(self.root, img_path))
        img = self.transform(image=img)['image']
        if self.targets is not None:
            target = self.targets[img_path]
        else:
            target = -1

        return img, img_path, target


    @staticmethod
    def get_classes(path_to_classes_json):
        file = open(path_to_classes_json)
        data = json.load(file)
        class_to_idx = {k: data[k]['id'] for k in data.keys()}
        classes = list(data.keys())
        file.close()
        return classes, class_to_idx


class CustomNetwork(pl.LightningModule):
    """
    Класс, реализующий нейросеть для классификации.
    :param features_criterion: loss-функция на признаки, извлекаемые нейросетью перед классификацией (None когда нет такого лосса)
    :param internal_features: внутреннее число признаков
    """
    def __init__(self, features_criterion = None, internal_features = 1024):
        super(CustomNetwork, self).__init__()
        ### YOUR CODE HERE
        feature_extractor = torchvision.models.resnet50(pretrained=False)
        self.feature_extractor =  nn.Sequential(
            feature_extractor.conv1,
            feature_extractor.bn1,
            feature_extractor.relu,
            feature_extractor.maxpool,
            feature_extractor.layer1,
            feature_extractor.layer2,
            feature_extractor.layer3,
            feature_extractor.layer4,
            feature_extractor.avgpool
        )

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(2048, internal_features)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(internal_features, CLASSES_CNT)
        self.features_criterion = features_criterion


    def predict(self, x):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param x: батч с картинками
        """
        out = self.feature_extractor(x)
        out = self.flatten(out)
        features = self.dense(out)
        out = self.classifier(self.relu(features))
        return features, out ### YOUR CODE HERE

    def forward(self, x, labels):
        out = self.feature_extractor(x)
        out = self.flatten(out)
        emb = self.dense(out)
        if self.features_criterion is not None:
            loss = self.features_criterion(emb, labels)
        else:
            loss = None
        out = self.classifier(self.relu(emb))
        return out, loss


def train_simple_classifier():
    """Функция для обучения простого классификатора на исходных данных."""
    ### YOUR CODE HERE

    root_folders = ['./cropped-train/']
    path_to_classes_json = './classes.json'
    batch_size = 128
    num_epochs = 5

    train_dataset = DatasetRTSD(root_folders, path_to_classes_json)
    print("Number of train samples:", len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = CustomNetwork(features_criterion=None)
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of model trainable parameters:", trainable_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(num_epochs):
        print(f"Training epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        processed_data = 0
        y_labels = []
        scores = []
        model.train()
        for k, (img, img_path, class_idx) in enumerate(tqdm.tqdm(train_dataloader)):
            optimizer.zero_grad()
            _, outs = model.predict(img)
            loss = criterion(outs, class_idx)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outs, 1)
            y_labels.extend(class_idx.cpu().numpy())
            scores.extend(preds.cpu().detach().numpy())
            running_loss += loss.item() * img.size(0)
            processed_data += img.size(0)

        train_loss = running_loss / processed_data
        print("Train loss:", train_loss)
        print("Train accuracy:", accuracy_score(scores, y_labels))
        torch.save(model.state_dict(), f'simple_model_{epoch + 1}.pth')

        model.eval()
        total_acc, rare_recall, freq_recall = test_classifier(model, './smalltest', './classes.json', './smalltest_annotations.csv')
        print("freq_recall:", freq_recall)

    torch.save(model.state_dict(), 'simple_model.pth')

    return model


def apply_classifier(model, test_folder, path_to_classes_json):
    """
    Функция, которая применяет модель и получает её предсказания.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param path_to_classes_json: путь до файла с информацией о классах classes.json
    """
    file = open(path_to_classes_json)
    data = json.load(file)
    idx_to_class = {data[k]['id']: k for k in data.keys()}
    file.close()

    test_dataset = TestData(test_folder, path_to_classes_json)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    results = []  ### YOUR CODE HERE - список словарей вида {'filename': 'имя файла', 'class': 'строка-название класса'}

    for k, (img, img_path, _) in enumerate(tqdm.tqdm(test_dataloader)):
        with torch.no_grad():
            _, out = model.predict(img)

            if torch.is_tensor(out):
                class_idx = torch.argmax(out, dim=-1).detach().numpy()[0]
            else:
                class_idx = out[0]
            class_name = idx_to_class[class_idx]
            results.append({'filename': img_path[0], 'class': class_name})

    return results


def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        reader = csv.DictReader(fhandle)
        for row in reader:
            res[row['filename']] = row['class']
    return res


def calc_metric(y_true, y_pred, cur_type, class_name_to_type):
    ok_cnt = 0
    all_cnt = 0
    for t, p in zip(y_true, y_pred):
        if cur_type == 'all' or class_name_to_type[t] == cur_type:
            all_cnt += 1
            if t == p:
                ok_cnt += 1
    return ok_cnt / max(1, all_cnt)


def test_classifier(model, test_folder, path_to_classes_json, annotations_file):
    """
    Функция для тестирования качества модели.
    Возвращает точность на всех знаках, Recall на редких знаках и Recall на частых знаках.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    ### YOUR CODE HERE
    results = apply_classifier(model, test_folder, path_to_classes_json)
    results = {elem['filename']: elem['class'] for elem in results}
    gt = read_csv(annotations_file)
    y_pred = []
    y_true = []
    for k, v in results.items():
        y_pred.append(v)
        y_true.append(gt[k])

    with open(path_to_classes_json, "r") as fr:
        classes_info = json.load(fr)
    class_name_to_type = {k: v['type'] for k, v in classes_info.items()}

    total_acc = calc_metric(y_true, y_pred, 'all', class_name_to_type)
    rare_recall = calc_metric(y_true, y_pred, 'rare', class_name_to_type)
    freq_recall = calc_metric(y_true, y_pred, 'freq', class_name_to_type)

    return total_acc, rare_recall, freq_recall


class SignGenerator(object):
    """
    Класс для генерации синтетических данных.
    :param background_path: путь до папки с изображениями фона
    """
    def __init__(self, background_path):
        ### YOUR CODE HERE
        self.background_path = background_path

    def get_sample(self, icon_and_mask):
        """
        Функция, встраивающая иконку на случайное изображение фона.
        :param icon: Массив с изображением иконки
        """
        # random resize
        mask, icon = icon_and_mask[:, :, -1], icon_and_mask[:, :, :-1]
        size = np.random.randint(16, 128)
        icon = cv2.resize(icon, (size, size))
        mask = cv2.resize(mask, (size, size))

        # icon, mask = self.RPTransform(icon, mask, w0=size, w1=size - 10)

        # random padding
        pad_size = int(np.random.uniform(0, 0.15) * size)
        icon = cv2.copyMakeBorder(icon, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)
        # random color
        hcv_icon = cv2.cvtColor(icon, cv2.COLOR_RGB2HSV).astype('int16')
        random_hue = np.random.randint(-20, 20)
        random_color = np.random.randint(-25, 25)
        random_value = np.random.randint(-25, 25)
        hcv_icon[:,:,0] += random_hue
        hcv_icon[:,:,1] += random_color
        hcv_icon[:,:,2] += random_value
        hcv_icon = np.clip(hcv_icon, 0, 255).astype('uint8')
        icon = cv2.cvtColor(hcv_icon, cv2.COLOR_HSV2RGB)
        # random rotation
        center = (size / 2, size / 2)
        random_angle = np.random.randint(-15, 15)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=random_angle, scale=1)
        icon = cv2.warpAffine(src=icon, M=rotate_matrix, dsize=(size, size))
        mask = cv2.warpAffine(src=mask, M=rotate_matrix, dsize=(size, size), borderValue=0)

        # motion blur
        kernel_size = 10
        kernel = np.zeros((kernel_size, kernel_size))
        if np.random.uniform(0, 1) > 0.5:
            kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        else:
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel /= kernel_size
        icon = cv2.filter2D(icon, -1, kernel)

        # gaussian blur
        icon = cv2.GaussianBlur(icon, (7, 7), 0)

        # background random crop
        bg = np.random.choice(os.listdir(self.background_path)) ### YOUR CODE HERE - случайное изображение фона
        bg_img = cv2.imread(os.path.join(self.background_path, bg))
        bg_img = self.get_random_crop(bg_img, size, size)

        # inpainting
        icon = icon * (mask[:, :, None] / 255) + bg_img * (1 - (mask[:, :, None] / 255))

        return icon ### YOUR CODE HERE

    def get_random_crop(self, image, crop_height, crop_width):
        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        crop = image[y: y + crop_height, x: x + crop_width]
        return crop

    def TwoPointRandInterP(self, p1, p2):
        v = (p2[0] - p1[0], p2[1] - p1[1])
        rr = random.random()
        v = (v[0] * rr, v[1] * rr)
        v = (int(v[0] + 0.5), int(v[1] + 0.5))
        return (p1[0] + v[0], p1[1] + v[1])

    def RPTransform(self, img, mask, w0=192, w1=128):

        d = (w0 - w1) // 2
        A = (0, 0)
        B = (0, w0)
        C = (w0, w0)
        D = (w0, 0)
        a = (d, d)
        b = (d, w0 - d)
        c = (w0 - d, w0 - d)
        d = (w0 - d, d)

        At = self.TwoPointRandInterP(A, a)
        Bt = self.TwoPointRandInterP(B, b)
        Ct = self.TwoPointRandInterP(C, c)
        Dt = self.TwoPointRandInterP(D, d)
        Pts0 = np.array([At, Bt, Ct, Dt]).astype('float32')
        Pts1 = np.array([(0, 0), (0, w1), (w1, w1), (w1, 0)]).astype('float32')
        Tm = cv2.getPerspectiveTransform(Pts0, Pts1)
        res = cv2.warpPerspective(img, Tm, dsize=(w1, w1))
        mask = cv2.warpPerspective(mask, Tm, dsize=(w1, w1))

        return res, mask


def generate_one_icon(args):
    """
    Функция, генерирующая синтетические данные для одного класса.
    :param args: Это список параметров: [путь до файла с иконкой, путь до выходной папки, путь до папки с фонами, число примеров каждого класса]
    """
    ### YOUR CODE HERE
    icon_path, output_folder, background_path, samples_per_class = args[0], args[1], args[2], args[3]
    sign_generator = SignGenerator(background_path)
    class_name = icon_path.split(sep='/')[-1]
    class_name = class_name[:-4]
    output_path = os.path.join(output_folder, class_name)
    # read icon img
    icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
    os.makedirs(output_path, exist_ok=True)

    for i in range(samples_per_class):
        synt_img = sign_generator.get_sample(icon)
        synt_img_name = str(i) + '.png'
        cv2.imwrite(os.path.join(output_path, synt_img_name), synt_img)


def generate_all_data(output_folder, icons_path, background_path, samples_per_class = 1000):
    """
    Функция, генерирующая синтетические данные.
    Эта функция запускает пул параллельно работающих процессов, каждый из которых будет генерировать иконку своего типа.
    Это необходимо, так как процесс генерации очень долгий.
    Каждый процесс работает в функции generate_one_icon.
    :param output_folder: Путь до выходной директории
    :param icons_path: Путь до директории с иконками
    :param background_path: Путь до директории с картинками фона
    :param samples_per_class: Количество примеров каждого класса, которые надо сгенерировать
    """
    with ProcessPoolExecutor(8) as executor:
        params = [[os.path.join(icons_path, icon_file), output_folder, background_path, samples_per_class]
                  for icon_file in os.listdir(icons_path)]
        list(tqdm.tqdm(executor.map(generate_one_icon, params)))


def train_synt_classifier():
    """Функция для обучения простого классификатора на смеси исходных и ситетических данных."""
    ### YOUR CODE HERE
    root_folders = ['./synt-cropped-train/']
    path_to_classes_json = './classes.json'
    batch_size = 128
    num_epochs = 5

    train_dataset = DatasetRTSD(root_folders, path_to_classes_json)
    print("Number of train samples:", len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = CustomNetwork(features_criterion=None)
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of model trainable parameters:", trainable_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)

    for epoch in range(num_epochs):
        print(f"Training epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        processed_data = 0
        y_labels = []
        scores = []
        model.train()
        for k, (img, img_path, class_idx) in enumerate(tqdm.tqdm(train_dataloader)):
            optimizer.zero_grad()
            _, outs = model.predict(img)
            loss = criterion(outs, class_idx)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outs, 1)
            y_labels.extend(class_idx.cpu().numpy())
            scores.extend(preds.cpu().detach().numpy())
            running_loss += loss.item() * img.size(0)
            processed_data += img.size(0)

        train_loss = running_loss / processed_data
        print("Train loss:", train_loss)
        print("Train accuracy:", accuracy_score(scores, y_labels))
        torch.save(model.state_dict(), f'simple_model_with_synt_{epoch + 1}.pth')

        model.eval()
        total_acc, rare_recall, freq_recall = test_classifier(model, './smalltest', './classes.json', './smalltest_annotations.csv')
        print("freq_recall:", freq_recall)
        print("rare_recall:", rare_recall)
        print("total_acc:", total_acc)

    torch.save(model.state_dict(), 'simple_model_with_synt.pth')
    return model


class FeaturesLoss(torch.nn.Module):
    """
    Класс для вычисления loss-функции на признаки предпоследнего слоя нейросети.
    """
    def __init__(self, margin: float) -> None:
        super(FeaturesLoss, self).__init__()
        ### YOUR CODE HERE
        self.margin = margin

    def forward(self, emb, label):
        label = label.float()
        label_batch = ((label[:, None] + 1) @ (1 / (1 + label[None, :])))
        mask = (label_batch == 1.0).float()
        norm_emb = F.normalize(emb, p=2, dim=1)
        euclidean_distance = F.pairwise_distance(norm_emb[:, None], norm_emb[None, :])
        loss_contrastive = torch.sum((mask) * torch.pow(euclidean_distance, 2) / (2 * torch.sum(mask) + 1e-7) + \
                                      (1 - mask) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2) / (2 * torch.sum(1 - mask) + 1e-7))

        return loss_contrastive


class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Класс для семплирования батчей с контролируемым числом классов и примеров каждого класса.
    :param data_source: Это датасет RTSD
    :param elems_per_class: Число элементов каждого класса
    :param classes_per_batch: Количество различных классов в одном батче
    """
    def __init__(self, data_source, elems_per_class, classes_per_batch):
        ### YOUR CODE HERE
        self.data_source = data_source
        self.elems_per_class = elems_per_class
        self.classes_per_batch = classes_per_batch

        self.batch_count = math.ceil(len(self.data_source) / (self.elems_per_class * self.classes_per_batch))

    def __iter__(self):
        ### YOUR CODE HERE

        samples = []
        for batch in range(self.batch_count):
            sample = []
            classes = np.random.choice(self.data_source.classes, size=self.classes_per_batch, replace=False)
            for c in classes:
                class_idx = self.data_source.class_to_idx[c]
                sample += random.choices(self.data_source.classes_to_samples[class_idx], k=self.elems_per_class)
            samples.append(sample)

        return iter(samples)


def train_better_model():
    """Функция для обучения классификатора на смеси исходных и ситетических данных с новым лоссом на признаки."""
    ### YOUR CODE HERE

    root_folders = ['./synt-cropped-train/']
    path_to_classes_json = './classes.json'

    elems_per_class = 4
    classes_per_batch = 32
    batch_size = elems_per_class * classes_per_batch
    num_epochs = 5

    train_dataset = DatasetRTSD(root_folders, path_to_classes_json)
    print("Number of train samples:", len(train_dataset))
    sampler = CustomBatchSampler(train_dataset, elems_per_class, classes_per_batch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    features_criterion = FeaturesLoss(margin=2)

    model = CustomNetwork(features_criterion=features_criterion)
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of model trainable parameters:", trainable_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)

    for epoch in range(num_epochs):
        print(f"Training epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        processed_data = 0
        y_labels = []
        scores = []
        model.train()
        for k, (img, img_path, class_idx) in enumerate(tqdm.tqdm(train_dataloader)):
            optimizer.zero_grad()
            outs, feature_loss_value = model(img, class_idx)
            cross_entropy_loss = criterion(outs, class_idx)
            loss = feature_loss_value + cross_entropy_loss

            loss.backward()
            optimizer.step()

            preds = torch.argmax(outs, 1)
            y_labels.extend(class_idx.cpu().numpy())
            scores.extend(preds.cpu().detach().numpy())
            running_loss += loss.item() * img.size(0)
            processed_data += img.size(0)

        train_loss = running_loss / processed_data
        print("Train loss:", train_loss)
        print("Train accuracy:", accuracy_score(scores, y_labels))
        torch.save(model.state_dict(), f'improved_features_model_{epoch + 1}.pth')

        model.eval()
        total_acc, rare_recall, freq_recall = test_classifier(model, './smalltest', './classes.json', './smalltest_annotations.csv')
        print("freq_recall:", freq_recall)
        print("rare_recall:", rare_recall)
        print("total_acc:", total_acc)

    torch.save(model.state_dict(), 'improved_features_model.pth')
    return model


class ModelWithHead:
    """
    Класс, реализующий модель с головой из kNN.
    :param n_neighbors: Количество соседей в методе ближайших соседей
    """
    def __init__(self, n_neighbors):
        ### YOUR CODE HERE
        self.n_neighbors = n_neighbors

    def load_nn(self, nn_weights_path):
        """
        Функция, загружающая веса обученной нейросети.
        :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
        """
        ### YOUR CODE HERE
        self.model = CustomNetwork(features_criterion=None)
        self.model.load_state_dict(torch.load(nn_weights_path, map_location='cpu'))
        self.model.eval()

    def load_head(self, knn_path):
        """
        Функция, загружающая веса kNN (с помощью pickle).
        :param knn_path: Путь, откуда надо прочитать веса kNN
        """
        ### YOUR CODE HERE
        self.head = KNeighborsClassifier(n_neighbors = self.n_neighbors)
        with open(knn_path, "rb") as model_filename:
            self.head = pickle.load(model_filename)

    def predict(self, imgs):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param imgs: батч с картинками
        """
        features, model_pred = self.model.predict(imgs) ### YOUR CODE HERE - предсказание нейросетевой модели
        model_pred = torch.argmax(model_pred, 1).cpu().detach().numpy()
        features = features.detach().numpy()
        features = features / np.linalg.norm(features, axis=1)[:, None]
        knn_pred = self.head.predict(features) ### YOUR CODE HERE - предсказание kNN на features
        return model_pred, knn_pred


class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.
    :param data_source: Это датасет RTSD с синтетическими примерами
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    def __init__(self, data_source, examples_per_class) -> None:
        ### YOUR CODE HERE
        self.data_source = data_source
        self.examples_per_class = examples_per_class
    def __iter__(self):
        """Функция, которая будет генерировать список индексов элементов в батче."""
        classes_idx = [self.data_source.class_to_idx[c] for c in self.data_source.classes]
        samples = []
        for idx in classes_idx:
            samples += random.choices(self.data_source.classes_to_samples[idx], k=self.examples_per_class)
        return iter(samples) ### YOUR CODE HERE


def train_head(nn_weights_path, examples_per_class=20):
    """
    Функция для обучения kNN-головы классификатора.
    :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    ### YOUR CODE HERE
    root_folders = ['./synt-cropped-train']
    path_to_classes_json = './classes.json'

    train_dataset = DatasetRTSD(root_folders, path_to_classes_json)
    batch_size = 32 # examples_per_class * len(train_dataset.classes)
    print("Number of train samples:", examples_per_class * len(train_dataset.classes))
    sampler = IndexSampler(train_dataset, examples_per_class)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)


    model = CustomNetwork(features_criterion=None)
    model.load_state_dict(torch.load(nn_weights_path))
    model.eval()

    knn_model = KNeighborsClassifier(n_neighbors=20)

    all_class_idx = []
    all_features = np.zeros((1, 1024))

    print(f"Training epoch kNN head")

    for k, (img, img_path, class_idx) in enumerate(tqdm.tqdm(train_dataloader)):

        features, _ = model.predict(img)
        features = features.cpu().detach().numpy()
        class_idx = class_idx.cpu().detach().numpy()
        all_class_idx.extend(class_idx)
        all_features = np.concatenate((all_features, features), axis=0)

    all_features = all_features[1:, :]
    knn_model.fit(all_features, all_class_idx)

    with open("knn_model.bin", "wb") as f:
        pickle.dump(knn_model, f)

    return knn_model
