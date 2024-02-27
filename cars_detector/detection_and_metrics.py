# ============================== 1 Classifier model ============================
import numpy as np
import torch.optim
import torchvision.transforms

from tqdm import tqdm


class CarsDataset(torch.utils.data.Dataset):
    def __init__(self, X, y) -> None:
        super(CarsDataset, self).__init__()
        self.X = X
        self.y = y

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
        ])

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        ### YOUR CODE HERE
        img, label = self.X[index], self.y[index]
        img = self.transforms(img)

        return img, label


def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """
    # your code here \/
    from torch.nn import Sequential
    return Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.AvgPool2d(kernel_size=2),

        torch.nn.Conv2d(64, 48, kernel_size=3, padding=1, stride=1),
        torch.nn.BatchNorm2d(48),
        torch.nn.ReLU(),
        torch.nn.AvgPool2d(kernel_size=2),

        torch.nn.Conv2d(48, 32, kernel_size=3, padding=1, stride=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.AvgPool2d(kernel_size=2),

        torch.nn.Flatten(),
        torch.nn.Linear(1920, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 2)
    )
    # your code here /\


def fit_cls_model(X, y):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    # your code here \/
    train_dataset = CarsDataset(X, y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    model = get_cls_model((40, 100, 1))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    NUM_EPOCHS = 10
    criterion = torch.nn.CrossEntropyLoss()
    # train model
    for i in range(NUM_EPOCHS):
        print(f"Training {i + 1}/{NUM_EPOCHS} epoch")
        for k, (img, label) in enumerate(tqdm(train_dataloader)):
            opt.zero_grad()
            y_pred = model(img)
            loss = criterion(y_pred, label)
            loss.backward()
            opt.step()

    model.eval()
    # torch.save(model.state_dict(), 'classifier_model.pth')

    return model
    # your code here /\


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    from torch.nn import Sequential

    detection_model = Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.AvgPool2d(kernel_size=2),

        torch.nn.Conv2d(64, 48, kernel_size=3, padding=1, stride=1),
        torch.nn.BatchNorm2d(48),
        torch.nn.ReLU(),
        torch.nn.AvgPool2d(kernel_size=2),

        torch.nn.Conv2d(48, 32, kernel_size=3, padding=1, stride=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.AvgPool2d(kernel_size=2),

        torch.nn.Flatten(),
        torch.nn.Linear(1920, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 2)
    )

    detection_model.load_state_dict(cls_model)
    detection_model[-6] = torch.nn.Conv2d(32, 100, kernel_size=(5, 12), padding=0)

    detection_model[-6].weight = torch.nn.parameter.Parameter(torch.reshape(detection_model[-5].weight, (100, 32, 5, 12)))
    detection_model[-6].bias = detection_model[-5].bias

    del detection_model[-5]

    second_last_linear = copy.deepcopy(detection_model[-3])

    detection_model[-3] = torch.nn.Conv2d(100, 20, kernel_size=(1, 1), padding=0)
    detection_model[-3].weight = torch.nn.parameter.Parameter(
        torch.reshape(second_last_linear.weight, (20, 100, 1, 1)))
    detection_model[-3].bias = second_last_linear.bias

    last_linear = copy.deepcopy(detection_model[-1])

    detection_model[-1] = torch.nn.Conv2d(20, 2, kernel_size=(1, 1), padding=0)
    detection_model[-1].weight = torch.nn.parameter.Parameter(
        torch.reshape(last_linear.weight, (2, 20, 1, 1)))
    detection_model[-1].bias = last_linear.bias

    return detection_model
    # your code here /\


# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    # your code here \/
    detection_model.eval()

    res = {k: [] for k in dictionary_of_images.keys()}
    padded_imgs = []
    for img in dictionary_of_images.values():
        H, W = img.shape
        pad_img = np.pad(img, ((0, 220 - H), (0, 370 - W)))
        pad_img = pad_img[None, :, :]
        padded_imgs.append(pad_img)
    test_batch = torch.FloatTensor(np.stack(padded_imgs, axis=0))

    for i, (img_name, img) in enumerate(tqdm(dictionary_of_images.items())):
        with torch.no_grad():
            H_in, W_in = 220, 370

            # tens = torch.FloatTensor(img[None, None, :, :])

            heat_map = detection_model(test_batch[i].unsqueeze(0)) # tens test_batch[i].unsqueeze(0)
            _, _, H_out, W_out = heat_map.shape

            mask = (heat_map > -5).float()
            indexes = (mask == 1).nonzero(as_tuple=False).numpy()
            heat_map = heat_map.numpy()

            for idx in indexes:
                if int(H_in * idx[2] / H_out) < H_in - 40 and int(W_in * idx[3] / W_out) < W_in - 100 and idx[1] == 1:
                    res[img_name].append([int(idx[2]*H_in/H_out), int(idx[3]*W_in/W_out), 40, 100,
                                          heat_map[idx[0], 1, idx[2], idx[3]]])
    print(res)
    return res
    # your code here /\


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    row1, col1, n_rows1, n_cols1 = first_bbox
    row2, col2, n_rows2, n_cols2 = second_bbox

    x_right = min(col1 + n_cols1, col2 + n_cols2)
    y_bottom = min(row1 + n_rows1, row2 + n_rows2)
    x_left = max(col1, col2)
    y_top = max(row1, row2)

    if x_right - x_left < 0 or y_bottom - y_top < 0:
        return 0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = n_rows1 * n_cols1 + n_rows2 * n_cols2 - intersection + 1e-8

    return intersection / union
    # your code here /\


# =============================== 6 AUC ========================================
from matplotlib import pyplot as plt

def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/
    iou_thr = 0.5
    filenames = list(pred_bboxes.keys())
    tp = []
    fp = []
    all_rects = 0

    for file in filenames:
        pred = pred_bboxes[file]
        sorted_preds = sorted(pred, key=lambda pred: pred[-1], reverse=True)
        all_rects += len(gt_bboxes[file])
        for p_bbox in sorted_preds:
            curr_iou = []
            max_iou = 0
            for gt_bbox in gt_bboxes[file]:
                curr_iou.append(calc_iou(p_bbox[:-1], gt_bbox))
            if len(curr_iou) != 0:
                max_iou = max(curr_iou)
            if max_iou >= iou_thr:
                tp.append(p_bbox[-1])
                idx = curr_iou.index(max_iou)
                del gt_bboxes[file][idx]
            else:
                fp.append(p_bbox[-1])

    all_detects = tp + fp
    all_detects = np.array(sorted(all_detects))
    tp = np.array(sorted(tp))

    res = []
    threshholds = all_detects
    threshholds = np.append(threshholds, np.max(all_detects) + 1)

    for c in threshholds:
        true_detects_c = np.sum(tp >= c)
        all_detects_c = np.sum(all_detects >= c)

        if all_detects_c == 0:
            precision = 1
        else:
            precision = true_detects_c / all_detects_c
        recall = true_detects_c / all_rects
        res.append([precision, recall, c])

    auc = 0
    for i in range(len(res) - 1):
        auc += (res[i][1] - res[i+1][1]) * (res[i][0] + res[i+1][0]) / 2

    return auc
    # your code here /\


# =============================== 7 NMS ========================================
import copy

def nms(detections_dictionary, iou_thr=0.3):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    filenames = list(detections_dictionary.keys())
    nms_detects = {}
    for file in filenames:

        detections = copy.deepcopy(detections_dictionary[file])
        detections = sorted(detections, key=lambda detections: detections[-1], reverse=True)
        dets_for_file = []

        while len(detections) != 0:
            curr_file = copy.copy(detections[0])
            del detections[0]
            dets_for_file.append(curr_file)
            cur_lenght = len(detections)
            i = 0
            while i < cur_lenght:
                if calc_iou(curr_file[:-1], detections[i][:-1]) > iou_thr:
                    del detections[i]
                    cur_lenght -= 1
                else:
                    i += 1

        nms_detects[file] = dets_for_file

    return nms_detects
    # your code here /\
