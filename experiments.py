import matplotlib.pyplot as plt
from matplotlib import patches
from datetime import datetime
import cv2

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from torchvision import transforms
from PIL import Image

import preproccess
import models


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BOX_1_HYPERPARAMS = {
    'lr': 0.001,
    'epochs': 50,
    'batch_size': 64,
    'target': 'box'}

BOX_2_HYPERPARAMS = {
    'lr': 0.005,
    'epochs': 50,
    'batch_size': 64,
    'target': 'box'}

MASK_HYPERPARAMS = {
    'lr': 0.001,
    'epochs': 30,
    'batch_size': 64,
    'target': 'mask'}

TRAIN_FOLDER = 'data/train'
TEST_FOLDER = 'data/test'


def calc_iou(bbox_a, bbox_b):
    """
    Calculate intersection over union (IoU) between two bounding boxes with a (x, y, w, h) format.
    :param bbox_a: Bounding box A. 4-tuple/list.
    :param bbox_b: Bounding box B. 4-tuple/list.
    :return: Intersection over union (IoU) between bbox_a and bbox_b, between 0 and 1.
    """
    x1, y1, w1, h1 = bbox_a
    x2, y2, w2, h2 = bbox_b
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0.0 or h_intersection <= 0.0:  # No overlap
        return 0.0
    intersection = w_intersection * h_intersection
    union = w1 * h1 + w2 * h2 - intersection    # Union = Total Area - Intersection
    return intersection / union


def show_images_and_bboxes(epoch, img_dir, img_name, y_true, y_pred, mask):
    """
    Plot images with bounding boxes. Predicts random bounding boxes and computes IoU.
    :param data: Iterable with (filename, image_id, bbox, proper_mask) structure.
    :param image_dir: Path to directory with images.
    :return: None
    """
    x1, y1, w1, h1 = y_true.detach().numpy()
    x2, y2, w2, h2 = y_pred.detach().numpy()
    # Calculate IoU
    iou = calc_iou(y_true, y_pred)
    # Plot image and bboxes
    fig, ax = plt.subplots()
    func = transforms.Compose([transforms.ToTensor()])
    img_tensor = func(Image.open(f"{img_dir}/{img_name}"))
    ax.imshow(img_tensor.permute(1, 2, 0))
    rect = patches.Rectangle((x1, y1), w1, h1,
                             linewidth=2, edgecolor='g', facecolor='none', label='ground-truth')
    ax.add_patch(rect)
    rect = patches.Rectangle((x2, y2), w2, h2,
                             linewidth=2, edgecolor='b', facecolor='none', label='predicted')
    ax.add_patch(rect)
    fig.suptitle(f"epoch={epoch}, proper_mask={mask}, IoU={iou:.2f}")
    ax.axis('off')
    fig.legend()
    plt.show()


def plot_history(history, model_type):
    for graph in ['loss', 'score']:
        plt.figure()
        for k in history:
            if graph not in k:
                continue
            plt.plot(history[k].keys(), history[k].values(), label=k)
        plt.legend()
        plt.title(graph)
        plt.xlabel('epochs')
        plt.ylabel(graph)
        plt.show()


def iou(y_true, y_pred):
    y_pred = torch.round(y_pred)

    w_intersection = torch.min(y_true[:, 0] + y_true[:, 2], y_pred[:, 0] + y_pred[:, 2]) \
                     - torch.max(y_true[:, 0], y_pred[:, 0])
    h_intersection = torch.min(y_true[:, 1] + y_true[:, 3], y_pred[:, 1] + y_pred[:, 3]) \
                     - torch.max(y_true[:, 1], y_pred[:, 1])
    w_intersection = torch.max(torch.zeros(size=w_intersection.shape), w_intersection)
    h_intersection = torch.max(torch.zeros(size=h_intersection.shape), h_intersection)
    inter_area = w_intersection * h_intersection
    union = y_true[:, 2] * y_true[:, 3] + y_pred[:, 2] * y_pred[:, 3] - inter_area
    res = inter_area / union
    return torch.sum(res) / res.shape[0]

    # y_pred = torch.round(y_pred)
    # inter_x_start = torch.max(y_true[:, 0], y_pred[:, 0]).unsqueeze(1)
    # inter_y_start = torch.max(y_true[:, 1], y_pred[:, 1]).unsqueeze(1)
    # inter_start = torch.cat([inter_x_start, inter_y_start], dim=1)
    # inter_x_end = torch.min(y_true[:, 0] + y_true[:, 2], y_pred[:, 0] + y_pred[:, 2]).unsqueeze(1)
    # inter_y_end = torch.min(y_true[:, 1] + y_true[:, 3], y_pred[:, 1] + y_pred[:, 3]).unsqueeze(1)
    # inter_end = torch.cat([inter_x_end, inter_y_end], dim=1)
    # inter_dim = torch.max(inter_end - inter_start, torch.zeros(size=inter_end.shape))
    # inter_area = inter_dim[:, 0] * inter_dim[:, 1]
    # union_area = y_true[:, 2] * y_true[:, 3] + y_pred[:, 2] * y_pred[:, 3] - inter_area
    # res = inter_area / union_area
    # return torch.sum(res) / res.shape[0]


def accuracy(y_true, y_pred):
    y_pred_bool = torch.argmax(y_pred, dim=1)
    acc = y_true == y_pred_bool
    return torch.sum(acc) / acc.shape[0]


def perform_experiment(model, train_set, test_set, hyperparams):
    batched_train = DataLoader(train_set, batch_size=hyperparams['batch_size'])
    batched_test = DataLoader(test_set, batch_size=hyperparams['batch_size'])
    train_batches = int(len(train_set) / hyperparams['batch_size']) + 1
    test_batches = int(len(test_set) / hyperparams['batch_size']) + 1
    history = {'train_loss': {}, 'train_score': {}, 'test_loss': {}, 'test_score': {}}
    start_time = datetime.now()
    optimizer = Adam(model.parameters(), lr=hyperparams['lr'])
    for epoch in range(hyperparams['epochs']):
        for k in history:
            if 'train' in k or epoch % 5 == 0:
                history[k][epoch] = 0
        i = 0
        for img_name, x, o_box, r_box, mask in batched_train:
            optimizer.zero_grad()
            y_pred = model(x.to(DEVICE))
            if model.name == 'box_1':
                y_true = o_box.to(DEVICE)
                score = iou(y_true, y_pred)
            if model.name == 'box_2':
                y_true = r_box.to(DEVICE)
                x_real = torch.round(y_pred[:, torch.Tensor([0, 2]).long()] * o_box[:, 0].unsqueeze(1))
                y_real = torch.round(y_pred[:, torch.Tensor([1, 3]).long()] * o_box[:, 1].unsqueeze(1))
                y_pred_real = torch.cat([x_real[:, 0].unsqueeze(1), y_real[:, 0].unsqueeze(1),
                                         x_real[:, 1].unsqueeze(1), y_real[:, 1].unsqueeze(1)], dim=1)
                if epoch == 0 and i == 0:
                    y_pred_real[:, 2:] = y_pred_real[:, 2:] * 2
                y_pred = y_pred_real
                score = iou(o_box, y_pred_real)
            if model.name == 'mask':
                y_true = mask
                score = accuracy(y_true, y_pred)
            loss = model.loss_func(y_true, y_pred)

            if i == 0:
                if 'box' in model.name:
                    print([x.item() for x in y_true[0]], [x.item() for x in y_pred[0]], loss.item(), score.item())
                else:
                    print(y_true[0], y_pred[0], loss.item(), score.item())
                if epoch % 5 == 0:
                    if model.name == 'box_1':
                        show_images_and_bboxes(epoch, TRAIN_FOLDER, img_name[0], y_true[0], y_pred[0], mask[0])
                    if model.name == 'box_2':
                        show_images_and_bboxes(epoch, TRAIN_FOLDER, img_name[0], o_box[0], y_pred_real[0], mask[0])
                i += 1
            history['train_loss'][epoch] += loss.item()
            history['train_score'][epoch] += score.item()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            i = 0
            for img_name, x, o_box, r_box, mask in batched_test:
                y_pred = model(x.to(DEVICE))
                if model.name == 'box_1':
                    y_true = o_box.to(DEVICE)
                    score = iou(o_box, y_pred) if 'box' in model.name else accuracy(y_true, y_pred)
                if model.name == 'box_2':
                    y_true = r_box.to(DEVICE)
                    x_real = torch.round(y_pred[:, torch.Tensor([0, 2]).long()] * o_box[:, 0].unsqueeze(1))
                    y_real = torch.round(y_pred[:, torch.Tensor([1, 3]).long()] * o_box[:, 1].unsqueeze(1))
                    y_pred_real = torch.cat([x_real[:, 0].unsqueeze(1), y_real[:, 0].unsqueeze(1),
                                             x_real[:, 1].unsqueeze(1), y_real[:, 1].unsqueeze(1)], dim=1)
                    score = iou(o_box, y_pred_real)
                if model.name == 'mask':
                    y_true = mask
                    score = accuracy(y_true, y_pred)
                loss = model.loss_func(y_true, y_pred)
                history['test_loss'][epoch] += loss.item()
                history['test_score'][epoch] += score.item()
                if i == 0:
                    if model.name == 'box_1':
                        show_images_and_bboxes(epoch, TEST_FOLDER, img_name[0], y_true[0], y_pred[0], mask[0], )
                    if model.name == 'box_2':
                        show_images_and_bboxes(epoch, TEST_FOLDER, img_name[0], o_box[0], y_pred_real[0], mask[0])
                    i += 1
        for k in history:
            if 'train' in k or epoch % 5 == 0:
                history[k][epoch] /= (train_batches if 'train' in k else test_batches)
        dt = datetime.now() - start_time
        print(f'{datetime.now()} - epoch: {epoch}, average train loss: {history["train_loss"][epoch]}, '
              f'average train score: {history["train_score"][epoch]}')
        if epoch % 5 == 0:
            print(f'{datetime.now()} - epoch: {epoch}, average test loss: {history["test_loss"][epoch]}, '
              f'average test score: {history["test_score"][epoch]}')
    return history


print("---------------PREPROCESS---------------")
train_dataset = preproccess.ImageDataset2(TRAIN_FOLDER, "train")
test_dataset = preproccess.ImageDataset2(TEST_FOLDER, "test")

print(f"----------------------------------------EXPERIMENTS(DEVICE: {DEVICE})-----------------------------------------")
print("box model 1:", BOX_1_HYPERPARAMS)
box_model = models.BoxModel1()
box_model_hist = perform_experiment(box_model, train_dataset, test_dataset, BOX_1_HYPERPARAMS)
plot_history(box_model_hist, 'box')

print("box model 2:", BOX_2_HYPERPARAMS)
box_model = models.BoxModel2()
box_model_hist = perform_experiment(box_model, train_dataset, test_dataset, BOX_2_HYPERPARAMS)
plot_history(box_model_hist, 'box')


# print("mask model 1:", MASK_HYPERPARAMS)
# mask_model = models.MaskModel2()
# mask_model_hist = perform_experiment(mask_model, train_dataset, test_dataset, MASK_HYPERPARAMS)
# plot_history(mask_model_hist, 'mask')
