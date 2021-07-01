import matplotlib.pyplot as plt
from matplotlib import patches
from datetime import datetime
import cv2
import sys

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from torchvision import transforms
from PIL import Image

import preproccess
import models


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BOX_1_HYPERPARAMS = {
    'lr': 0.005,
    'epochs': 100,
    'batch_size': 64,
    'target': 'box'}

BOX_PLAIN_HYPERPARAMS = {
    'lr': 0.01,
    'epochs': 100,
    'batch_size': 64,
    'target': 'box'}


BOX_2_HYPERPARAMS = {
    'lr': 0.005,
    'epochs': 50,
    'batch_size': 64,
    'target': 'box'}

MASK_HYPERPARAMS = {
    'lr': 0.001,
    'epochs': 50,
    'batch_size': 64,
    'target': 'mask'}

DETECTOR_HYPERPARAMS = {
    'lr': 0.01,
    'epochs': 50,
    'batch_size': 64,
    'target': 'mask'}


TRAIN_FOLDER = 'data/train'
TEST_FOLDER = 'data/test'


def calc_iou(bbox_a, bbox_b):
    x1, y1, w1, h1 = bbox_a
    x2, y2, w2, h2 = bbox_b
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0.0 or h_intersection <= 0.0:  # No overlap
        return 0.0
    intersection = w_intersection * h_intersection
    union = w1 * h1 + w2 * h2 - intersection    # Union = Total Area - Intersection
    return intersection / union


def show_images_and_bboxes(epoch, img_dir, img_name, y_true, y_pred, mask, model_name):
    """
    Plot images with bounding boxes. Predicts random bounding boxes and computes IoU.
    :param data: Iterable with (filename, image_id, bbox, proper_mask) structure.
    :param image_dir: Path to directory with images.
    :return: None
    """
    x1, y1, w1, h1 = y_true.cpu().detach().numpy()
    x2, y2, w2, h2 = y_pred.cpu().detach().numpy()
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
    t = f"epoch={epoch}, proper_mask={mask}, IoU={iou:.2f}\n pred: {round(x2)}, {round(y2)}, {round(w2)}, {round(h2)}"
    fig.suptitle(t)
    ax.axis('off')
    fig.legend()
    plt.show()
    plt.savefig(f'visualised_results/{model_name}_{epoch}_{img_dir.split("/")[1]}.jpg')


def show_images_and_bboxes2(epoch, img_dir, img_name, o_box, tile_scores):
    """
    Plot images with bounding boxes. Predicts random bounding boxes and computes IoU.
    :param data: Iterable with (filename, image_id, bbox, proper_mask) structure.
    :param image_dir: Path to directory with images.
    :return: None
    """
    x1, y1, w1, h1 = o_box.detach().numpy()
    # Calculate IoU
    # Plot image and bboxes
    fig, ax = plt.subplots()
    func = transforms.Compose([transforms.ToTensor()])
    img_tensor = func(Image.open(f"{img_dir}/{img_name}"))
    # ax.imshow(img_tensor.permute(1, 2, 0))
    rect = patches.Rectangle((x1, y1), w1, h1,
                             linewidth=2, edgecolor='g', facecolor='none', label='ground-truth')
    ax.add_patch(rect)

    t = torch.zeros(size=(3, 224, 224))
    for i in range(tile_scores.shape[0]):
        x = i // 56
        y = i % 56
        h = torch.Tensor([4 * x] * 4 + [4 * x + 1] * 4 + [4 * x + 2] * 4 + [4 * x + 3] * 4).long()
        w = torch.Tensor(list(range(4 * y, (4 * y) + 4)) * 4).long()
        t[0][h, w] = tile_scores[i]
        t[1][80, 70] = 1
        t[2][80, 70] = 1
    ax.imshow(t.permute(1, 2, 0).detach().numpy())
    fig.suptitle(f"epoch={epoch}")
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
        plt.savefig(f'visualised_results/{model_type}_{graph}.jpg')


def plot_combined_history(history, model_type):
    for graph in ['iou', 'accuracy']:
        plt.figure()
        plt.plot(history['train_' + graph].keys(), history['train_' + graph].values(), label='train_' + graph)
        plt.plot(history['test_' + graph].keys(), history['test_' + graph].values(), label='test_' + graph)
        plt.legend()
        plt.title(graph)
        plt.xlabel('epochs')
        plt.ylabel(graph)
        plt.savefig(f'visualised_results/{model_type}_{graph}.jpg')


def iou(y_true, y_pred):
    y_pred[:, torch.Tensor([2, 3]).long()] = torch.max(y_pred[:, torch.Tensor([2, 3]).long()],
                                                       torch.Tensor([1]).to(DEVICE))
    y_pred = torch.round(y_pred)
    w_intersection = torch.min(y_true[:, 0] + y_true[:, 2], y_pred[:, 0] + y_pred[:, 2]) \
                     - torch.max(y_true[:, 0], y_pred[:, 0])
    h_intersection = torch.min(y_true[:, 1] + y_true[:, 3], y_pred[:, 1] + y_pred[:, 3]) \
                     - torch.max(y_true[:, 1], y_pred[:, 1])
    w_intersection = torch.max(torch.zeros(size=w_intersection.shape).to(DEVICE), w_intersection)
    h_intersection = torch.max(torch.zeros(size=h_intersection.shape).to(DEVICE), h_intersection)
    inter_area = w_intersection * h_intersection
    union = y_true[:, 2] * y_true[:, 3] + y_pred[:, 2] * y_pred[:, 3] - inter_area
    res = inter_area / union
    return torch.sum(res) / res.shape[0]


def accuracy(y_true, y_pred):
    y_pred_bool = torch.round(y_pred)
    acc = y_true == y_pred_bool.squeeze(1)
    s = torch.sum(acc)
    return torch.sum(acc) / acc.shape[0]


def perform_individual_experiment(model, train_set, test_set, hyperparams):
    batched_train = DataLoader(train_set, batch_size=hyperparams['batch_size'])
    batched_test = DataLoader(test_set, batch_size=hyperparams['batch_size'])
    train_batches = int(len(train_set) / hyperparams['batch_size']) + 1
    test_batches = int(len(test_set) / hyperparams['batch_size']) + 1
    history = {'train_loss': {}, 'train_score': {}, 'test_loss': {}, 'test_score': {}}
    start_time = datetime.now()
    optimizer = Adam(model.parameters(), lr=hyperparams['lr'], weight_decay=0.0001)
    for epoch in range(hyperparams['epochs']):
        for k in history:
            if 'train' in k or epoch % 5 == 0:
                history[k][epoch] = 0
        i = 0
        for img_name, x, o_box, adj_box, r_box, h_ratio, w_ratio, mask in batched_train:
            optimizer.zero_grad()
            y_pred = model(x.to(DEVICE))
            if 'box' in model.name:
                y_true = adj_box.to(DEVICE)
                score = iou(y_true, y_pred)
                loss = model.loss_func(y_true.unsqueeze(1), y_pred)
            if 'mask' in model.name:
                y_true = mask.float().to(DEVICE)
                score = accuracy(y_true, y_pred)
                loss = model.loss_func(y_true.unsqueeze(1), y_pred)

            if i == 0:
                if epoch % 10 == 0:
                    if 'box' in model.name:
                        show_images_and_bboxes(epoch, TRAIN_FOLDER, img_name[0], y_true[0], y_pred[0], mask[0], model.name)
                i += 1
            history['train_loss'][epoch] += loss.item()
            history['train_score'][epoch] += score.item()
            loss.backward()
            optimizer.step()
            y_true = y_true.detach()
        if epoch % 5 == 0:
            i = 0
            for img_name, x, o_box, adj_box, r_box, h_ratio, w_ratio, mask in batched_test:
                y_pred = model(x.to(DEVICE))
                if 'box' in model.name:
                    y_true = o_box.to(DEVICE)
                    score = iou(o_box.to(DEVICE), y_pred) if 'box' in model.name else accuracy(y_true, y_pred)
                if 'mask' in model.name:
                    y_true = mask.float().to(DEVICE)
                    score = accuracy(y_true, y_pred)
                loss = model.loss_func(y_true.unsqueeze(1), y_pred)
                history['test_loss'][epoch] += loss.item()
                history['test_score'][epoch] += score.item()
                if i == 0:
                    if 'box' in model.name:
                        show_images_and_bboxes(epoch, TEST_FOLDER, img_name[0], y_true[0], y_pred[0], mask[0], model.name)
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
    print(f"training of {len(train_set)} images with {hyperparams['epochs']} epochs finished in "
          f"{datetime.now() - start_time}")
    torch.save(model.state_dict(), f'saved_models/{model.name}.pkl')
    return history


def perform_combined_experiment(model, train_set, test_set, hyperparams):
    batched_train = DataLoader(train_set, batch_size=hyperparams['batch_size'])
    batched_test = DataLoader(test_set, batch_size=hyperparams['batch_size'])
    train_batches = int(len(train_set) / hyperparams['batch_size']) + 1
    test_batches = int(len(test_set) / hyperparams['batch_size']) + 1
    history = {'train_loss': {}, 'train_iou': {}, 'train_accuracy': {}, 'test_loss': {}, 'test_iou': {},
               'test_accuracy': {}}
    start_time = datetime.now()
    optimizer = Adam(model.parameters(), lr=hyperparams['lr'])
    for epoch in range(hyperparams['epochs']):
        for k in history:
            if 'train' in k or epoch % 5 == 0:
                history[k][epoch] = 0
        i = 0
        for img_name, x, o_box, adj_box, r_box, h_ratio, w_ratio, mask in batched_train:
            optimizer.zero_grad()
            bbox_pred, mask_pred = model(x.to(DEVICE))
            loss = model.loss_func(mask.to(DEVICE).float().unsqueeze(1), mask_pred, r_box.to(DEVICE), bbox_pred)
            # loss = hyperparams['mult'] * bbox_loss + mask_loss
            if epoch % 10 == 0 and i == 0:
                # print(loss)
                # print(bbox_loss, mask_loss, loss)
                show_images_and_bboxes(epoch, TRAIN_FOLDER, img_name[0], adj_box[0], bbox_pred[0]*224, mask[0], model.name)
                i += 1
            history['train_loss'][epoch] += loss.item()
            iou_score = iou(adj_box.to(DEVICE), torch.round(bbox_pred * 224))
            acc_score = accuracy(mask.to(DEVICE), mask_pred.to(DEVICE))
            history['train_iou'][epoch] += iou_score.item()
            history['train_accuracy'][epoch] += acc_score.item()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            i = 0
            for img_name, x, o_box, adj_box, r_box, h_ratio, w_ratio, mask in batched_test:
                bbox_pred, mask_pred = model(x.to(DEVICE))
                loss = model.loss_func(mask.to(DEVICE).float().unsqueeze(1), mask_pred, r_box.to(DEVICE), bbox_pred)
                history['test_loss'][epoch] += loss.item()
                iou_score = iou(adj_box.to(DEVICE), torch.round(bbox_pred * 224))
                acc_score = accuracy(mask.to(DEVICE), mask_pred.to(DEVICE))
                history['test_iou'][epoch] += iou_score.item()
                history['test_accuracy'][epoch] += acc_score.item()
                if i == 0 and epoch % 10 == 0:
                    show_images_and_bboxes(epoch, TEST_FOLDER, img_name[0], adj_box[0], bbox_pred[0]*224, mask[0], model.name)
                    i += 1
        for k in history:
            if 'train' in k or epoch % 5 == 0:
                history[k][epoch] /= (train_batches if 'train' in k else test_batches)

        print(f'{datetime.now()} - epoch: {epoch}, average train loss: {history["train_loss"][epoch]}, '
              f'average train iou score: {history["train_iou"][epoch]}, average train accuracy: '
              f'{history["train_accuracy"][epoch]}')
        if epoch % 5 == 0:
            print(f'{datetime.now()} - epoch: {epoch}, average test loss: {history["test_loss"][epoch]}, '
                  f'average test iou score: {history["test_iou"][epoch]}, average test accuracy: '
                  f'{history["test_accuracy"][epoch]}')
    print(f"training of {len(train_set)} images with {hyperparams['epochs']} epochs finished in "
          f"{datetime.now() - start_time}")
    torch.save(model.state_dict(), f'saved_models/combined_detector.pkl')
    return history


print("---------------PREPROCESS---------------")
train_dataset = preproccess.ImageDataset(TRAIN_FOLDER, "train")
test_dataset = preproccess.ImageDataset(TEST_FOLDER, "test")

print(f"----------------------------------------EXPERIMENTS(DEVICE: {DEVICE})-----------------------------------------",)

print("box model 1:", BOX_1_HYPERPARAMS)
box_model = models.BoxModel1().to(DEVICE)
box_model_hist = perform_individual_experiment(box_model, train_dataset, test_dataset, BOX_1_HYPERPARAMS)
plot_history(box_model_hist, 'box_1')

torch.cuda.empty_cache()

print("mask model 1:", MASK_HYPERPARAMS)
mask_model = models.MaskModel1().to(DEVICE)
mask_model_hist = perform_individual_experiment(mask_model, train_dataset, test_dataset, MASK_HYPERPARAMS)
plot_history(mask_model_hist, 'mask_1')

torch.cuda.empty_cache()

print("resnet_combined model:", DETECTOR_HYPERPARAMS)
box_model = models.ResNetCombined().to(DEVICE)
box_model_hist = perform_combined_experiment(box_model, train_dataset, test_dataset, DETECTOR_HYPERPARAMS)
plot_combined_history(box_model_hist, box_model.name)
