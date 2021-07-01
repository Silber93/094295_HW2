import os
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import models
import preproccess

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BBOX_STATE_PATH = 'saved_models/box_1.pkl'
MASK_STATE_PATH = 'saved_models/mask.pkl'
COMBINED_STATE_PATH = 'saved_models/combined_detector.pkl'


def predict_1(test_dir):
    test_set = preproccess.ImageDataset(test_dir, 'test')
    bbox_model = models.BoxModel1().to(DEVICE)
    bbox_model.load_state_dict(torch.load(BBOX_STATE_PATH))
    mask_model = models.MaskModel1().to(DEVICE)
    mask_model.load_state_dict(torch.load(MASK_STATE_PATH))
    test_dl = DataLoader(test_set, batch_size=64)
    values_to_save = []
    for img_name, img_tensor, o_box, adj_box, r_box, h_ratio, w_ratio, mask in test_dl:
        img_tensor = img_tensor.to(DEVICE)
        pred_bbox = bbox_model(img_tensor)
        pred_bbox = torch.round(pred_bbox.detach())
        pred_mask = mask_model(img_tensor)
        pred_mask = torch.round(pred_mask.detach())
        for i in range(img_tensor.shape[0]):
            x, y, w, h = [e.item() for e in pred_bbox[i]]
            x, w = round(x / w_ratio[i].item()), round(w / w_ratio[i].item())
            y, h = round(y / h_ratio[i].item()), round(h / h_ratio[i].item())
            mask_bool = 'True' if pred_mask[i] == 1 else 'False'
            values_to_save.append([img_name[i], x, y, w, h, mask_bool])
    prediction_df = pd.DataFrame(values_to_save, columns=['filename', 'x', 'y', 'w', 'h', 'proper_mask'])
    prediction_df.to_csv("prediction.csv", index=False, header=True)
    print("test finished, results saved to prediction.csv")


def predict_2(test_dir):
    test_set = preproccess.ImageDataset(test_dir, 'test')
    model = models.ResNetCombined().to(DEVICE)
    test_dl = DataLoader(test_set, batch_size=64)
    values_to_save = []
    for img_name, img_tensor, o_box, adj_box, r_box, h_ratio, w_ratio, mask in test_dl:
        bbox_pred, mask_pred = model(img_tensor.to(DEVICE))
        for i in range(img_tensor.shape[0]):
            x, y, w, h = [e.item() * 224 for e in bbox_pred[i]]
            x, w = round(x / w_ratio[i].item()), round(w / w_ratio[i].item())
            y, h = round(y / h_ratio[i].item()), round(h / h_ratio[i].item())
            mask_bool = 'True' if torch.round(mask_pred[i]) == 1 else 'False'
            values_to_save.append([img_name[i], x, y, w, h, mask_bool])
    prediction_df = pd.DataFrame(values_to_save, columns=['filename', 'x', 'y', 'w', 'h', 'proper_mask'])
    prediction_df.to_csv("prediction.csv", index=False, header=True)
    print("test finished, results saved to prediction.csv")



# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder
# files = os.listdir(args.input_folder)

#####
# TODO - your prediction code here

predict_1(args.input_folder)
# predict_2(args.input_folder)

# # Example (A VERY BAD ONE):
# bbox_pred = np.random.randint(0, high=224, size=(4, len(files)))
# proper_mask_pred = np.random.randint(2, size=len(files)).astype(np.bool)
# prediction_df = pd.DataFrame(zip(files, *bbox_pred, proper_mask_pred),
#                              columns=['filename', 'x', 'y', 'w', 'h', 'proper_mask'])
# ####
#
# TODO - How to export prediction results
# prediction_df.to_csv("prediction.csv", index=False, header=True)
