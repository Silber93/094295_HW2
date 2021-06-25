import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import h5py

import matplotlib.pyplot as plt

TO_TENSOR = transforms.Compose([transforms.ToTensor()])

RESIZE = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

TRANS_2 = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((224, 224)),
    transforms.Pad(padding=(0, 0, 100, 0)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def transform_image(pil_image):
    img = TO_TENSOR(pil_image)
    t_func = transforms.Compose([
    transforms.Pad(padding=(0, 0, 224 - img.shape[2], 224 - img.shape[1])),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return t_func(img)


# class ImageDataset(Dataset):
#     def __init__(self, img_dir, transform_func=TRANSFORM_FUNC):
#         self.img_dir = img_dir
#         self.transform_func = transform_func
#         self.img_data = {}
#         for img_file in sorted(os.listdir(img_dir)):
#             if '.jpg' not in img_file:
#                 continue
#             features = img_file.replace('.jpg', '').split('__')
#             img_id = int(features[0])
#             box = torch.Tensor([int(x) for x in features[1].replace('[', '').replace(']', '').split(', ')]).float()
#             mask = 1 if features[2] == 'True' else 0
#             self.img_data[img_id] = {'name': img_file, 'box': box, 'mask': mask}
#
#     def __len__(self):
#         return len(self.img_data)
#
#     def __getitem__(self, idx):
#         img_path = f'{self.img_dir}/{self.img_data[idx]["name"]}'
#         image = Image.open(img_path)
#         image = self.transform_func(image)
#         return image, self.img_data[idx]['box'], self.img_data[idx]['mask']
#
#
# class ImageDataset2(Dataset):
#     def __init__(self, data_type, img_dir, transform_func=TRANSFORM_FUNC):
#         self.img_dir = img_dir
#         self.transform_func = transform_func
#         self.img_data = {}
#         self.h5py_filepath = f'{H5_DIR}/{data_type}.h5'
#         for i, img_file in enumerate(sorted(os.listdir(img_dir))):
#             if '.jpg' not in img_file:
#                 continue
#             if i % 1000 == 0:
#                 print(i)
#             features = img_file.replace('.jpg', '').split('__')
#             img_id = int(features[0])
#             box = torch.Tensor([int(x) for x in features[1].replace('[', '').replace(']', '').split(', ')]).float()
#             mask = 1 if features[2] == 'True' else 0
#             self.img_data[img_id] = {'box': box, 'mask': mask}
#
#     def __len__(self):
#         return len(self.img_data)
#
#     def __getitem__(self, idx):
#         f = h5py.File(self.h5py_filepath, 'r')
#         return torch.tensor(f.get(str(idx))), self.img_data[idx]['box'], self.img_data[idx]['mask']
#
#
# class ImageDataset3(Dataset):
#     def __init__(self, img_dir, proccessed_img_dir,  transform_func=TRANSFORM_FUNC):
#         self.proccessed_img_dir = proccessed_img_dir
#         if not os.path.exists(proccessed_img_dir):
#             os.mkdir(proccessed_img_dir)
#         self.transform_func = transform_func
#         self.img_data = {}
#         i = 0
#         for img_file in sorted(os.listdir(img_dir)):
#             if '.jpg' not in img_file:
#                 continue
#             if i % 1000 == 0:
#                 print(i)
#             features = img_file.replace('.jpg', '').split('__')
#             img_id = int(features[0])
#             box = torch.Tensor([int(x) for x in features[1].replace('[', '').replace(']', '').split(', ')]).float()
#             mask = 1 if features[2] == 'True' else 0
#             img_path = f'{img_dir}/{img_file}'
#             image = Image.open(img_path)
#             image = self.transform_func(image)
#             save_image(image, f'{proccessed_img_dir}/{img_file}')
#             self.img_data[img_id] = {'name': img_file, 'box': box, 'mask': mask}
#             i += 1
#
#     def __len__(self):
#         return len(self.img_data)
#
#     def __getitem__(self, idx):
#         img_path = f'{self.proccessed_img_dir}/{self.img_data[idx]["name"]}'
#         image = Image.open(img_path)
#         transform_func_2 = transforms.Compose([
#             transforms.ToTensor()
#         ])
#         image = transform_func_2(image)
#         return image, self.img_data[idx]['box'], self.img_data[idx]['mask']


class ImageDataset(Dataset):
    def __init__(self, img_dir, data_type):
        self.img_data = {}
        all_files = [x for x in sorted(os.listdir(img_dir)) if '.jpg' in x]
        print(f"loading {data_type} dataset, size: {len(all_files)}")
        for i, img_file in enumerate(all_files):
            if i % int(len(all_files) / 4) == 0:
                print(f"{100 * (i/len(all_files))}%")
            if i == 400 and data_type == 'test':
                break
            if i == 1600:
                break
            features = img_file.replace('.jpg', '').split('__')
            img_id = i
            box = torch.Tensor([int(x) for x in features[1].replace('[', '').replace(']', '').split(', ')]).float()
            mask = 1 if features[2] == 'True' else 0
            self.img_data[img_id] = {'id': features[0], 'original_box': box, 'mask': mask}
            img_path = f'{img_dir}/{img_file}'
            image = Image.open(img_path)
            image = TO_TENSOR(image)
            self.img_data[img_id]['relative_box'] = torch.Tensor([box[0] / image.shape[2], box[1] / image.shape[1],
                                                                    box[2] / image.shape[2], box[3] / image.shape[1]])
            image = RESIZE(image)
            self.img_data[img_id]['tensor'] = image
        print(f"{data_type} dataset load completed\n")

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        return self.img_data[idx]['tensor'], self.img_data[idx]['original_box'], \
               self.img_data[idx]['relative_box'], self.img_data[idx]['mask']


class ImageDataset2(Dataset):
    def __init__(self, img_dir, data_type):
        self.img_data = {}
        all_files = [x for x in sorted(os.listdir(img_dir)) if '.jpg' in x]
        print(f"loading {data_type} dataset, size: {len(all_files)}")
        i = 0
        for img_file in all_files:
            if i % int(len(all_files) / 4) == 0:
                print(f"{100 * (i/len(all_files))}%")
            if i == 400 and data_type == 'test':
                break
            if i == 1600:
                break
            features = img_file.replace('.jpg', '').split('__')
            img_id = i
            box = torch.Tensor([int(x) for x in features[1].replace('[', '').replace(']', '').split(', ')]).float()
            if any([box[i] <= 0 for i in range(4)]):
                print(f'ignoring image {features[0]} with bad bbox of: {box}')
                continue
            mask = 1 if features[2] == 'True' else 0
            img_path = f'{img_dir}/{img_file}'
            image = Image.open(img_path)
            image = transform_image(image)
            self.img_data[img_id] = {'id': features[0], 'original_box': box, 'mask': mask}
            self.img_data[img_id]['relative_box'] = torch.Tensor([box[0] / image.shape[2], box[1] / image.shape[1],
                                                                    box[2] / image.shape[2], box[3] / image.shape[1]])
            self.img_data[img_id]['name'] = img_file
            image = RESIZE(image)
            self.img_data[img_id]['tensor'] = image
            i += 1
        print(f"{data_type} dataset load completed\n")

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        return self.img_data[idx]['name'], \
               self.img_data[idx]['tensor'], self.img_data[idx]['original_box'], \
               self.img_data[idx]['relative_box'], self.img_data[idx]['mask']