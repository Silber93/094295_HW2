import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


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


class ImageDataset(Dataset):
    def __init__(self, img_dir, data_type):
        self.img_data = {}
        self.minmax = {'min_x': 224, 'min_y': 224, 'min_w': 224, 'min_h': 224,
                       'max_x': 0, 'max_y': 0, 'max_w': 0, 'max_h': 0}
        all_files = [x for x in sorted(os.listdir(img_dir)) if '.jpg' in x]
        print(f"loading {data_type} dataset, size: {len(all_files)}")
        i = 0
        for img_file in all_files:
            if i % int(len(all_files) / 4) == 0:
                print(f"{100 * (i/len(all_files))}%")
            features = img_file.replace('.jpg', '').split('__')
            img_id = i
            box = torch.Tensor([int(x) for x in features[1].replace('[', '').replace(']', '').split(', ')]).float()
            self.minmax['min_x'] = min(self.minmax['min_x'], box[0].item())
            self.minmax['min_y'] = min(self.minmax['min_y'], box[1].item())
            self.minmax['min_w'] = min(self.minmax['min_w'], box[2].item())
            self.minmax['min_h'] = min(self.minmax['min_h'], box[3].item())
            self.minmax['max_x'] = max(self.minmax['max_x'], box[0].item())
            self.minmax['max_y'] = max(self.minmax['max_y'], box[1].item())
            self.minmax['max_w'] = max(self.minmax['max_w'], box[2].item())
            self.minmax['max_h'] = max(self.minmax['max_h'], box[3].item())
            if any([box[i] <= 0 for i in range(4)]):
                continue
            mask = 1 if features[2] == 'True' else 0
            img_path = f'{img_dir}/{img_file}'
            image = Image.open(img_path)
            image = transforms.ToTensor()(image)
            w_out_of_bounds = box[0] + box[2] >= image.shape[2]
            h_out_of_bounds = box[1] + box[3] >= image.shape[1]
            if w_out_of_bounds or h_out_of_bounds:
                if w_out_of_bounds:
                    box[2] = image.shape[2] - box[0] - 1
                if h_out_of_bounds:
                    box[3] = image.shape[1] - box[1] - 1
            h_ratio = 224 / image.shape[1]
            w_ratio = 224 / image.shape[2]
            adjusted_bbox = torch.round(
                torch.Tensor([box[0] * w_ratio, box[1] * h_ratio, box[2] * w_ratio, box[3] * h_ratio]))
            image = RESIZE(image)
            self.img_data[img_id] = {'id': features[0],
                                     'name': img_file,
                                     'tensor': image,
                                     'original_box': box,
                                     'adjusted_box': adjusted_bbox,
                                     'relative_box': adjusted_bbox / 224,
                                     'h_ratio': h_ratio,
                                     'w_ratio': w_ratio,
                                     'mask': mask}
            i += 1
        print(f"{data_type} dataset of {len(self.img_data)} has successfully loaded\n")

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        return self.img_data[idx]['name'], \
               self.img_data[idx]['tensor'], \
               self.img_data[idx]['original_box'], \
               self.img_data[idx]['adjusted_box'], \
               self.img_data[idx]['relative_box'], \
               self.img_data[idx]['h_ratio'], \
               self.img_data[idx]['w_ratio'], \
               self.img_data[idx]['mask']
