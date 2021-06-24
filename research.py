import torch
import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


tsfrm = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

train_folder = 'data/train'
test_folder = 'data/test'
train_proccessed_folder = 'data/train_proccessed'
train_with_boxes = 'data/train_with_boxes'




def boundries(train_folder, train_with_boxes):
    if not os.path.exists(train_with_boxes):
        os.mkdir(train_with_boxes)
    for i, filename in enumerate(sorted(os.listdir('data/train'))):
        image = Image.open(f'{train_folder}/{filename}').convert("RGB")
        torch_img = tsfrm(image)
        img_id, box, mask = filename.replace('.jpg', '').split('__')
        x, y, w, h = [int(x) for x in box.replace('[', '').replace(']', '').split(', ')]
        print(i, x, y, w, h, torch_img[0].shape)
        torch_img[0][y-1:y+2, x-1:x+2] = 1
        torch_img[1][y-1:y+2, x-1:x+2] = 0
        torch_img[2][y-1:y+2, x-1:x+2] = 0
        w = x + w
        h = y + h
        torch_img[0][h - 1:h + 2, w - 1:w + 2] = 1
        torch_img[1][h - 1:h + 2, w - 1:w + 2] = 0
        torch_img[2][h - 1:h + 2, w - 1:w + 2] = 0
        save_image(torch_img, f'{train_with_boxes}/{filename}')
        if i == 100:
            break

def shapes(train_folder, test_folder):
    min_h, max_h = 999, 0
    min_w, max_w = 999, 0
    for folder in [train_folder, test_folder]:
        for i, filename in enumerate(sorted(os.listdir(folder))):
            image = Image.open(f'{folder}/{filename}').convert("RGB")
            torch_img = tsfrm(image)
            min_h = min(min_h, torch_img.shape[1])
            max_h = max(max_h, torch_img.shape[1])
            min_w = min(min_w, torch_img.shape[2])
            max_w = max(max_w, torch_img.shape[2])
            if i % 1000 == 0:
                print(i, min_h, max_h, min_w, max_w)
    print(f'min_h: {min_h}, max_h: {max_h}, min_w: {min_w}, max_w: {max_w}')

boundries(train_folder, train_with_boxes)
# shapes(train_folder, test_folder)