from torch.utils.data import Dataset
import numpy as np
import os
import json
from PIL import Image, ImageDraw


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        supported_image_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
        if train:
            images_list = os.listdir(path_Data+'train/images/')
            masks_list = [f for f in os.listdir(path_Data+'train/masks/') 
                         if any(f.lower().endswith(ext) for ext in supported_image_formats)]
            json_list = [f for f in os.listdir(path_Data+'train/masks/') if f.endswith('.json')]
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            json_list = sorted(json_list)
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                json_path = path_Data+'train/masks/' + json_list[i] if i < len(json_list) else None
                self.data.append([img_path, mask_path, json_path])
            self.transformer = config.train_transformer
        else:
            images_list = os.listdir(path_Data+'val/images/')
            masks_list = [f for f in os.listdir(path_Data+'val/masks/') 
                         if any(f.lower().endswith(ext) for ext in supported_image_formats)]
            json_list = [f for f in os.listdir(path_Data+'val/masks/') if f.endswith('.json')]
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            json_list = sorted(json_list)
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                json_path = path_Data+'val/masks/' + json_list[i] if i < len(json_list) else None
                self.data.append([img_path, mask_path, json_path])
            self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path, json_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        
        if os.path.exists(msk_path):
            msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        elif json_path is not None:
            with open(json_path, 'r') as f:
                label_data = json.load(f)
            
            msk = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
            pil_msk = Image.fromarray(msk[:, :, 0])
            draw = ImageDraw.Draw(pil_msk)
            
            for shape in label_data['shapes']:
                points = [(int(x), int(y)) for x, y in shape['points']]
                draw.polygon(points, fill=255)
            
            msk = np.expand_dims(np.array(pil_msk), axis=2) / 255
        else:
            raise FileNotFoundError(f"Neither png nor json mask file found for image {img_path}")
            
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
        
    