import os
import json
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.gt_dic = {}
        self.gt_file = os.path.join(self.data_dir, 'metadata.jsonl')
        with open(self.gt_file, "r") as gt_file: 
            for line in gt_file: 
                gt_json = json.loads(line)
                file_name = gt_json['file_name']
                gt = gt_json['ground_truth']
                self.gt_dic[file_name] = gt

    def __len__(self):
        return len(self.gt_dic)

    def __getitem__(self, idx):
        file_name = list(self.gt_dic.keys())[idx]
        img_path = os.path.join(self.data_dir, file_name)
        image = Image.open(img_path)
        gt = self.gt_dic[file_name]
        return image, gt, file_name