import json
from torchvision.datasets.folder import default_loader
from collections import defaultdict
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image


class COCO(Dataset):
    def __init__(self, coco_dir, split='train', transform=None):
        self.image_dir = os.path.join(coco_dir, f"images/{split}2017/")
        with open(os.path.join(coco_dir, f"annotations/instances_{split}2017.json"), 'r') as file:
            coco = json.load(file)
        
        self.transform = transform
        self.annIm_dict = defaultdict(list)        
        self.cat_dict = {} 
        self.annId_dict = {}
        self.im_dict = {}

        for ann in coco['annotations']:           
            self.annIm_dict[ann['image_id']].append(ann) 
            self.annId_dict[ann['id']] = ann
        
        for img in coco['images']:
            self.im_dict[img['id']] = img
        
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat

        
    def __len__(self):
        return len(list(self.im_dict.keys()))
    
    def __getitem__(self, idx):
        img = self.im_dict[idx]
        path = os.path.join(self.image_dir, img['file_name'])
        image = default_loader(path)
        if self.transform is not None:
            image = image.resize(self.transform, Image.LANCZOS)

        return image, path
        
        
    def get_targets(self, idx):
        return [self.cat_dict[ann['category_id']]['name'] for ann in self.annIm_dict[idx]]
    
    def get_categories(self, supercategory):
        return [self.cat_dict[cat_id]['name'] for cat_id in self.cat_dict.keys() if self.cat_dict[cat_id]['supercategory']==supercategory]
    

    def get_all_supercategories(self):
        return {self.cat_dict[cat_id]['supercategory'] for cat_id in self.cat_dict.keys()}
    
    def get_spurious_supercategories(self):
        return ['kitchen', 'food', 'vehicle',
                'furniture', 'appliance', 'indoor',
                'outdoor', 'electronic', 'sports',
                'accessory', 'animal']
    
    def get_no_classes(self, supercategories):
        return len([self.cat_dict[cat_id]['name'] for cat_id in self.cat_dict.keys() if self.cat_dict[cat_id]['supercategory'] in supercategories])
    

    def get_imgIds(self):
        return list(self.im_dict.keys())
    
    def get_all_targets_names(self):
        return [self.cat_dict[cat_id]['name'] for cat_id in self.cat_dict.keys()]
    
    def get_imgIds_by_class(self, present_classes=[], absent_classes=[]):
        # Return images that has at least one of the present_classes, and none of the absent_classes
        ids = []
        for img_id in self.get_imgIds():
            targets = self.get_targets(img_id)
            flag = False
            for c in present_classes:
                if c in targets:
                    flag = True
                    break
            for c in absent_classes:
                if c in targets:
                    flag = False
                    break
            if flag:
                ids.append(img_id)
        return ids