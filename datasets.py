from torch.utils.data import Dataset
import torchvision.transforms as transforms

import glob 
import os 
from PIL import Image

categories = {
    'G3' : 0, 
    'G4' : 1,
    'G5' : 2,
    'Normal' : 3, 
    'Stroma' : 4 
}

class ImageDataset(Dataset):
    def __init__(self, data_dir, txt_dir=None, resolution=256, is_label=False):
        
        self.image_paths = [] 
        self.labels = [] 
        # if txt_dir is not None: 
        with open(txt_dir, 'r') as f:
            for line in f:
                line = os.path.join(data_dir, line.strip())
                if os.path.exists(line):
                    self.image_paths.append(line)
                    self.labels.append(line.split('/')[-2]) 
        # else: 
        #     self.image_paths = glob.glob(os.path.join(data_dir, "**", "*.tif"), recursive=True)
        #     self.image_paths.extend(glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True))
        #     self.labels = [os.path.basename(os.path.dirname(path)) for path in self.image_paths]
        
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # 이미지를 -1에서 1 범위로 정규화
        ])
        
        self.is_label = is_label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = categories[self.labels[idx]]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        if self.is_label: 
            return image, label
        else: 
            return image