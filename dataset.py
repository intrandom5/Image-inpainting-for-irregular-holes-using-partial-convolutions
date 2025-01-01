import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from random import randint
from PIL import Image
import numpy as np
import cv2


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # 컬러 이미지로 변환
        if self.transform:
            image = self.transform(image)

        mask = self.generate_irregular_mask(image.size())
        corrupted_image = image * mask  # 가려진 흑백 이미지 생성
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

        return corrupted_image, image, mask
    
    def generate_irregular_mask(self, size):
        """Generates a random irregular mask with lines, circles and elipses"""
        if len(size) == 3:
            _, height, width = size
        else:
            height, width = size

        img = np.zeros((height, width), np.uint8)

        # Set size scale
        size = int((width + height) * 0.03)
        if width < 64 or height < 64:
            raise Exception("Width and Height of mask must be at least 64!")
        
        # Draw random lines
        for _ in range(randint(1, 20)):
            x1, x2 = randint(1, width), randint(1, width)
            y1, y2 = randint(1, height), randint(1, height)
            thickness = randint(3, size)
            cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
            
        # Draw random circles
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, width), randint(1, height)
            radius = randint(3, size)
            cv2.circle(img,(x1,y1),radius,(1,1,1), -1)
            
        # Draw random ellipses
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, width), randint(1, height)
            s1, s2 = randint(1, width), randint(1, height)
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(3, size)
            cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)
        
        return 1-img

def get_loader(img_paths, batch_size: int, shuffle: bool):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),               # 크기 조정
        transforms.ToTensor(),                       # Tensor로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화 (-1 ~ 1 범위)
    ])
    dataset = ImageDataset(img_paths, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)