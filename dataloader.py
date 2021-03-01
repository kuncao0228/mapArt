import os
import torch
import pandas as pd
from skimage import io, transform
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.io import imsave, imread
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


import cv2


from PIL import Image

class QuickDrawDataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path, transform = None):
            super().__init__()
            self.path = path
            self.transform = transform
            self.classes = ['airplane', 'apple', 'angel', 'ant', 'anvil', 'axe', 'backpack',
                            'banana', 'bandage', 'baseball', 'basketball', 'bear',
                            'bed', 'bear', 'bee', 'bicycle', 'binoculars', 'bird',
                            'book', 'boomerang', 'bowtie', 'brain', 'bus', 'butterfly',
                            'cactus', 'camel', 'candle', 'cannon', 'car', 'castle', 'cat', 'cello', 'chair',
                            'clarinet', 'computer', 'cow', 'crab', 'crocodile', 'crown', 
                            'diamond', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'clock', 
                            'drums', 'duck', 'elephant', 'eye', 'fish', 'flamingo', 'flower', 'frog',
                            'giraffe', 'guitar', 'fish']
        
    def __len__(self):
        return len(os.listdir(self.path))
    
    def __getitem__(self, index):
        img_name = os.listdir(self.path)[index]
        label = self.classes.index(img_name.split("_")[0])       
        img_path = os.path.join(self.path, img_name)
        
        
        image = cv2.imread(img_path,0)
        
        image = np.repeat(image[..., np.newaxis], 3, -1)
        image = image.transpose(2,0,1)


        
        
        if self.transform is not None:
            image = self.transform(image)
        
        
        return image, label