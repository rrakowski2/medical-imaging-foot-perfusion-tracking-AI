#!/usr/bin/python


'''
Script/function to predict foot's mask for the foot-tracking app
by Rafal Apr 2025
''' 


# Set compute environment
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import csv
import cv2
#cv2.setNumThreads(0) 
import numpy as np
import torch
print(torch.version.cuda)
print(torch.cuda.get_device_name())
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torchvision import transforms
import torchmetrics
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from torchmetrics import JaccardIndex
from torchmetrics.functional import accuracy
from unet import UNet
from datetime import datetime
import time
from PIL import Image
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"


# Helper classes


# Read-in data
class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.ids  = os.listdir(root_dir + 'images/')
        self.ids = sorted(self.ids, key=lambda x: int(os.path.splitext(x)[0]))
        self.imgs = [os.path.join(root_dir, 'images/', i) for i in self.ids] 
        self.mask = [os.path.join(root_dir, 'masks/', i) for i in self.ids] 
        self.transforms = transforms

    # Get a single item from the data set
    def __getitem__(self, idx):
        image = cv2.imread(self.imgs[idx], 0)
        masks = cv2.imread(self.mask[idx], 0)
        sample = {'image': image, 
                  'mask': masks}
        
        # Apply any transforms
        if self.transforms:
            new = self.transforms(image=sample['image'], mask=sample['mask'])
            sample['image'] = new['image']
            sample['mask'] = new['mask']
        return sample
    
    # Get the length of the data set
    def __len__(self):
        return len(self.ids)


# Construct PyTorch data module
class ISBI_data_module(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = './data/',
                 batch_size: int = 6,
                 #if device == "cuda":
                 #num_workers = torch.cuda.device_count() * 4
                 num_workers: int = 0, 
                 timeout: int = 0,
                 train_samples=None,
                 valid_samples=None
                ): 
        
        super().__init__()

        # Specify where to save the data and how to load 
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.timeout = timeout

        # Specify key information about the dataset
        self.dims = (1, 240, 320)
        self.num_classes = 1

        # Get the number of samples to use in training and validation
        self.train_samples = train_samples
        self.valid_samples = valid_samples

        # Initialise the specified default train and test transforms
        self._set_default_transforms()


    # Get the data sets
    def setup(self, stage=None):
        # Load the datasets :)
        self.train_dataset = SegmentationDataset(self.data_dir + 'dummy_dataset/', transforms=self.train_transform)
        self.test_dataset = SegmentationDataset(self.data_dir + 'dummy_dataset/', transforms=self.test_transform)
        self.val_dataset = SegmentationDataset(self.data_dir + 'dummy_dataset/', transforms=self.train_transform)
        # set the training and validation samplers
        if self.train_samples:
            self.train_sampler = RandomSampler(self.train_dataset, replacement=True, num_samples=self.train_samples * self.batch_size)
        else:
            self.train_sampler = None
        if self.valid_samples:
            self.valid_sampler = RandomSampler(self.val_dataset, replacement=True, num_samples=self.valid_samples * self.batch_size) 
        else:
            self.valid_sampler = None


    # Get the data loaders 
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=self.train_sampler, persistent_workers=True)  
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=self.valid_sampler,  persistent_workers=True)  
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
  
    def _set_default_transforms(self):    

        # For training use elastic transformations and some shifting and rotations
        self.train_transform = A.Compose([
            A.Affine(shear=0.05, p=1.0),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=45, p=1.0),  
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=1.0), 
            # Normalise the image and convert both image and mask to tensors
            A.Normalize(mean=(0), std=(1)),
            A.pytorch.ToTensorV2()
        ])

        # For the test dataconvertion to a tensor (which also rescales to 0 - 1)
        self.test_transform = A.Compose([
            A.PadIfNeeded(240, 320),
            # Normalise the image and convert both image and mask to tensors
            A.Normalize(mean=(0), std=(1)),
            A.pytorch.ToTensorV2()
        ])  


# Construct U-Net class
class LightningUNet(pl.LightningModule):
    def __init__(self, batch_size=2, lr=1e-4):
        super().__init__()
        self.batch_size = batch_size
        self.save_hyperparameters()
        self.model = UNet(num_classes=2,                                   
                          enc_channels=[1, 64, 128, 256, 512, 1024],
                          dec_channels=[1024, 512, 256, 128, 64])
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat.squeeze(), y.squeeze().long())
        self.log('train_loss', loss) 
        preds = torch.argmax(y_hat.squeeze(), dim=1)
        jaccard = JaccardIndex(task="binary").to('cuda')
        iou = jaccard(preds, y)
        self.log('train_iou', iou, prog_bar=True) 
        return loss
    
    def evaluate(self, batch, stage=None):
        x, y = batch['image'], batch['mask']      
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat.squeeze(), y.squeeze().long())
        #loss = F.binary_cross_entropy_with_logits(y_hat.squeeze(), y.squeeze().float())
        preds = torch.argmax(y_hat.squeeze(), dim=1)
        jaccard = JaccardIndex(task="binary").to('cuda') # task='multiclass'
        iou = jaccard(preds, y)
        #acc = accuracy(preds, y, task="binary")  # task='multiclass'
        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_iou', iou, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    # Set up the optimisers
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)        
        scheduler_dict = {
            'scheduler': ReduceLROnPlateau(optimizer, patience=0),
            'monitor': 'val_loss',  
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}


# Main function
def prediction(img):

    # Set the model
    model = LightningUNet.load_from_checkpoint("../checkpoints/best_model_run150.ckpt", map_location=device) 
        
    # Define transformation for preprocessing the image
    transform = transforms.Compose([
        # Convert the image to a tensor
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5], std=[0.5])])

    # Add batch dimension: 1 x C x H x W
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE) 
    input_image = transform(img).unsqueeze(0)  

    # Move the image to GPU if available
    input_image = input_image.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Set model to evaluation mode
    model.eval()  

    # Forward pass through the model
    with torch.no_grad():
        # Prediction for current single image
        prediction = model(input_image)  

    # Post-process prediction - remove batch dimension
    predicted_mask = torch.argmax(prediction.squeeze(), dim=0).cpu().detach().numpy()
    
    return predicted_mask
    

# 'Main' to execute as standalone
if __name__ == '__main__':
    
    # Test execution time
    start_time = datetime.now() 

    # Load images
    images_folder_path = f'./data/'      # <- data folder (placeholder) containing input images/frames
    ids  = os.listdir(images_folder_path)                                      
    ids = sorted(ids, key=lambda x: int(os.path.splitext(x)[0]))
    images = [os.path.join(images_folder_path, i) for i in ids] 
    # Image frame counter
    no = 1

    # Loop through the images
    for img in images:
        mask = prediction(img)
        print(no)
        duration = datetime.now() - start_time
        print('Processing_rate', no/duration.total_seconds(), '[fps]')
        no += 1
        
