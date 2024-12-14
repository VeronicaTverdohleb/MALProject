from pycocotools.coco import COCO

# List of category IDs for the objects of interest
category_ids = [
    52, 50, 43, 66, 53, 55, 63, 60, 63, 57, 55, 63, 62, 59, 64, 66, 69, 72, 74, 67, 72, 85, 86, 35, 40, 39, 37, 15, 34
]

# Load COCO annotations
annotation_file = 'coco/annotations/annotations/instances_train2017.json'
coco = COCO(annotation_file)

# Get the IDs of images that contain the objects of interest
image_ids = coco.getImgIds(catIds=category_ids)

# Get the relevant annotations for these images
annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_ids, catIds=category_ids))

# Filter images and annotations for these objects
filtered_image_ids = set([annotation['image_id'] for annotation in annotations])

import requests
import os
import zipfile

# List of image IDs for objects of interest (from your filtered categories)
# You can limit the number of image IDs you want to process
subset_image_ids = list(filtered_image_ids)  # Use the previously filtered image IDs

# Function to download and extract image files
def download_images(image_ids, target_dir, zip_url_template='http://images.cocodataset.org/zips/'):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    for image_id in image_ids:
        image_file_name = f"{image_id:012d}.jpg"  # COCO image file naming
        image_url = zip_url_template + image_file_name
        img_data = requests.get(image_url).content
        with open(f"{target_dir}/{image_file_name}", 'wb') as f:
            f.write(img_data)
        print(f"Downloaded: {image_file_name}")

# Specify the directory to save images
image_dir = 'coco/images_subset'
download_images(subset_image_ids[:100], image_dir)  # Download the first 100 images, for example


##################################################################################################
# Define the dataset - Create a custom dataset class that loads images and annotations, applies the required 
# transformations, and prepared them for training

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Define the transformation for images (resizing and converting to tensor)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CocoDetectionDataset(Dataset):
    def __init__(self, coco, image_ids, transform=None):
        self.coco = coco
        self.image_ids = image_ids
        self.transform = transform
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = f'coco/images_subset/{img_info["file_name"]}'
        
        # Load image
        image = Image.open(img_path)
        
        # Load annotations
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        boxes = []
        labels = []
        for ann in annotations:
            boxes.append(ann['bbox'])  # COCO bounding box format [xmin, ymin, width, height]
            labels.append(ann['category_id'])
        
        # Convert to tensor
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        
        target = {"boxes": boxes_tensor, "labels": labels_tensor}
        
        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)
        
        return image, target

##############################################################################################
# Create the data loader
from torch.utils.data import DataLoader

# Convert image_ids to a list (if you haven’t already)
subset_image_ids_list = list(subset_image_ids)

# Create dataset and DataLoader
dataset = CocoDetectionDataset(coco, subset_image_ids_list[:10], transform=transform)  # Limit to first 10 images for example
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)



