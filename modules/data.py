from torchvision import transforms, datasets, utils, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
from PIL import Image
import numpy as np
import torch
import os
import shutil
import random
from general_arguments import genargs


#Move test Images to another directory
src_dir = "/content/img_align_celeba_new"
dst_dir = "/content/test"

# Create destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

all_images = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

testset_size = 40551

selected_images = random.sample(all_images, testset_size)
for img in selected_images:
    shutil.move(os.path.join(src_dir, img), os.path.join(dst_dir, img))


# Change the attribute file accordingly
moved_images = os.listdir('/content/test')
# Load the original attribute file
attr_df = pd.read_csv('/content/list_attr_celeba.csv')

# Filter out the moved images
filtered_df = attr_df[attr_df.iloc[:, 0].isin(moved_images)]
attr_df.drop(filtered_df.index, inplace=True)

print(len(filtered_df), len(attr_df))
# Save the new CSV
attr_df.to_csv('/content/list_attr_celeba_train.csv', index=False)
filtered_df.to_csv('/content/list_attr_celeba_test.csv', index=False)

print(f"Moved {len(selected_images)} images to {dst_dir}")
print(f"Number of training samples: {len(os.listdir(src_dir))}")
print(f"Number of test samples: {len(os.listdir(dst_dir))}")


class build_dataset(Dataset):
    def __init__(self, root_dir, attr_path, transforms, drop_features = True):
        self.attr_frame = pd.read_csv(attr_path)
        self.attr_frame[self.attr_frame == -1] = 0
        self.root_dir = root_dir
        self.transforms = transforms
        self.dropped_attributes = None
        if drop_features:
            counts = self.attr_frame.iloc[:, 1:].sum(axis=0)/len(self.attr_frame)
            self.dropped_attributes = list(counts.sort_values(ascending=False).index[30:])
            self.attr_frame = self.attr_frame.drop(self.dropped_attributes, axis=1)

    def __len__(self):
        return len(self.attr_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.attr_frame.iloc[idx, 0])
        image = Image.open(img_name)
        #attr = torch.tensor(self.attr_frame.iloc[idx, 1:]).type(torch.FloatTensor)
        attr = torch.tensor(self.attr_frame.iloc[idx, 1:]).type(torch.LongTensor)
        image = self.transforms(image)
        #print(f'Dataset created with {self.__len__()} images')
        return image, attr

    def train_test_split(self, split_index):
        train_idx, val_idx = list(range(0, split_index)), list(range(split_index, self.__len__()))
        return Subset(self, train_idx), Subset(self, val_idx)
    
TRAIN_DIR = "/content/img_align_celeba_new"
TEST_DIR = "/content/test"
ATTR_FILE_TRAIN = "/content/list_attr_celeba_train.csv"
ATTR_FILE_TEST = "/content/list_attr_celeba_test.csv"


train_transforms = transforms.Compose([
                        transforms.Resize(genargs.img_dim),
                        transforms.CenterCrop(genargs.img_dim),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize(([0.5,0.5,0.5]),([0.5,0.5,0.5]))
                        ])

test_transforms = transforms.Compose([
                        transforms.Resize(genargs.img_dim),
                        transforms.CenterCrop(genargs.img_dim),
                        transforms.ToTensor(),
                        transforms.Normalize(([0.5,0.5,0.5]),([0.5,0.5,0.5]))
                        ])

#dataset = build_dataset(ROOT_DIR, ATTR_FILE_PATH, transformations)
trainset = build_dataset(TRAIN_DIR, ATTR_FILE_TRAIN, train_transforms)

if trainset.dropped_attributes is not None:
    test_attributes = pd.read_csv(ATTR_FILE_TEST)
    try:
        test_attributes.drop(trainset.dropped_attributes, axis=1, inplace=True)
        test_attributes.to_csv('/content/list_attr_celeba_test.csv', index=False)
    except:
        pass

testset = build_dataset(TEST_DIR, ATTR_FILE_TEST, test_transforms, drop_features=False)
# Assume `dataset` is your PyTorch Dataset object
small_testset_index = 128*200
indices = list(range(0, small_testset_index))
small_testset = Subset(testset, indices)

#trainset, testset = dataset.train_test_split((int(0.8*len(dataset)//genargs.batch_size))*genargs.batch_size) #202496

print(len(trainset), len(testset), 'Total', len(trainset)+len(testset))
train_loader = DataLoader(trainset, batch_size=genargs.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(small_testset, batch_size=genargs.batch_size, shuffle=False, drop_last=True)
print('Dataloader lengths', len(train_loader), len(test_loader))