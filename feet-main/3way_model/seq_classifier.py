import torch 
import torch.utils.data as Dataset 
import os

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

        self.classes = ['fracture', 'edema', 'healthy' ]
        self.image_paths = []
        self.labels = []
        self.image_names = []

        for image_name in os.listdir(root_dir):
            pass


def main():
    pass


if __name__ == '__main__':
    main()
