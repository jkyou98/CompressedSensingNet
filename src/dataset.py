import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import os
from PIL import Image
class BSDS(Dataset):
    def __init__(self, rootDirImg):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
        self.rootDirImg = rootDirImg
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.listData = sorted([
            f for f in os.listdir(rootDirImg)
            if f.lower().endswith(valid_exts)
        ])
        
    def __getitem__(self, i):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        # input images
        inputName = self.listData[i]
        # process the images
        transf = transforms.Compose([transforms.Resize((64, 64)),transforms.ToTensor(),transforms.Normalize((0.4507, 0.4481, 0.3688),(0.2375, 0.2221, 0.2297))])
        inputImage = transf(Image.open(self.rootDirImg + inputName).convert('RGB'))
        return inputImage
    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.listData)
    
class BSDS_val(Dataset):
    def __init__(self, rootDirImg):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
        self.rootDirImg = rootDirImg
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.listData = sorted([
            f for f in os.listdir(rootDirImg)
            if f.lower().endswith(valid_exts)
        ])
        
    def __getitem__(self, i):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        # input images
        inputName = self.listData[i]
        # process the images
        transf = transforms.Compose([transforms.Resize((64, 64)),transforms.ToTensor(),transforms.Normalize((0.4612, 0.4608, 0.4009),(0.2511, 0.2495, 0.2706))])
        inputImage = transf(Image.open(self.rootDirImg + inputName).convert('RGB'))
        return inputImage
    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.listData)