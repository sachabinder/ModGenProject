from PIL import Image
from glob import glob
import torch


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, return_path=False):
        self.root = root
        self.images = []
        self.return_path = return_path
        if root[-3:] == "txt":
            f = open(root, "r")
            lines = f.readlines()
            for line in lines:
                self.images.append(line.strip())
        else:
            self.images = sorted(glob(root + "/**/*.png", recursive=True))
        self.transform = transform

    def __getitem__(self, index):
        try:
            img = Image.open(self.images[index]).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            if self.return_path == False:
                return img
            else:
                return img, self.images[index]
        except Exception:
            print("bad image {}".format(self.images[index]))
            return self.__getitem__(0)

    def __len__(self):
        return len(self.images)
