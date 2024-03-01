import torch,glob,tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
from macenko import normalizeStaining
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(np.array(img))
                img=Image.fromarray(img.astype('uint8'), 'RGB')
                # save the transformed image back
                img.save(img_path)
        except:
            print(img_path)
            return 0
        return 0

# example usage


cases=glob.glob('/data/ipp-6635/melanoma/valid/*')
for case in cases:
    img_paths = glob.glob(case +'/*')
    dataset = CustomDataset(img_paths, transform=normalizeStaining)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=20, shuffle=False)
    try: 
        for batch in tqdm.tqdm(dataloader):
        # use the transformed images in the batch
            pass

    except:
        print(case)

cases=glob.glob('/data/ipp-6635/melanoma/train/*/*')
for case in cases:
    img_paths = glob.glob(case +'/*')
    dataset = CustomDataset(img_paths, transform=normalizeStaining)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=20, shuffle=False)
    try: 
        for batch in tqdm.tqdm(dataloader):
        # use the transformed images in the batch
            pass

    except:
        print(case)

    