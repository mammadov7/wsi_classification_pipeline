import torch
import argparse
import pandas as pd
from PIL import Image
import json, os, sys, glob
from pathlib import Path
from omegaconf import OmegaConf
from solo.methods import METHODS
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms.functional as VF
from timm.models.vision_transformer import _create_vision_transformer

class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)
        sample = {'input': img}
        
        if self.transform:
            sample = self.transform(sample)
        return sample 


class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        img=transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
        return {'input': img} 
    

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def bag_dataset(csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=256, shuffle=False, num_workers=5, drop_last=False)
    return dataloader, len(transformed_dataset)

device = "cuda:0"
parser = argparse.ArgumentParser(description='Convert H5 file to png')
parser.add_argument('--src', default='./', type=str, help='path to source data')
parser.add_argument('--dest', default='./', type=str, help='path to destination')
parser.add_argument('--model', default='./', type=str, help='path to model')

args = parser.parse_args()

if args.model=='resnet18':
    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc=nn.Identity()
    model=resnet18
elif args.model=='vit_small':
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, num_classes=0, img_size=256, pretrained=True)
    model = _create_vision_transformer("vit_small_patch16_224", **model_kwargs)

elif args.model=='vit_tiny':
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, num_classes=0, img_size=256, pretrained=True)
    model = _create_vision_transformer("vit_tiny_patch16_224", **model_kwargs)

else:
    ckpt_dir = Path(args.model)
    args_path = ckpt_dir / "args.json"
    ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][0]

    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)

    cfg = OmegaConf.create(method_args)

    model = (
        METHODS[method_args["method"]]
        .load_from_checkpoint(ckpt_path, strict=False, cfg=cfg)
        .backbone
    )
# move model to the gpu
model = model.to(device)
model.eval()

bags_path = os.path.join(args.src,'*','*','*')
bags_list = glob.glob(bags_path)
n_classes = glob.glob(os.path.join(args.src,'train','*'+os.path.sep))
save_path = os.path.join(args.dest)
num_bags=len(bags_list)
# prepare data
Tensor = torch.FloatTensor
for i in range(0, num_bags):
    feats_list = []
    csv_file_path = glob.glob(os.path.join(bags_list[i], '*.png'))
    dataloader, bag_size = bag_dataset(csv_file_path)
    print(bags_list[i])
    with torch.no_grad():
        for iteration, batch in enumerate(dataloader):
            patches = batch['input'].float().cuda() 
            feats = model(patches)
            feats_list.extend(feats.cpu().numpy())
            sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))
    if len(feats_list) == 0:
        print('No valid patch extracted from: ' + bags_list[i])
    else:
        df = pd.DataFrame(feats_list)
        os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-3], bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
        df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-3], bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')
    
