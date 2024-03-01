import h5py, os, argparse
import numpy as np
from PIL import Image
# from macenko import normalizeStaining


parser = argparse.ArgumentParser(description='Convert H5 file to png')
parser.add_argument('--src', default='./', type=str, help='path to source')
parser.add_argument('--dest', default='./', type=str, help='path to destination')
args = parser.parse_args()

path= args.src  #'/tsi/clusterhome/ipp-6635/wsi_patcher/tumor/patches'
out = args.dest #'/tsi/clusterhome/ipp-6635/wsi_patcher/train/tumor/'
for h5_case in os.listdir(path):
    try:
        file = h5py.File(path+'/'+h5_case, 'r')
        out_case = out+'/'+h5_case[:-3]
        os.makedirs(out_case,exist_ok=True)
        size = len(file['coords'])
        for i in range(size):
            try:
                img = Image.fromarray(file['imgs'][i].astype('uint8'), 'RGB')
                img.save(out_case+'/'+str(file['coords'][i][0])+'_'+str(file['coords'][i][1])+'.png', "png")
            except:
                print(i)
    except:
        print(h5_case)
