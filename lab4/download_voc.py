"""Confirms the PASCAL VOC 2012 segmentation dataset is in place under ./data.

The dataset was already downloaded and extracted in the Pre-Lab, so the download
line below is commented out to avoid downloading it again.
"""
from torchvision.datasets import VOCSegmentation
# from torchvision.datasets.utils import download_and_extract_archive

DATA = './data'

# download_and_extract_archive(
#     'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
#     DATA, md5='6cd6e144f989b92b3379bac3b3de84fd')

train = VOCSegmentation(DATA, year='2012', image_set='train', download=False)
val = VOCSegmentation(DATA, year='2012', image_set='val', download=False)
print(f'train: {len(train)} val: {len(val)}')
