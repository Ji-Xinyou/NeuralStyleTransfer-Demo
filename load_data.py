from PIL import Image
from Macros import *
import os

def load_images():
    img_dir = os.getcwd() + '/Images/'
    img_dirs = [img_dir, img_dir]
    img_names = ['content{}.png'.format(CONTENT_IDX), 'style{}.png'.format(STYLE_IDX)]
    imgs = [Image.open(img_dirs[i] + name) for i,name in enumerate(img_names)]
    return imgs