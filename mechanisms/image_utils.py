from PIL import Image
from PIL.Image import Exif
import os, gc, random, sys, json, random, time
from shared.running_config import get_config

def save_images(target_dir, subfolder, images):
    target = os.path.join(target_dir, subfolder)
    if not os.path.exists(target):
        os.makedirs(target, exist_ok=True)
    
    for i, img in enumerate(images):
        iname = str(i) + '.png'
        img.save(os.path.join(target, iname), exif=img._exif.tobytes())
        
def in_memory_encode_exif(img, description):
    img._exif = Exif()
    
    tags = get_config("exif_tags")
    if tags:
        for x in tags:
            img._exif[x] = description
    
    img._exif._loaded = False
    img.info["exif"] = img.getexif()
    return img