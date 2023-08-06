from PIL import Image
import os, gc, random, sys, json, random, time

def save_images(target_dir, subfolder, images, prompts):
    target = os.path.join(target_dir, subfolder)
    if not os.path.exists(target):
        os.makedirs(target, exist_ok=True)
        
    for i, img in enumerate(images):
        iname = str(i) + "_" + prompts[i][:70] + '.png'
        img.save(os.path.join(target, iname))