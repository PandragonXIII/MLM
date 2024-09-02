import os
from PIL import Image

imgdir = "./images"
for fn in os.listdir(imgdir):
    idx = int(fn.strip("image_.png"))
    newname = "image_{:0>3d}.jpg".format(idx)
    img = Image.open(f"{imgdir}/{fn}")
    img.convert("RGB")
    img.save(f"./jpgimgs/{newname}")
