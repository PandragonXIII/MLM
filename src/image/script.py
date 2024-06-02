import os
from PIL import Image

filenames = os.listdir("./valset")
files = []
for fn in filenames:
    img = Image.open(f"./valset/{fn}")
    img = img.convert("RGB")
    files.append(img)
    os.remove(f"./valset/{fn}")
    img.save(f"./valset/{fn.split('.')[0]}.jpg")

