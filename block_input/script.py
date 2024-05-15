import os

clean_img_path = "/home/xuyue/QXYtemp/MLM/src/image/clean_test.bmp"
save_dir = "/home/xuyue/QXYtemp/MLM/block_input/img"
for i in range(100):
    os.system(
        f"cp {clean_img_path} {save_dir}/celan_test_{i}.bmp")