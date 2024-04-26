# """
# read all files in the path(but not dirs) or the specific file,
# denoise them for multiple iterations and save them to save_path
# """
# from DRM import *
# import os 
# import matplotlib.pyplot as plt
# import torch
# import numpy as np
# import tqdm


# # print(torch.cuda.is_available())
# DEVICE = "cuda:0"
# model = DiffusionRobustModel()

# path = "MLM/src/image/adversarial/prompt_constrained_inf2.bmp"
# save_path = "MLM/src/image/denoised"
# batch = []
# names = []
# if os.path.isdir(path): # denoise all image under given dir
#     dir1 = os.listdir(path)
#     dir1.sort()
#     for fn in dir1:
#         if not os.path.isfile(path+"/"+fn):
#             continue
#         img = np.array(plt.imread(path+"/"+fn)/255*2-1).astype(np.float32)
#         batch.append(img)
#         names.append(os.path.splitext(fn)[0])
# elif os.path.isfile(path): # denoise specific image
#     img = np.array(plt.imread(path)/255*2-1).astype(np.float32)
#     batch.append(img)
#     names.append(os.path.splitext(os.path.basename(path))[0])
# else:
#     print("Invalid path")
#     exit()
# batch = torch.tensor(batch).to(DEVICE).permute(0,3,1,2)

# # set denoise iteration time
# iterations = range(0,450,50)
# for it in tqdm.tqdm(iterations):
#     if it!=0:
#         denoise_batch = np.array(model.denoise(batch, it).to("cpu"))
#     else:
#         denoise_batch = np.array(batch.to("cpu"))
#     denoise_batch = denoise_batch.transpose(0,2,3,1)
#     denoise_batch = (denoise_batch+1)/2
#     # print(denoise_batch.shape)
#     # print(denoise_batch.max())
#     # print(denoise_batch.min())

#     for i in range(batch.shape[0]):
#         plt.imsave("{save_path}/{name}_denoised_{:0>3d}times.bmp".format(
#             it, save_path=save_path, name=names[i]
#         ), denoise_batch[i])
