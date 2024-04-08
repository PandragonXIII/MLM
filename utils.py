
import numpy as np
import matplotlib.pyplot as plt
import os

def show_sampled_pic(path):
    # read the npz file and show the sampled images
    # Samples are saved as a large npz file, where arr_0 in the file is a large batch of samples
    # read samples
    samples = []
    with np.load(path) as data:
        samples = data['arr_0']
        if len(data)>1:
            labels = data['arr_1']
        else:
            labels = np.array([])
    print("Samples shape:", samples.shape)
    print("Labels shape:", labels.shape)
    print("labels: ", labels)
    
    # show n samples
    n=1
    for i in range(n):
        plt.imshow(samples[i])
        plt.show()


def img_to_array(path:str):
    # convert image under the path dir to array and save them as one .npz file, 
    # the image is arr_0 and label(388) is arr_1 
    arr = np.array([])
    for image in os.listdir(path):
        if image.endswith((".jpeg")):
            img = plt.imread(path+"\\"+image)
            if arr.size == 0:
                arr = np.array([img])
            else:
                arr = np.append(arr, [img], axis=0)
    print("arr_0 shape:", arr.shape)
    np.savez(path+"/pack.npz", arr, np.array([388]*len(arr)))# which is gaint panda in image net
    return

def array_to_img(path:str):
    # convert array to image and save them as .bmp files
    with np.load(path) as data:
        samples = data['arr_0']
    print("Samples shape:", samples.shape)
    for i in range(len(samples)):
        name = path.split("\\")[-1].split("/")[-1].split(".")[0]
        plt.imsave(f"E:\\research\\MLLM\\src\\image\\denoised\\{name}_{i}.bmp", samples[i])
    return

if __name__ == "__main__":
    # img_to_array(".\\src\\image")
    # show_sampled_pic("src\image\prompt_constrained_16.npz")
    # show_sampled_pic("src\samples\samples_1x256x256x3_pandas16.npz")
    array_to_img("src\samples\samples_4x256x256x3.npz")