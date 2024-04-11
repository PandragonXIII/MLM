import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import joypy
import skimage

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
    """ convert image under the path dir to array and save them as one .npz file, 
     the image is arr_0 and label(388) is arr_1 """
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

def cos_sim_distribution(path:str):
    '''input: cosine similarity csv file 
            rows: text
            cols: image
        output: None
        save the visualization of the cosine similarity distribution
    '''
    # read the csv file
    df = pd.read_csv(path)
    df.drop(["text"],inplace=True, axis=1)
    print(f"shape: {df.shape}")

    pic_types = df.columns

    # plot the distribution, x-axis: cossim, y-axis: frequency
    nplot = df.shape[1]
    # Draw Plot
    fig, axes = joypy.joyplot(df,
                            colormap=plt.cm.get_cmap("Spectral", nplot),
                            figsize=(10, 6))
    # plt.show()
    plt.savefig(".\\src\\results\\cos_sim_of_mix_text.png")

    # create a new dataframe with columns: adversarial, benign, pic_type
    # and the value is the cosine similarity, with the corresponding type
    df2 = pd.DataFrame(columns=["adversarial", "benign", "pic_type"])
    for i in range(df.shape[1]):
        adversarial = df.iloc[:40,i]
        benign = df.iloc[40:,i].reset_index(drop=True)
        # get a 40 row , 2 col dataframe
        temp = pd.DataFrame({"adversarial":adversarial, "benign":benign, "pic_type":[pic_types[i]]*40})
        df2 = pd.concat([df2, temp])
    # Draw Plot
    plt.figure(figsize=(16,10), dpi= 80)
    fig, axes = joypy.joyplot(df2,
                            by="pic_type",
                            column=['adversarial', 'benign'],
                            colormap=plt.cm.get_cmap("Spectral", nplot),
                            figsize=(10, 6))
    
    # plt.show()
    plt.savefig(".\\src\\results\\cos_sim_of_sep_text.png")
    return

def add_gaussian_noise(fpath:str, savepath:str, var=0.01, times=1):
    '''add gaussian noise to the image, 
    '''
    img = skimage.io.imread(fpath)
    for _ in range(times):
        img = skimage.util.random_noise(img, mode='gaussian', var=var)
    img = skimage.util.img_as_ubyte(img)
    skimage.io.imsave(savepath, img)
    return

def denoise(fpath:str, savepath:str,w=0.1):
    """filter with sklearn Total variation filter"""
    img = skimage.io.imread(fpath)
    denoised = skimage.restoration.denoise_tv_chambolle(img, weight=w)
    denoised = skimage.util.img_as_ubyte(denoised)
    skimage.io.imsave(savepath, denoised)
    return

def robustness_test(fpath:str, save:str, var=0.01, times=1):
    '''add gaussian noise to the image, and filter it with TV filter
    '''
    add_gaussian_noise(fpath, save, var, times)
    denoise(save, save)
    return

def batch_process(func, dir:str, save_dir=None, **kwargs):
    '''batch process the images in the dir with the function
    save in a sub dir "processed" by default
    '''
    if save_dir==None:
        save_dir = dir + "\\processed"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for image in os.listdir(dir):
        if image.endswith((".jpeg",".bmp")):
            save_path = save_dir + "\\" + image
            func(dir + "\\" + image, save_path, **kwargs)
    return

if __name__ == "__main__":
    # img_to_array(".\\src\\image")
    # show_sampled_pic("src\image\prompt_constrained_16.npz")
    # show_sampled_pic("src\samples\samples_1x256x256x3_pandas16.npz")
    # array_to_img("src\samples\samples_4x256x256x3.npz")
    # cos_sim_distribution(".\\src\\analysis\\cosine_similarity.csv")
    # #############
    # noise = ".\\src\\image\\clean.jpeg"
    # plt.imshow(skimage.io.imread(noise))
    # for j in range(1,8,3):
    #     new_path = f".\\src\\image\\clean_noise\\{j}.bmp"
    #     add_gaussian_noise(noise,new_path, var=0.01,times=j)
    #     plt.imshow(skimage.io.imread(new_path))
    #     plt.show()
    #     # denoise the noised image
    #     denoised_path = f".\\src\\image\\clean_noise\\{j}_filtered.bmp"
    #     denoise(new_path, denoised_path)
    #     plt.imshow(skimage.io.imread(denoised_path))
    #     plt.show()

    # img_to_array("./src/image/clean_noise")
    # ##############
    cos_sim_distribution("./src/analysis/image+noise+filter_embeddings_valcosine-similarity.csv")