"""
read cosine similarity data from `src/intermediate-data/*.csv` and visualize it
"""

import os
import pandas as pd
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import numpy as np
import joypy.joyplot
import matplotlib.cm as cm
<<<<<<< Updated upstream
=======
import csv
import time
>>>>>>> Stashed changes



def clustered_text_errorbar(df:pd.DataFrame, ax:matplotlib.axes.Axes):
    '''
    input: 
        df: cosine similarity dataframe 
            rows: text
            cols: image+"is_malicious"
        ax: matplotlib axis
    '''
    # compress malicious(first 40 rows) and benign(last 40 rows) text to one row each
    df = df.groupby("is_malicious").mean()
    # TODO: calculate error
    # plot errorbar
    ax.errorbar(df.columns[:], df.iloc[0,:], yerr=0.0, fmt='o', label="benign")
    ax.errorbar(df.columns[:], df.iloc[1,:], yerr=0.0, fmt='o', label="malicious")
    ax.xaxis.set_tick_params(rotation=30)
    ax.set_xlabel("image")
    ax.set_ylabel("cosine similarity")
    ax.legend()
    # 加入另一坐标轴画出两者差值相对值
    # ax2 = ax.twinx()
    # ax2.plot(df.columns[:], (df.iloc[0,:]-df.iloc[1,:]), 'r', label="relative difference")
    # ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=df.iat[0,1]-df.iat[1,1]))
    # ax2.set_ylabel("relative difference")
    return

def single_type_text_line(df:pd.DataFrame, ax:matplotlib.axes.Axes, errorbar=None):
    '''
    input: 
        df: cosine similarity dataframe 
            rows: text(**only one type**)
            cols: images
        ax: matplotlib axis
        errorbar: may use ('se') or None or other
    output: plot
    '''
    # drop is_malicious column
    df = df.drop(columns=["is_malicious"])
    # use undenoised img name as legend
    line_name = "_".join(df.columns[0].split("_")[:-2])
    c = df.columns[:]
    xticks = [] # use denoise times as xticks
    for x in c:
        xticks.append("_".join(x.split("_")[-2:]))
    # change df column names to match other images result
    df.columns = xticks
    # plot line
    # ax.plot(xticks, df = df.groupby("is_malicious").mean().iloc[0,:],label=line_name)
    sns.lineplot(data=df.melt(var_name="denoise_times",value_name="cos_sim"), ax=ax,
                 x="denoise_times",y="cos_sim",label=line_name, dashes=False, markers=True, errorbar=errorbar)

    ax.xaxis.set_tick_params(rotation=30)
    ax.set_xlabel("denoise times")
    ax.set_ylabel("cosine similarity")
    ax.legend()
    ax.set_title("malicious text vs denoised images(standard error)")


def train_data_decline_line():
    """
    plot the decline value of cosine similarity between
     origin clean image vs malicious text(40) """
    f = "./src/intermediate-data/similarity_matrix_validation.csv"
    df = pd.read_csv(f)
    df = df[df["is_malicious"]==1]
    clean_heads = [col for col in df.columns if "clean_resized" in col]
    clean_heads = clean_heads[:8]
    raw_data = df[clean_heads]
    ret=[]
    for r in range(raw_data.shape[0]):
                row = raw_data.iloc[r]
                ret.extend(np.array(row[1:]) - row[0])
    ret = pd.DataFrame(ret, columns=["Δ cosine similarity"])

    # plot with joyplot
    fig, axes = joypy.joyplot(ret, figsize=(10, 6), 
                              color="#CFDAEC", linecolor="#BFBBBA", overlap=2)

    # draw a line on 90% percentile
    for ax in axes:
        ax.axvline(ret.quantile(0.99).values, color="#BFBBBA", linestyle='--')

    plt.savefig("./src/results/train_data_decline_distribution.png")
    return

def cos_sim_distribution(path:str):
    '''input: cosine similarity csv file 
            rows: text
            cols: image
        output: None
        save the visualization(joyplot-mountain) of the cosine similarity distribution
        grouping by image type, on 250 iterations of denoise
        only consider malicious text
    '''
    # read the csv file
    df = pd.read_csv(path)
    df = df[df["is_malicious"]==1]
    col_names = [col for col in df.columns if "250" in col and "clean_test" not in col]
    data = df[col_names]
    # rename columns and throw the postfix "denoised_250times"
    data.columns = ["_".join(col.split("_")[:-2]) for col in col_names]
    # data = data.melt(var_name="pic_type", value_name="cos_sim")
    print(f"shape: {data.shape}")
    print(data.head())

    # plot the distribution, x-axis: cossim, y-axis: frequency
    # Draw Plot
    data["is_malicious"] = 1 # add a common column to group by(draw in one axis)
    # set alpha=0.6 to show the overlapping
    colors = ["#1399B2","#BD0026","#FD8D3C","#F1D756","#EF767B"]
    fig, axes = joypy.joyplot(data, by="is_malicious", column=list(data.columns),
                              alpha=0.3, color=colors, legend=True, loc="upper left"
                              )
    plt.tight_layout()
    # plt.show()
    plt.savefig("./src/results/denoise250_cossim_distribution.png")
    return

def delta_cos_sim_distribution(path:str, it=250, picname="denoised250_delta_cossim_distribution.png"):
    '''input: cosine similarity csv file 
            rows: text
            cols: image
        output: None
        visualize (joyplot-mountain) the **difference** of cosine similarity distribution
        grouping by image type, on it(default=250) iterations of denoise
        only consider malicious text
    '''
    # read the csv file
    df = pd.read_csv(path)
    df = df[df["is_malicious"]==1]
    df.rename(columns={"# clean_resized_denoised_000times":"clean_resized_denoised_000times"},inplace=True)
    # use the new unconstrained adv image
    col_names =        [col for col in df.columns if f"{it}" in col and "clean_test" not in col and "_inf_" not in col]
    origin_col_names = [col for col in df.columns if "000" in col and "clean_test" not in col and "_inf_" not in col]
    # denoised250 cosine similarity
    data = df[col_names]
    data.columns = ["_".join(col.split("_")[:-2]) for col in col_names]
    # origin cosine similarity
    origin_data = df[origin_col_names]
    origin_data.columns = ["_".join(col.split("_")[:-2]) for col in origin_col_names]
    # calcualte delta value
    data = data-origin_data

    # rename columns and throw the postfix "denoised_250times"
    data.columns = ["_".join(col.split("_")[:-2]) for col in col_names]
    # data = data.melt(var_name="pic_type", value_name="cos_sim")
    

    # plot the distribution, x-axis: cossim, y-axis: frequency
    # Draw Plot
    data["var"] = "Δ cosine similarity" # add a common column to group by(draw in one axis)
    # set alpha=0.6 to show the overlapping
    colors = ["#1399B2","#BD0026","#FD8D3C","#F1D756","#EF767B"]
    fig, axes = joypy.joyplot(data, by="var",
                              alpha=0.6, color=colors, legend=True, loc="upper left", linecolor="#01010100"
                              )
    plt.title(f"delta value of cosine similarity after {it} iters denoise")
    plt.tight_layout()
    # plt.show()
    picname = picname.replace("250", str(it))
    plt.savefig(f"./src/results/{picname}")
    return

def cossim_line_of_all_image(path:str, cpnum=8):
    """
    plot the line plot of cosine similarity of image after denoising.
    given the path of the similarity matrix csv file, plot the line plot of each image type.
    """
    df = pd.read_csv(path)
    df = df[df["is_malicious"]==1]
    # delete is_malicious
    df = df.drop(columns=["is_malicious"])
    # get all image classes
    img_classes = set(["_".join(col.split("_")[:-2])+"_" for col in df.columns])
    img_classes = sorted(list(img_classes))
    #
    fig, ax = plt.subplots(1,1)
    xticks = [i for i in range(0,cpnum*50,50)]
    for img_class in img_classes:
        cols = [col for col in df.columns if img_class in col]
        assert len(cols) >= cpnum # check if there are enough columns
        cols = cols[:cpnum] # and only use the first cpnum columns
        data = df[cols]
        ax.plot(xticks, data.mean(), label=img_class)
    ax.legend(bbox_to_anchor=(0.05, 1.05), loc='lower left', borderaxespad=0.)
    ax.set_xlabel("denoise times")
    ax.set_ylabel("cosine similarity")
    plt.tight_layout()
    plt.savefig(f"./src/results/cossim_line.png")

def cossim_line_of_adv_vs_clean_image(path:str, cpnum=8):
    """
    plot the line plot of cosine similarity of image after denoising.
    given the path of the similarity matrix csv file, plot the line plot of each image type.
    """
    df = pd.read_csv(path)
    df = df[df["is_malicious"]==1]
    # delete is_malicious
    df = df.drop(columns=["is_malicious"])
    collist = []
    for col in df.columns:
         if "prompt_constrained_inf_" in col:
            collist.append(col)
    df = df.drop(columns=collist)
    # get all image classes
    img_classes = set(["_".join(col.split("_")[:-2])+"_" for col in df.columns])
    img_classes = sorted(list(img_classes))
    #
    fig, ax = plt.subplots(1,1)
    xticks = [i for i in range(0,cpnum*50,50)]
    clean_means = []
    adv_means = []
    for img_class in img_classes:
        cols = [col for col in df.columns if img_class in col]
        assert len(cols) >= cpnum # check if there are enough columns
        cols = cols[:cpnum] # and only use the first cpnum columns
        data = df[cols].mean().tolist()
        if "clean" in img_class:
            clean_means.append(data)
        else:
            adv_means.append(data)
    clean_means = np.mean(clean_means,axis=0)
    adv_means = np.mean(adv_means,axis=0)
    ax.plot(xticks, clean_means, label="clean")
    ax.plot(xticks, adv_means, label="adversarial")

    ax.legend(bbox_to_anchor=(0.05, 1.05), loc='lower left', borderaxespad=0.)
    ax.set_xlabel("denoise times")
    ax.set_ylabel("cosine similarity")
    plt.tight_layout()
    plt.savefig(f"./src/results/cossim_line_grouped.png")

def cossim_line_grouped(clean_path:str,adv_path:str, cpnum=8):
    """
    plot the line plot of cosine similarity of image after denoising.
    given the path of the similarity matrix csv file, plot the line plot of each image type.
    """
    dfadv = pd.read_csv(adv_path)
    dfadv = dfadv[dfadv["is_malicious"]==1]
    dfadv = dfadv.drop(columns=["is_malicious"])
    advdata = dfadv[[col for col in dfadv.columns if "prompt_constrained" in col and "prompt_constrained_inf_" not in col]]
    advvar = advdata.var(axis=0)
    advvar = advvar.to_numpy().reshape((4,-1)).mean(axis=0).T[:8]
    advmean = advdata.mean(axis=0)
    advmean = advmean.to_numpy().reshape((4,-1)).mean(axis=0).T[:8]
    advr1 = list(map(lambda x:x[0]-x[1],zip(advmean,advvar)))
    advr2 = list(map(lambda x:x[0]+x[1],zip(advmean,advvar)))


    df = pd.read_csv(clean_path)
    df = df[df["is_malicious"]==1]
    # delete is_malicious
    df = df.drop(columns=["is_malicious"])
    cleanvar = df.var(axis=0)
    cleanvar = cleanvar.to_numpy().reshape((-1,cpnum)).mean(axis=0)
    cleandata = df.mean(axis=0)
    print(cleandata)
    cleanmean = cleandata.to_numpy().reshape((-1,cpnum)).mean(axis=0)
    cleanr1 = list(map(lambda x:x[0]-x[1],zip(cleanmean,cleanvar)))
    cleanr2 = list(map(lambda x:x[0]+x[1],zip(cleanmean,cleanvar)))

    xticks = [i*50 for i in range(0,cpnum)]
    plt.figure(figsize=(4.2,2))
    plt.plot(xticks, cleanmean, label="clean",color="#23bac5")
    plt.plot(xticks, advmean, label="adversarial",color="#fd763f")

    # variance
    plt.fill_between(xticks,advr1,advr2,color="#fd763f",alpha=0.2,edgecolor=None)
    plt.fill_between(xticks,cleanr1,cleanr2,color="#23bac5",alpha=0.2,edgecolor=None)
    # plt.legend(bbox_to_anchor=(0.05, 1.05), loc='lower left', borderaxespad=0.,ncol=2)
    plt.legend(ncol=2)
    plt.xlabel("Denoise Times")
    plt.ylabel("Cosine Similarity")
    plt.tight_layout()
    plt.savefig(f"./src/results/cossim_line_grouped(new).png")
    plt.savefig(f"./src/results/cossim_line_grouped(new).pdf")

def mmvet_double_histogram():
    data = { #model:origin,defence
        "blip":(19.37,12.67),
        "llava":(30.97,19.13),
        "minigpt4":(21.43,12.97),
        "qwen":(46.23,29.23)
    }
    origin_scores = []
    defence_scores = []
    for v in data.values():
        origin_scores.append(v[0])
        defence_scores.append(v[1])
    plt.figure()
    total_width, n = 0.8,2
    width = total_width/n
    x = np.arange(len(data))
    x1 = x-width/2
    x2 = x1+width
    plt.title("MM-Vet score")
    plt.ylabel("Score")
    plt.bar(x1,origin_scores,width=width,label="Base",color="#295f7a")
    plt.bar(x2,defence_scores,width=width,label="CIDer",color="#629b8b")
    plt.xticks(x,data.keys())
    plt.legend()
    plt.savefig("./src/results/mmvet_score_bar.png")

def grouped_cossim_hist(denoise_times=0):
    cleanfp = "/home/xuyue/QXYtemp/MLM/src/intermediate-data/10clean_similarity_matrix_val.csv"
    cleandf = pd.read_csv(cleanfp)
    cleandf = cleandf[cleandf["is_malicious"]==1].drop(["is_malicious"],axis=1)

    wanted_name = "_{:0>3d}times".format(denoise_times)
    wanted_cols = [col for col in cleandf.columns if wanted_name in col]
    cleandata = cleandf[wanted_cols]
    # flatten to 1d
    cleandata = pd.DataFrame(cleandata.to_numpy().reshape(-1))
    cleandata.columns = ["Cosine Similarity"]
    cleandata["img_type"]="clean"
    print(f"there are {cleandata.shape} clean data points")

    advfp = "/home/xuyue/QXYtemp/MLM/src/intermediate-data/similarity_matrix_validation.csv"
    advdf = pd.read_csv(advfp)
    advdf = advdf[advdf["is_malicious"]==1].drop(["is_malicious"],axis=1)
    advdf = advdf[[col for col in advdf.columns if "prompt_constrained_" in col and "inf_" not in col]]

    wanted_name = "_{:0>3d}times".format(denoise_times)
    wanted_cols = [col for col in advdf.columns if wanted_name in col]
    advdata = advdf[wanted_cols]
    # flatten to 1d
    advdata = pd.DataFrame(advdata.to_numpy().reshape(-1))
    advdata.columns = ["Cosine Similarity"]
    advdata["img_type"]="adversarial"
    print(f"there are {advdata.shape} adv data points")

    plt.clf()
    data = pd.concat([cleandata,advdata],axis=0,ignore_index=True)
    f,ax = plt.subplots(1,1,figsize=(4.3,2))
    sns.kdeplot(cleandata,x="Cosine Similarity",color = "#23bac5",ax=ax,fill=True,alpha=0.5)
    sns.kdeplot(advdata,x="Cosine Similarity",color = "#fd763f",ax=ax,fill=True,alpha=0.5)
    plt.legend(["clean","adversarial"],loc="upper left")
    plt.tight_layout()

    plt.savefig(f"./src/results/cossim_hist_grouped2.jpg")
    plt.savefig(f"./src/results/cossim_hist_grouped2.pdf")

def grouped_delta_cossim_hist(denoise_times="min"):
    cleanfp = "/home/xuyue/QXYtemp/MLM/src/intermediate-data/4clean_similarity_matrix_val.csv"
    cleandf = pd.read_csv(cleanfp)
    cleandf = cleandf[cleandf["is_malicious"]==1].drop(["is_malicious"],axis=1)
    cleandf = cleandf[[col for col in cleandf.columns if "clean" in col]]

    # wanted_name = "_{:0>3d}times".format(denoise_times)
    # wanted_cols = [col for col in cleandf.columns if wanted_name in col]
    # cleandata = cleandf[wanted_cols]
    cleandata = cleandf.to_numpy().reshape((-1,8))
    cleandata = cleandata.min(axis=1)
    # cleandata = cleandata[:,7]

    cleanbase = cleandf[[col for col in cleandf.columns if "_000times" in col]]
    cleanbase = cleanbase.to_numpy().reshape((-1))
    # flatten to 1d
    cleandata = pd.DataFrame((cleanbase-cleandata).reshape(-1))
    cleandata.columns = ["Δ Cosine Similarity"]
    cleandata["img_type"]="clean"
    print(f"there are {cleandata.shape} clean data points")

    advfp = "/home/xuyue/QXYtemp/MLM/src/intermediate-data/similarity_matrix_validation.csv"
    advdf = pd.read_csv(advfp)
    advdf = advdf[advdf["is_malicious"]==1].drop(["is_malicious"],axis=1)
    advdf = advdf[[col for col in advdf.columns if "prompt_constrained_" in col and "inf_" not in col]]

    # wanted_name = "_{:0>3d}times".format(denoise_times)
    # wanted_cols = [col for col in advdf.columns if wanted_name in col]
    # advdata = advdf[wanted_cols]
    advdata = advdf.to_numpy().reshape((-1,9)).min(axis=1)
    # advdata = advdata[:,7]

    advbase = advdf[[col for col in advdf.columns if "_000times" in col]]
    advbase = advbase.to_numpy().reshape((-1))
    # flatten to 1d
    advdata = pd.DataFrame((advbase-advdata).reshape(-1))
    advdata.columns = ["Δ Cosine Similarity"]
    advdata["img_type"]="adversarial"
    print(f"there are {advdata.shape} adv data points")

    data = pd.concat([cleandata,advdata],axis=0,ignore_index=True)

    plt.clf()
    f,ax = plt.subplots(figsize=(4.2,2))
    sns.kdeplot(cleandata,x="Δ Cosine Similarity",color = "#23bac5",ax=ax,fill=True,alpha=0.5)
    sns.kdeplot(advdata,x="Δ Cosine Similarity",color = "#fd763f",ax=ax,fill=True,alpha=0.5)
    plt.legend(["clean","adversarial"])

    plt.tight_layout()
    plt.savefig(f"./src/results/deltacossim_hist_grouped_{denoise_times}times2.jpg")
    plt.savefig(f"./src/results/deltacossim_hist_grouped_{denoise_times}times2.pdf")

def asr_barplot(data = None, name=None,*,width=0.29,OTHER_FONT=13,w=10,h=2.4,uniquename=True):
    # data = {#(Base,CIDER,Jailguard) with adv image
    #     "LLaVA-v1.5-7B":(0.609375   ,0          ,0.45),
    #     "InstructBlip":(0.096875    ,0.0234375  ,0.16),
    #     "MiniGPT4-7B":  (0.5734375  ,0.0921875  ,0.195),
    #     "Qwen":         (0.071875   ,0.0109375  ,0.055),
    #     "GPT4v":        (0,0,0)
    # }
    if data is None:
        data = {# ["LLaVA-v1.5-7B","InstructBlip","MiniGPT4-7B","Qwen","GPT4v"]
            "attack w/o defense":(0.609375,0.096875,0.5734375,0.071875,0),
            "CIDER":(0,0.0234375,0.0921875,0.0109375,0),
            "Jailguard":(0.45,0.16,0.195,0.055,0.005)
        }
    # data = { # output denoised img on detecting adversarial
    #     "attack w/o defense":     (0.609375,0.096875,0.5734375,0.071875,0),
    #     "CIDER-de": (0.25625,0.21875  ,0.43375  ,0.055,0),
    #     "CIDER":    (0      ,0.0234375,0.0921875,0.0109375,0),
    #     "Jailguard":(0.45   ,0.16     ,0.195,0.055,0.005)
    # }
    xlbl = ["LLaVA-v1.5-7B","InstructBlip","MiniGPT4","Qwen","GPT4V"]
    xlen = len(data["CIDER"])
    # df = pd.DataFrame.from_dict(data=data,orient="index",columns=["Base","CIDER","Jailguard"])
    # print(df)
    x=np.arange(xlen)
    counter=0
    plt.clf()
    plt.rcParams['font.size'] = 8
    f,ax = plt.subplots(layout="constrained",figsize=(w,h))
    if len(data)==3:
        colors=["#fd763f","#23bac5","#eeca40"]
    elif len(data)==4:
        colors = ["#fd763f","#91dce0","#23bac5","#eeca40"]
    else:
        raise RuntimeError("dismatch color and bar num")
    for k,v in data.items():
        offset = width*counter
        rects = ax.bar(x+offset, v, width, label=k,color=colors[counter])
        ax.bar_label(rects,padding=1, fmt=lambda x: f'{(x)*100:0.2f}%')
        counter+=1
    ax.set_ylabel("Attack Success Rate",fontsize=OTHER_FONT)
    ax.set_xticks(x+width,xlbl[:xlen],
                  rotation=0,fontsize=OTHER_FONT)
    ax.legend(fontsize=OTHER_FONT)
    ax.set_yticklabels([f'{x1*100:.0f}%' for x1 in ax.get_yticks()],fontsize=OTHER_FONT) 
    ax.set_ylim([0,0.7])

    plt.tight_layout()
    if name is None:
        name="asr_bar"
    if uniquename:
        t = int(time.time())
        name=f"{name}_{t}"
    plt.savefig(f"./src/results/{name}.jpg")
    plt.savefig(f"./src/results/{name}.pdf")

def multi_mmvet_histo(data,name=None):
    # data = { #model:origin,defence
    #     "blip":[[],[]],
    #     "llava":[],
    #     "minigpt4":,
    #     "qwen":
    # }
    plt.figure()
    f,axs = plt.subplots(2,2,layout="constrained",figsize=(10,4))
    xlbl = ["rec","ocr","know","gen","spat","math","total"]
    OTHER_FONT=13
    for i,key in enumerate(data.keys()):
        ax = axs.flatten()[i]
        total_width, n = 0.8,2
        width = total_width/n
        x = np.arange(len(xlbl))
        x1 = x-width/2
        x2 = x1+width
        if i==0:
            ax.set_title("\n"+key)
        else:
            ax.set_title(key)
        ax.set_ylabel("Score",fontsize=OTHER_FONT)
        rects1 = ax.bar(x1,data[key][0],width=width,label="Base",color="#fd763f")
        ax.bar_label(rects1,padding=1,fontsize=8,rotation=15)
        rects2 = ax.bar(x2,data[key][1],width=width,label="CIDer",color="#23bac5")
        ax.bar_label(rects2,padding=1,fontsize=8,rotation=15)
        ax.set_xticks(x,xlbl,fontsize=OTHER_FONT)
        ax.set_ylim([0,62])
    # plt.title("\n")
    plt.tight_layout()
    plt.legend(ncol=2,bbox_to_anchor=(-0.35,2.73), loc='lower left')
    t = int(time.time())
    if name is None:
        name='mmvet_detailscore_bar'
    plt.savefig(f"./src/results/{name}.png")
    plt.savefig(f"./src/results/{name}.pdf")

from model_tools import get_img_embedding,generate_denoised_img,compute_cosine

def img_embed_distribution():
    """display the difference of embeddings before and after denoise, regarding clean and adversarial images."""
    # img_dirs = {
    #     "clean":"src/image/additinoal-adv/clean",
    #     "adv16":"src/image/additinoal-adv/visual_constrained_eps_16",
    #     "adv64":"src/image/additinoal-adv/visual_constrained_eps_64",
    #     "adv unconatrained":"src/image/additinoal-adv/visual_unconstrained"
    # }
    # for k,path in img_dirs.items():
    #     sp = f"src/image/additinoal-adv/{k}_denoised"
    #     if os.path.exists(sp):
    #         os.system(f"rm -rf {sp}/*")
    #     else:
    #         os.mkdir(sp)
    #     generate_denoised_img(path=path,save_path=sp,cps=7)
    denoised_dirs = ["src/image/additinoal-adv/clean_denoised","src/image/additinoal-adv/adv16_denoised","src/image/additinoal-adv/adv64_denoised","src/image/additinoal-adv/adv-unconatrained_denoised"]
    fignames = ["clean","adv16","adv64","unconatrained"]
    f,axs = plt.subplots(1,1,layout="constrained",figsize=(10,4))
    xlbl = [i for i in range(50,301,50)]
    for i,dir in enumerate(denoised_dirs):
        img_paths = os.listdir(dir)
        img_paths.sort()
        img_paths = [dir+"/"+n for n in img_paths]
        embeddings = get_img_embedding(img_paths)
        temp = []
        assert len(embeddings)%7==0
        for j in range(7,len(embeddings),7):
            temp.append(embeddings[j-7:j])
        embeddings=temp
        group_result = []
        for line in embeddings:
            line_result = []
            for col in line[1:]:
                a,b = line[0][0],col[0]
                cossim = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
                line_result.append(cossim)
            group_result.append(line_result)
        group_result = np.array(group_result)
        meandata = group_result.mean(axis=0)
        stddata = group_result.std(axis=0)
        # ax = axs.flatten()[i]
        ax = axs
        ax.plot(xlbl,meandata,label=fignames[i])
        ax.legend()
        # ax.fill_between(xlbl,meandata-stddata,meandata+stddata,alpha=0.2)
        # ax.set_title(fignames[i])
    name = "image_embedding_distribution_line"
    plt.savefig(f"./src/results/{name}.png")


if __name__ == "__main__":
    img_embed_distribution()
    # grouped_cossim_hist()
    # multi_mmvet_histo({
    #     "InstructBLIP":[[23.5,11.8,17.2,18.8,13.6,11.3,19.4],[14.5,9.4,8.3,8.7,12.3,7.7,12.7]],
    #     "LLaVA-v1.5-7B":[[34.4,23.2,16.8,19.2,26.8,3.8,31.0],[19.7,18.8,6.7,7.3,21.5,2.5,19.1]],
    #     "MiniGPT4":[[25.7,15.2,15.6,16.9,18.1,3.8,21.4],[14.9,11.4,8.7,7.5,14.4,3.8,13.0]],
    #     "Qwen-VL":[[51.9,37.1,41.5,37.9,38.4,19.0,46.2],[29.0,31.5,20.0,17.7,32.2,16.4,29.2]]
    # },"mmvet")
    # asr_barplot(data = {
    #     "attack w/o defense":     (0.609375,0.096875,0.5734375,0.071875,0),
    #     "CIDER-de": (0.25625,0.21875  ,0.43375  ,0.055,0),
    #     "CIDER":    (0      ,0.0234375,0.0921875,0.0109375,0),
    #     "Jailguard":(0.45   ,0.16     ,0.195,0.055,0.005)
    # },name="ciderde_asr_bar",w=11,width=0.23,uniquename=False,OTHER_FONT=11)
    # asr_barplot()
    # grouped_delta_cossim_hist()
    # grouped_cossim_hist()
    # cossim_line_grouped("/home/xuyue/QXYtemp/MLM/src/intermediate-data/10clean_similarity_matrix_val.csv",
    #                     "/home/xuyue/QXYtemp/MLM/src/intermediate-data/similarity_matrix_validation.csv")
    # mmvet_double_histogram()
    # cossim_line_of_adv_vs_clean_image(
    # "/home/xuyue/QXYtemp/MLM/src/intermediate-data/similarity_matrix_validation.csv")
    delta_cos_sim_distribution("./src/intermediate-data/10clean_similarity_matrix_val.csv",it=350, picname="10clean_denoised250_delta_cossim_distribution.png")
    delta_cos_sim_distribution("./src/intermediate-data/10clean_similarity_matrix_val.csv",it=300, picname="10clean_denoised250_delta_cossim_distribution.png")
    # train_data_decline_line()

    # cossim_line_of_all_image("./src/intermediate-data/similarity_matrix_validation.csv", cpnum=8)
