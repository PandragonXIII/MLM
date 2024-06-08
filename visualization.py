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
if __name__ == "__main__":
    mmvet_double_histogram()
    # cossim_line_of_adv_vs_clean_image(
    # "/home/xuyue/QXYtemp/MLM/src/intermediate-data/similarity_matrix_validation.csv")
    # delta_cos_sim_distribution("./src/intermediate-data/10clean_similarity_matrix_val.csv",it=350, picname="10clean_denoised250_delta_cossim_distribution.png")
    # delta_cos_sim_distribution("./src/intermediate-data/10clean_similarity_matrix_val.csv",it=300, picname="10clean_denoised250_delta_cossim_distribution.png")
    # train_data_decline_line()

    # cossim_line_of_all_image("./src/intermediate-data/similarity_matrix_validation.csv", cpnum=8)
