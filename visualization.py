"""
read cosine similarity data from `src/intermediate-data/*.csv` and visualize it
"""

import os
import pandas as pd
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns



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


def all_line_with_error(df:pd.DataFrame, ax:matplotlib.axes.Axes):
    '''
    input: 
        df: cosine similarity dataframe 
            rows: text(**only one type**)
            cols: images(from different origin image)
        ax: matplotlib axis
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
    # plot line
    # ax.plot(xticks, df = df.groupby("is_malicious").mean().iloc[0,:],label=line_name)
    sns.lineplot(data=df, ax=ax, dashes=False, markers=True, legend="full")

    ax.xaxis.set_tick_params(rotation=30)
    ax.set_xlabel("image")
    ax.set_ylabel("cosine similarity")
    ax.legend()

if __name__ == "__main__":
    f = "MLM/src/intermediate-data/similarity_matrix_test.csv"
    df = pd.read_csv(f)
    # print(df.columns)
    IMAGE_NUM = 6
    MAX_DENOISE_TIMES = 350 #(0,550,50)
    STEP = 50
    CHECKPOINT_NUM = MAX_DENOISE_TIMES//STEP +1
    PLOT_X_NUM = 8

    # plot malicious text vs all images in one plot
    data = df[df["is_malicious"]==1]
    fig,ax = plt.subplots(1,1, figsize=(20,10))
    for i in range(IMAGE_NUM):
        single_type_text_line(data.iloc[:,[j for j in range(i*CHECKPOINT_NUM,i*CHECKPOINT_NUM+PLOT_X_NUM)]+[-1]], ax)
    plt.tight_layout()
    plt.savefig(f"MLM/src/results/malicious_text_TestSetImg_line.png")
    # plt.show()


# # group images by different constrained value(16,32,64, unconstrained)
# c16 = df.iloc[:,[i for i in range(0,8)]+[-1]]
# c32 = df.iloc[:,[i for i in range(8,16)]+[-1]]
# c64 = df.iloc[:,[i for i in range(16,24)]+[-1]]
# cunc = df.iloc[:,[i for i in range(24,32)]+[-1]]
# # clean = df.iloc[:,[i for i in range(28,35)]+[-1]]
# dfs = [c16,c32,c64,cunc]

# # plot in different subplots
# fig, axs = plt.subplots(len(dfs),1, figsize=(20,20))
# # clustered_text_errorbar(clean, axs[0])
# for i,df in enumerate(dfs):
#     clustered_text_errorbar(df, axs[i])
# plt.tight_layout()
# plt.savefig(f"./src/results/clustered_text_errorbar.png")
# plt.show()