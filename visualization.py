"""
read cosine similarity data from `src/analysis/*.csv` and visualize it
"""

import os
import pandas as pd
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib import ticker



def clustered_text_errorbar(df:pd.DataFrame, ax:matplotlib.axes.Axes):
    '''
    input: 
        df: cosine similarity dataframe 
            rows: text
            cols: image+"is_malicious"
        ax: matplotlib axis
    output: plot
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

def single_type_text_line(df:pd.DataFrame, ax:matplotlib.axes.Axes):
    '''
    input: 
        df: cosine similarity dataframe 
            rows: text(**only one type**)
            cols: image
        ax: matplotlib axis
    output: plot
    '''
    # calculate mean
    df = df.groupby("is_malicious").mean()
    # use undenoised img name as legend
    line_name = "_".join(df.columns[0].split("_")[:-2])
    c = df.columns[:]
    xticks = [] # use denoise times as xticks
    for x in c:
        xticks.append("_".join(x.split("_")[-2:]))
    # plot line
    ax.plot(xticks, df.iloc[0,:],label=line_name)
    ax.xaxis.set_tick_params(rotation=30)
    ax.set_xlabel("image")
    ax.set_ylabel("cosine similarity")
    ax.legend()

f = "E:\\research\\MLLM\\src\\analysis\\similarity_matrix.csv"
df = pd.read_csv(f)
# print(df.columns)

# plot malicious text vs all images in one plot
malicious = df[df["is_malicious"]==1]
fig,ax = plt.subplots(1,1, figsize=(20,10))
for i in range(4):
    single_type_text_line(malicious.iloc[:,[j for j in range(i*8,i*8+8)]+[-1]], ax)
plt.tight_layout()
plt.savefig(f"./src/results/malicious_text_line.png")
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