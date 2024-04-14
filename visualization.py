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
    \nSave the visualization of the cosine similarity distribution
    '''
    # compress malicious(first 40 rows) and benign(last 40 rows) text to one row each
    df = df.groupby("is_malicious").mean()
    # TODO: calculate error
    # plot errorbar
    ax.errorbar(df.columns[1:], df.iloc[0,1:], yerr=0.0, fmt='o', label="benign")
    ax.errorbar(df.columns[1:], df.iloc[1,1:], yerr=0.0, fmt='o', label="malicious")
    ax.xaxis.set_tick_params(rotation=30)
    ax.set_xlabel("image")
    ax.set_ylabel("cosine similarity")
    ax.legend()
    # 加入另一坐标轴画出两者差值相对值
    ax2 = ax.twinx()
    ax2.plot(df.columns[1:], (df.iloc[0,1:]-df.iloc[1,1:]), 'r', label="relative difference")
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=df.iat[0,1]-df.iat[1,1]))
    ax2.set_ylabel("relative difference")
    return

f = "E:\\research\\MLLM\\src\\analysis\\similarity_matrix.csv"
df = pd.read_csv(f)
# print(df.columns)

# group images by different constrained value(16,32,64, unconstrained)
c16 = df.iloc[:,[i for i in range(0,28,4)]+[-1]]
c32 = df.iloc[:,[i for i in range(1,28,4)]+[-1]]
c64 = df.iloc[:,[i for i in range(2,28,4)]+[-1]]
cunc = df.iloc[:,[i for i in range(3,28,4)]+[-1]]
clean = df.iloc[:,[i for i in range(28,35)]+[-1]]

# plot in different subplots
fig, axs = plt.subplots(5,1, figsize=(20,20))
clustered_text_errorbar(clean, axs[0])
clustered_text_errorbar(c16, axs[1])
clustered_text_errorbar(c32, axs[2])
clustered_text_errorbar(c64, axs[3])
clustered_text_errorbar(cunc, axs[4])
plt.tight_layout()
plt.savefig(f"./src/results/clustered_text_errorbar.png")
# plt.show()