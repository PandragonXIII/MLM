import os
import pandas as pd
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import numpy as np
import joypy.joyplot
import matplotlib.cm as cm
import csv

#calculate <query,clean>-<query,adv> for malicious queries and 
# 5 clean img, 4(all) adv img

clean_fp = "/home/xuyue/QXYtemp/MLM/src/intermediate-data/10clean_similarity_matrix_val.csv"
adv_fp = "/home/xuyue/QXYtemp/MLM/src/intermediate-data/4clean_similarity_matrix_val.csv"

cleandf = pd.read_csv(clean_fp)
cleandf = cleandf[cleandf["is_malicious"]==1].drop(["is_malicious"],axis=1)
cleandf = cleandf[[col for col in cleandf.columns if "denoised_000times" in col]]

smallest_colname = cleandf.sum(axis=0).sort_values()[:5]
smallest_colname = smallest_colname.index
clean_cols = cleandf[smallest_colname]

advdf = pd.read_csv(adv_fp)
advdf = advdf[advdf["is_malicious"]==1].drop(["is_malicious"],axis=1)
advdf = advdf[[col for col in advdf.columns if "prompt_constrained_" in col and "_000times" in col]]
adv_cols = advdf


# calculate difference
cleandata = clean_cols.sum(axis=1)
cleandata = cleandata.mul(0.2)
advdata = adv_cols.sum(axis=1)
advdata = advdata.mul(0.25)

diff = cleandata-advdata
# diff.mul(0.05)
diff = pd.DataFrame(diff,columns=["Difference of Cosine Similarity"])


plt.clf()
f,ax = plt.subplots(figsize=(4.3,2))
sns.kdeplot(diff,x="Difference of Cosine Similarity",color = "#eeca40",ax=ax,fill=True,alpha=0.5)
# plt.legend(["clean","adversarial"])

plt.tight_layout()
figpath = f"./src/results/deltacossim_hist_each_text"
plt.savefig(figpath+".jpg")
plt.savefig(figpath+".pdf")