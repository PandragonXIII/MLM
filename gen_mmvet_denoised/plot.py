import json, os,csv
import numpy as np

data = dict()
with open("./statics_raw.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        dname =  " ".join(row[0].split("_")[:3])
        if dname not in data.keys():
            data[dname]=[[float(temp) for temp in row[1:8]]]
        else:
            data[dname].append([float(temp) for temp in row[1:8]])
# print(data)
for k,v in data.items():
    data[k] = np.mean(v,axis=0)
# print(data)

import matplotlib.pyplot as plt 
FONTSIZE=8

models = ["blip","llava","minigpt4","qwen"]
model_names = ["InstructBLIP","LLaVA-v1.5-7B","MiniGPT4","Qwen-VL"]
xtik = ["rec","ocr","know","gen","spat","math","total"]
# fig = plt.figure(figsize=(7.3,4.2))
fig = plt.figure(figsize=(7.3,3))
for i in range(1,5):
    plt.subplot(2,2,i)
    for k in data.keys():
        if models[i-1] in k and "no " in k:
            origin_scores = data[k]
        if models[i-1] in k and "with defence" in k:
            defence_scores = data[k]
        if models[i-1] in k and "with denoise" in k:
            denoise_scores = data[k]
    total_width, n = 0.85,3
    width = total_width/n
    x = np.arange(len(origin_scores))
    x1 = x-width/2
    x2 = x1+width
    x3 = x2+width
    plt.title(model_names[i-1],fontsize=FONTSIZE)
    if i!=2 and i!=4:
        plt.ylabel("Score",fontsize=FONTSIZE)
    plt.bar(x1,origin_scores,width=width,label="Base",color="#fd763f")
    plt.bar(x2,denoise_scores,width=width,label="CIDER-de",color="#91dce0")
    plt.bar(x3,defence_scores,width=width,label="CIDER",color="#23bac5")
    plt.ylim((0,62))
    plt.yticks(fontsize=FONTSIZE)
    plt.xticks(x,xtik,fontsize=FONTSIZE)
    # plt.legend(loc="lower left",ncol=2)
    LABEL_FONT = 5
    ROTATE=40
    for a,b in zip(x1,origin_scores):
        plt.text(a+0.05,b+0.3,"%.1f"%b,ha="center",va="bottom",rotation=ROTATE,fontsize=LABEL_FONT)
    for a,b in zip(x2,denoise_scores):
        plt.text(a+0.05,b+0.3,"%.1f"%b,ha="center",va="bottom",rotation=ROTATE,fontsize=LABEL_FONT)
    for a,b in zip(x3,defence_scores):
        plt.text(a+0.05,b+0.3,"%.1f"%b,ha="center",va="bottom",rotation=ROTATE,fontsize=LABEL_FONT)

    if i==1:
        # plt.legend(bbox_to_anchor=(),ncol=2)
        pass
plt.suptitle("",fontsize=FONTSIZE,y=0.92)
plt.tight_layout()

lines,labels = fig.axes[-1].get_legend_handles_labels()
plt.legend(lines,labels,bbox_to_anchor=(-0.08,3.2),fontsize=FONTSIZE,loc="upper center",ncol=3)

plt.savefig("./performance_bar_denoised.pdf",bbox_inches='tight')
plt.savefig("./performance_bar_denoised.jpg",bbox_inches='tight')