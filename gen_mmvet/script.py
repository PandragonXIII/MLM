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
# plt.rc("font",size=10)
FONTSIZE=8

models = ["InstructBLIP","llava","minigpt4","qwen"]
model_names = ["InstructBLIP","LLaVA-v1.5-7B","MiniGPT4","Qwen-VL"]
xtik = ["rec","ocr","know","gen","spat","math","total"]
fig = plt.figure(figsize=(7,4.2))
for i in range(1,5):
    plt.subplot(2,2,i)
    for k in data.keys():
        if models[i-1] in k and "no" in k:
            origin_scores = data[k]
        if models[i-1] in k and "with" in k:
            defence_scores = data[k]
    total_width, n = 0.8,2
    width = total_width/n
    x = np.arange(len(origin_scores))
    x1 = x-width/2
    x2 = x1+width
    plt.title(model_names[i-1],fontsize=FONTSIZE)
    if i!=2 and i!=4:
        plt.ylabel("Score",fontsize=FONTSIZE)
    plt.bar(x1,origin_scores,width=width,label="Base",color="#295f7a")
    plt.bar(x2,defence_scores,width=width,label="CIDER",color="#629b8b")
    plt.ylim((0,63))
    plt.yticks(fontsize=FONTSIZE)
    plt.xticks(x,xtik,rotation=20,fontsize=FONTSIZE)
    # plt.legend(loc="lower left",ncol=2)

    for a,b in zip(x1,origin_scores):
        plt.text(a+0.05,b+0.3,"%.1f"%b,ha="center",va="bottom",rotation=80,fontsize=7)
    for a,b in zip(x2,defence_scores):
        plt.text(a+0.05,b+0.3,"%.1f"%b,ha="center",va="bottom",rotation=80,fontsize=7)
    if i==1:
        # plt.legend(bbox_to_anchor=(),ncol=2)
        pass
plt.suptitle("",fontsize=FONTSIZE,y=0.92)
plt.tight_layout()

lines,labels = fig.axes[-1].get_legend_handles_labels()
plt.legend(lines,labels,bbox_to_anchor=(-0.12,2.85),fontsize=FONTSIZE,loc="upper center",ncol=2)

plt.savefig("./performance_bar.eps",bbox_inches='tight')
plt.savefig("./performance_bar2.jpg",bbox_inches='tight')