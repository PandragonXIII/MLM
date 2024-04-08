import torch, os, csv
import numpy as np
import pandas as pd

###################
source_dir = "./src"
img_save_filename = "image_+denoised_embeddings_val.pt"
text1_save_filename = "harmbench_embeddings_val.pt"
text2_save_filename = "benign_embeddings_val.pt"
###################

# read the embedded files
img_embed_list = torch.load(f"{source_dir}/embedding/{img_save_filename}")
malicious_text_embed_list = torch.load(f"{source_dir}/embedding/{text1_save_filename}")
benign_text_embed_list = torch.load(f"{source_dir}/embedding/{text2_save_filename}")

# calculate cosine similarity
def compute_cosine(a_vec , b_vec):
    a_vec = a_vec.to("cpu").detach().numpy()
    b_vec = b_vec.to("cpu").detach().numpy()
    norms1 = np.linalg.norm(a_vec, axis=1)
    norms2 = np.linalg.norm(b_vec, axis=1)
    dot_products = np.sum(a_vec * b_vec, axis=1)
    cos_similarities = dot_products / (norms1 * norms2) # ndarray with size=1
    return cos_similarities[0]

# compute cosine similarity between malicious text and images
malicious_result = np.zeros((len(img_embed_list), len(malicious_text_embed_list)))
for i in range(len(img_embed_list)):
    for j in range(len(malicious_text_embed_list)):
        img_embed = img_embed_list[i]
        text_embed = malicious_text_embed_list[j]
        cos_sim = compute_cosine(img_embed, text_embed)
        malicious_result[i, j] = cos_sim
# compute cosine similarity between benign text and images
benign_result = np.zeros((len(img_embed_list), len(benign_text_embed_list)))
for i in range(len(img_embed_list)):
    for j in range(len(benign_text_embed_list)):
        img_embed = img_embed_list[i]
        text_embed = benign_text_embed_list[j]
        cos_sim = compute_cosine(img_embed, text_embed)
        benign_result[i, j] = cos_sim

pics = []
for image in os.listdir(f"{source_dir}/image"):
    pics.append(image.split(".")[0])

def save_csv():
    '''save text and corresponding cosine similarity'''    
    # get malicious text
    m_text = []
    with open(f"{source_dir}/text/harmbench_behaviors_text_val.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        # get the first 40 rows
        cnt=0
        for row in reader:
            if cnt>=40:
                break
            # only keep FunctionalCategory=standard rows
            if row[1]!="standard":
                continue
            cnt+=1
            m_text.append(row[0])

    # get benign text
    b_text = []
    with open(f"{source_dir}/text/first_line.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        cnt=0
        for row in reader:
            if cnt>=40: # get the first 40 rows
                break
            cnt+=1
            text = row[0]+"\tA.{}\tB.{}\tC.{}\tD.{}".format(row[1],row[2],row[3],row[4])
            b_text.append(text)


    result_dict = {
        "text": m_text + b_text,
        "clean": malicious_result[0].tolist() + benign_result[0].tolist(),
        "adv-16": malicious_result[1].tolist() + benign_result[1].tolist(),
        "adv-32": malicious_result[2].tolist() + benign_result[2].tolist(),
        "adv-64": malicious_result[3].tolist() + benign_result[3].tolist(),
        "adv-inf": malicious_result[4].tolist() + benign_result[4].tolist(),
    }
    if malicious_result.shape[1]>5:
        result_dict["clean-denoised"] = malicious_result[5].tolist() + benign_result[5].tolist()
        result_dict["adv-16-denoised"] = malicious_result[6].tolist() + benign_result[6].tolist()
        result_dict["adv-32-denoised"] = malicious_result[7].tolist() + benign_result[7].tolist()
        result_dict["adv-64-denoised"] = malicious_result[8].tolist() + benign_result[8].tolist()
        result_dict["adv-inf-denoised"] = malicious_result[9].tolist() + benign_result[9].tolist()
    # save the full result in csv
    result = pd.DataFrame(result_dict)
    result.to_csv(f"{source_dir}/analysis/cosine-similarity.csv", index=False)

save_csv()

# analysis

import matplotlib.pyplot as plt
from matplotlib import ticker


''' # 直方图展示两类text与img-16的cosine similarity; 以及均值和标准差
# plt.subplot(2,2,(1,3))
# plt.errorbar(["malicious", "benign"], [avg1, avg2], [std1, std2], fmt='o')
# plt.subplot(2,2,2)
# plt.hist(malicious_result[0], rwidth=0.8, bins=40, alpha=0.5, label='malicious', range=(0,0.12))
# plt.title("malicious")
# plt.subplot(2,2,4)
# plt.hist(benign_result[0], rwidth=0.8, bins=40, alpha=0.5, label='benign', range=(0,0.12))
# plt.title("benign")
# plt.tight_layout()
# # plt.show()
# # store the figure
# plt.savefig(f"{source_dir}/analysis2.png")
'''

# 热力图效果不好，同一个问题对应不同图片差距较小

"""# 两类text分别与所有img的cosine similarity均值和标准差
malicious_avg = np.mean(malicious_result, axis=1)
mailcious_std = np.std(malicious_result, axis=1)
benign_avg = np.mean(benign_result, axis=1)
benign_std = np.std(benign_result, axis=1)
print("malicious_avg:{}\tmailcious_std:{}\nbenign_avg:{}\tbenign_std{}".format(
    malicious_avg, mailcious_std, benign_avg, benign_std
))
"""

"""# 对同一图片的结果求平均画图(分为malicious & benign两类)

# plt.errorbar(pics, malicious_avg, mailcious_std, fmt='o', label='malicious', alpha=0.5)
# plt.errorbar(pics, benign_avg, benign_std, fmt='x', label='benign', alpha=0.5)
# plt.legend(loc = 'upper left')
# plt.xticks(rotation=45)

# # 加入另一坐标轴画出两者差值相对值
# clean_diff = benign_avg[0]-malicious_avg[0]
# plt.twinx()
# plt.plot(pics, benign_avg-malicious_avg, 'r', label='benign-malicious', alpha=0.5)
# plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=clean_diff))
# plt.legend()
# plt.xticks(rotation=30)
# plt.title("cosine similarity between image and text")

# plt.tight_layout()
# plt.savefig(f"{source_dir}/different-pic-result.png")
"""