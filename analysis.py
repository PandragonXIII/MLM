import torch, os, csv
import numpy as np

###################
source_dir = "./src"
text_malicious_file = "harmbench_behaviors_text_val.csv"
text_benign_file = ""
img_save_filename = "image_embeddings_val.pt"
text1_save_filename = "harmbench_embeddings_val.pt"
text2_save_filename = "benign_embeddings_val.pt"
###################

# read the embedded files
img_embed_list = torch.load(f"{source_dir}/embedding/{img_save_filename}.pt")
malicious_text_embed_list = torch.load(f"{source_dir}/embedding/{text1_save_filename}.pt")
benign_text_embed_list = torch.load(f"{source_dir}/embedding/{text2_save_filename}.pt")

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

# analysis
avg1 = np.mean(malicious_result.flatten())
std1 = np.std(malicious_result.flatten())
avg2 = np.mean(benign_result.flatten())
std2 = np.std(benign_result.flatten())

print("malicious avg:{}\tmalicious std:{}\nbenign avg:{}\tbenign std{}".format(
    avg1,std1,avg2,std2
))

import matplotlib.pyplot as plt

plt.subplot(2,2,(1,3))
plt.errorbar(["malicious", "benign"], [avg1, avg2], [std1, std2], fmt='o')
plt.subplot(2,2,2)
plt.hist(malicious_result[0], rwidth=0.8, bins=40, alpha=0.5, label='malicious')
plt.title("malicious")
plt.subplot(2,2,4)
plt.hist(benign_result[0], rwidth=0.8, bins=40, alpha=0.5, label='benign')
plt.title("benign")
# plt.show()
# store the figure
plt.savefig(f"{source_dir}/analysis.png")