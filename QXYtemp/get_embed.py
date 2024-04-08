from transformers import AutoProcessor, AutoModelForPreTraining, AutoTokenizer, AutoImageProcessor
import torch
from PIL import Image
import os, csv

MODEL_PATH = "/data1/qxy/models/llava-1.5-7b-hf"
DEVICE = "cuda:0"

model = AutoModelForPreTraining.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to(DEVICE)# this runs out of memory


# processor = AutoProcessor.from_pretrained(MODEL_PATH, cache_dir="./cache")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
imgprocessor = AutoImageProcessor.from_pretrained(MODEL_PATH)



def get_img_embedding(image_path):
    image = Image.open(image_path)
    # img embedding
    pixel_value = imgprocessor(image, return_tensors="pt").pixel_values.to(DEVICE)
    image_outputs = model.vision_tower(pixel_value, output_hidden_states=True)
    selected_image_feature = image_outputs.hidden_states[model.config.vision_feature_layer]
    selected_image_feature = selected_image_feature[:, 1:] # by default
    image_features = model.multi_modal_projector(selected_image_feature)
    # calculate average to compress the 2th dimension
    image_features = torch.mean(image_features, dim=1)
    return image_features

def get_text_embedding(text: str):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    input_embeds = model.get_input_embeddings()(input_ids)
    # calculate average to get shape[1, 4096]
    input_embeds = torch.mean(input_embeds, dim=1)
    return input_embeds

###################
source_dir = "/data1/qxy/src"
text_malicious_file = "harmbench_behaviors_text_val.csv"
text_benign_file = "first_lines_of_MMLU.csv"
img_save_filename = "image_embeddings_val.pt"
text1_save_filename = "harmbench_embeddings_val.pt"
text2_save_filename = "benign_embeddings_val.pt"
###################

# generate embeddings for images
img_embed_list = []
for img in os.listdir(f"{source_dir}/image"):
    ret = get_img_embedding(f"{source_dir}/image/{img}")
    img_embed_list.append(ret)
# img_embed_list.append(get_img_embedding(f"{source_dir}/image/prompt_constrained_16.bmp"))

# generate embeddings for text
malicious_text_embed_list = []
with open(f"{source_dir}/text/{text_malicious_file}", "r") as f:
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
        text = row[0]
        ret = get_text_embedding(text)
        malicious_text_embed_list.append(ret)
# binign
benign_text_embed_list = []
with open(f"{source_dir}/text/{text_benign_file}", "r") as f:
    reader = csv.reader(f)
    cnt=0
    for row in reader:
        if cnt>=40:
            break
        cnt+=1
        text = row[0]+"\tA.{}\tB.{}\tC.{}\tD.{}".format(row[1],row[2],row[3],row[4])
        ret = get_text_embedding(text)
        benign_text_embed_list.append(ret)

# save embeddings
torch.save(img_embed_list, f"{source_dir}/embedding/{img_save_filename}")
torch.save(malicious_text_embed_list, f"{source_dir}/embedding/{text1_save_filename}")
torch.save(benign_text_embed_list, f"{source_dir}/embedding/{text2_save_filename}")

# calculate cosine similarity
import numpy as np
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

# import matplotlib.pyplot as plt
# plt.subplot(2,2,(1,3))
# plt.errorbar(["malicious", "benign"], [avg1, avg2], [std1, std2], fmt='o')
# plt.subplot(2,2,2)
# plt.hist(malicious_result, bins=50, alpha=0.5, label='malicious')
# plt.subplot(2,2,4)
# plt.hist(benign_result, bins=50, alpha=0.5, label='benign')
# plt.show()