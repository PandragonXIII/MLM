"""
get embedding tensor of given text and image, through llava model
Also calculate the cosine similarity between text and image 
and save the result as csv(rows: text, cols: image+"is_malicious")

This runs on the server with GPU
"""
from transformers import AutoProcessor, AutoModelForPreTraining, AutoTokenizer, AutoImageProcessor
import torch
from PIL import Image
import os, csv
import numpy as np



def compute_cosine(a_vec , b_vec):
    """calculate cosine similarity"""
    a_vec = a_vec.to("cpu").detach().numpy()
    b_vec = b_vec.to("cpu").detach().numpy()
    norms1 = np.linalg.norm(a_vec, axis=1)
    norms2 = np.linalg.norm(b_vec, axis=1)
    dot_products = np.sum(a_vec * b_vec, axis=1)
    cos_similarities = dot_products / (norms1 * norms2) # ndarray with size=1
    return cos_similarities[0]

def get_similarity_matrix(save_tensors=False):
    """get the similarity matrix with each text and image embeddings combination
        is used while developing"""
    MODEL_PATH = "/data1/qxy/models/llava-1.5-7b-hf"
    DEVICE = "cuda:0"
    source_dir = "/data1/qxy/MLM/src"
    image_dir = "/data1/qxy/MLM/src/image/denoised"
    cosine_filename = "similarity_matrix_validation.csv"
    text_malicious_file = "harmbench_behaviors_text_val.csv"
    text_benign_file = "first_lines_of_MMLU.csv"
    img_save_filename = "image+noise+filter_embeddings_val.pt"
    text1_save_filename = "harmbench_embeddings_val.pt"
    text2_save_filename = "benign_embeddings_val.pt"
    ########################
    model = AutoModelForPreTraining.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to(DEVICE)# this runs out of memory

    # processor = AutoProcessor.from_pretrained(MODEL_PATH, cache_dir="./cache")
    # TODO maybe we can remove these later
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    imgprocessor = AutoImageProcessor.from_pretrained(MODEL_PATH)



    def get_img_embedding(image_path):
        # model.to(DEVICE)
        image = Image.open(image_path)
        # img embedding
        pixel_value = imgprocessor(image, return_tensors="pt").pixel_values.to(DEVICE)
        image_outputs = model.vision_tower(pixel_value, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[model.config.vision_feature_layer]
        selected_image_feature = selected_image_feature[:, 1:] # by default
        image_features = model.multi_modal_projector(selected_image_feature)
        # calculate average to compress the 2th dimension
        image_features = torch.mean(image_features, dim=1).detach().to("cpu")
        del pixel_value, image_outputs, selected_image_feature
        # model.to("cpu")
        torch.cuda.empty_cache()
        return image_features

    def get_text_embedding(text: str):
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
        input_embeds = model.get_input_embeddings()(input_ids)
        # calculate average to get shape[1, 4096]
        input_embeds = torch.mean(input_embeds, dim=1).detach().to("cpu")
        del input_ids
        torch.cuda.empty_cache()
        return input_embeds


    # generate embeddings for images
    img_embed_list = []
    img_names = []
    dir1 = os.listdir(image_dir)
    dir1.sort()
    for img in dir1:
        ret = get_img_embedding(f"{image_dir}/{img}")
        img_embed_list.append(ret)
        img_names.append(os.path.splitext(img)[0])
    # img_embed_list.append(get_img_embedding(f"{source_dir}/image/prompt_constrained_16.bmp"))
    


    # generate embeddings for malicious text
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

    # generate embeddings for benign text
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
    if save_tensors:
        torch.save(img_embed_list, f"{source_dir}/embedding/{img_save_filename}")
        torch.save(malicious_text_embed_list, f"{source_dir}/embedding/{text1_save_filename}")
        torch.save(benign_text_embed_list, f"{source_dir}/embedding/{text2_save_filename}")



    # compute cosine similarity between malicious text and images
    malicious_result = np.zeros((len(malicious_text_embed_list), len(img_embed_list)))
    for i in range(len(malicious_text_embed_list)):
        for j in range(len(img_embed_list)):
            text_embed = malicious_text_embed_list[i]
            img_embed = img_embed_list[j]
            cos_sim = compute_cosine(img_embed, text_embed)
            malicious_result[i, j] = cos_sim
    # add one column "is_malicious" with value 1
    malicious_result = np.concatenate((malicious_result, np.ones((malicious_result.shape[0],1))), axis=1)
    # compute cosine similarity between benign text and images
    benign_result = np.zeros((len(benign_text_embed_list), len(img_embed_list)))
    for i in range(len(benign_text_embed_list)):
        for j in range(len(img_embed_list)):
            text_embed = benign_text_embed_list[i]
            img_embed = img_embed_list[j]
            cos_sim = compute_cosine(img_embed, text_embed)
            benign_result[i, j] = cos_sim
    # add one column "is_malicious" with value 0
    benign_result = np.concatenate((benign_result, np.zeros((benign_result.shape[0],1))), axis=1)

    img_names.append("is_malicious")

    tot = np.concatenate((malicious_result, benign_result), axis=0)
    print(tot.shape)
    # save the full similarity matrix as csv
    np.savetxt(f"{source_dir}/analysis/{cosine_filename}", tot, delimiter=",",
                header=",".join(img_names))
    print(f"csv file saved at: {source_dir}/analysis/{cosine_filename}")

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

if __name__=="__main__":
    get_similarity_matrix(save_tensors=True)


