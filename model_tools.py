from transformers import AutoModelForPreTraining, AutoTokenizer, AutoImageProcessor
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

class Args:
    MODEL_PATH = "/data1/qxy/models/llava-1.5-7b-hf"
    DEVICE = "cuda:0"
    image_dir = "/data1/qxy/MLM/src/image/denoised"
    text_file = "harmbench_behaviors_text_val.csv"
    denoise_checkpoint_num = 8 # (0,350,50)
    # dir for internal variables
    temp_dir = "./temp"

def get_similarity_list(args:Args, save_internal=False):
    """get the similarity matrix of text and corresponding denoised images
        called by main.py
        return: np.ndarray
            width: args.denoise_checkpoint_num
            height: len(text_embed_list)"""
    model = AutoModelForPreTraining.from_pretrained(
        args.MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to(args.DEVICE)# this runs out of memory

    # TODO maybe we can remove these later
    tokenizer = AutoTokenizer.from_pretrained(args.MODEL_PATH)
    imgprocessor = AutoImageProcessor.from_pretrained(args.MODEL_PATH)

    def get_img_embedding(image_path)->np.ndarray:
        image = Image.open(image_path)
        # img embedding
        pixel_value = imgprocessor(image, return_tensors="pt").pixel_values.to(args.DEVICE)
        image_outputs = model.vision_tower(pixel_value, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[model.config.vision_feature_layer]
        selected_image_feature = selected_image_feature[:, 1:] # by default
        image_features = model.multi_modal_projector(selected_image_feature)
        # calculate average to compress the 2th dimension
        image_features = torch.mean(image_features, dim=1).detach().to("cpu").numpy()
        del pixel_value, image_outputs, selected_image_feature
        torch.cuda.empty_cache()
        return image_features

    def get_text_embedding(text: str)->np.ndarray:
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(args.DEVICE)
        input_embeds = model.get_input_embeddings()(input_ids)
        # calculate average to get shape[1, 4096]
        input_embeds = torch.mean(input_embeds, dim=1).detach().to("cpu").numpy()
        del input_ids
        torch.cuda.empty_cache()
        return input_embeds

    # generate embeddings for texts
    text_embed_list = []
    with open(args.text_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            ret = get_text_embedding(row[0])
            text_embed_list.append(ret)

    # generate embeddings for images
    img_embed_list = []
    img_names = []
    dir1 = os.listdir(args.image_dir)
    dir1.sort()
    for img in dir1:
        ret = get_img_embedding(f"{args.image_dir}/{img}")
        img_embed_list.append(ret)
        img_names.append(os.path.splitext(img)[0])    

    # compute cosine similarity between text and n denoised images
    # and form a table of size (len(text_embed_list), len(text_embed_list)*args.denoise_checkpoint_num)
    cossims = np.zeros((len(text_embed_list), len(text_embed_list)*args.denoise_checkpoint_num))
    for i in range(len(text_embed_list)):
        for j in range(args.denoise_checkpoint_num):
            text_embed = text_embed_list[i]
            img_embed = img_embed_list[i*args.denoise_checkpoint_num+j]
            cos_sim = compute_cosine(img_embed, text_embed)
            cossims[i, j] = cos_sim


    # save embeddings and cosine similarity matrix
    if save_internal:
        torch.save(img_embed_list, f"{args.temp_dir}/embedding/image_embeddings.pt")
        torch.save(text_embed_list, f"{args.temp_dir}/embedding/text_embeddings.pt")
        with open(f"{args.temp_dir}/cosine_similarity.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(img_names)
            writer.writerows(cossims)
        print(f"csv file saved at: {args.temp_dir}")

    return cossims