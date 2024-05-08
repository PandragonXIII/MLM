from transformers import AutoModelForPreTraining, AutoTokenizer, AutoImageProcessor
import torch
from PIL import Image
import os, csv
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time


from denoiser.imagenet.DRM import DiffusionRobustModel

# target models
from transformers import AutoProcessor, AutoModelForPreTraining
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat,CONV_VISION_Vicuna0

def compute_cosine(a_vec:np.ndarray , b_vec:np.ndarray):
    """calculate cosine similarity"""
    norms1 = np.linalg.norm(a_vec, axis=1)
    norms2 = np.linalg.norm(b_vec, axis=1)
    dot_products = np.sum(a_vec * b_vec, axis=1)
    cos_similarities = dot_products / (norms1 * norms2) # ndarray with size=1
    return cos_similarities[0]

class Args:
    model_path = "/data1/qxy/models/llava-1.5-7b-hf"
    DEVICE = "cuda:0"
    image_dir = "/data1/qxy/MLM/src/image/denoised"
    text_file = "harmbench_behaviors_text_val.csv"
    denoise_checkpoint_num = 8 # (0,350,50)
    # dir for output images
    output_dir = "./output"
    # dir for internal variables
    temp_dir = "./temp"
    out_text_file = f"{temp_dir}/text.csv"

def get_similarity_list(args:Args, save_internal=False):
    """get the similarity matrix of text and corresponding denoised images
        called by main.py
        return: np.ndarray
            width: args.denoise_checkpoint_num
            height: len(text_embed_list)"""
    model = AutoModelForPreTraining.from_pretrained(
        args.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to(args.DEVICE)# this runs out of memory

    # TODO maybe we can remove these later
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    imgprocessor = AutoImageProcessor.from_pretrained(args.model_path)

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
    # and form a table of size (len(text_embed_list), args.denoise_checkpoint_num)
    cossims = np.zeros((len(text_embed_list), args.denoise_checkpoint_num))
    for i in range(len(text_embed_list)):
        for j in range(args.denoise_checkpoint_num):
            text_embed = text_embed_list[i]
            img_embed = img_embed_list[i*args.denoise_checkpoint_num+j]
            cossims[i, j] = compute_cosine(img_embed, text_embed)


    # save embeddings and cosine similarity matrix
    if save_internal:
        os.mkdir(args.temp_dir+"/embedding")
        torch.save(img_embed_list, f"{args.temp_dir}/embedding/image_embeddings.pt")
        torch.save(text_embed_list, f"{args.temp_dir}/embedding/text_embeddings.pt")
        with open(f"{args.temp_dir}/cosine_similarity.csv", "x") as f:
            writer = csv.writer(f)
            writer.writerow(img_names)
            writer.writerows(cossims)
        print(f"csv file saved at: {args.temp_dir}")

    return cossims

def generate_denoised_img(path:str, save_path, cps:int , step=50, DEVICE="cuda:0"):
    """
    read the specific file or all files(but not dirs) in the path ,
    denoise them for multiple iterations and save them to save_path
    
    :path: path of dir or file of image(s)
    :cps: denoise range(0,step*n,step) times
            step=50 by default
    """
    model = DiffusionRobustModel()

    batch = []
    names = []
    if os.path.isdir(path): # denoise all image under given dir
        dir1 = os.listdir(path)
        dir1.sort()
        for fn in dir1:
            if not os.path.isfile(path+"/"+fn):
                continue
            img = np.array(plt.imread(path+"/"+fn)/255*2-1).astype(np.float32)
            batch.append(img)
            names.append(os.path.splitext(fn)[0])
    elif os.path.isfile(path): # denoise specific image
        img = np.array(plt.imread(path)/255*2-1).astype(np.float32)
        batch.append(img)
        names.append(os.path.splitext(os.path.basename(path))[0])
    else:
        print("Invalid path")
        exit()
    batch = torch.tensor(np.stack(batch,axis=0)).to(DEVICE).permute(0,3,1,2)

    # set denoise iteration time
    iterations = range(0,cps*step,step)
    for it in tqdm.tqdm(iterations):
        if it!=0:
            denoise_batch = np.array(model.denoise(batch, it).to("cpu"))
        else:
            denoise_batch = np.array(batch.to("cpu"))
        denoise_batch = denoise_batch.transpose(0,2,3,1)
        denoise_batch = (denoise_batch+1)/2
        # print(denoise_batch.shape)
        # print(denoise_batch.max())
        # print(denoise_batch.min())

        for i in range(batch.shape[0]):
            plt.imsave("{save_path}/{name}_denoised_{:0>3d}times.bmp".format(
                it, save_path=save_path, name=names[i]
            ), denoise_batch[i])

def get_response(model_name, texts, images):
    """
    get different model response with given texts and images pairs
    :model_name: on of "llava", "minigpt4", "blip",...
    """
    answers = []
    t_start=time.time()
    # TODO use batch encode and decode
    if model_name=="llava" or model_name=="blip":
        if model_name=="llava":
            processor = AutoProcessor.from_pretrained("/home/xuyue/Model/llava-1.5-7b-hf")
            model = AutoModelForPreTraining.from_pretrained("/home/xuyue/Model/llava-1.5-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        else: # blip
            processor = InstructBlipProcessor.from_pretrained("/home/xuyue/Model/Instructblip-vicuna-7b")
            model = InstructBlipForConditionalGeneration.from_pretrained("/home/xuyue/Model/Instructblip-vicuna-7b", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        model.to("cuda:0")
        for i in tqdm.tqdm(range(len(texts)),desc="generating response"):
            input = processor(text="<image>"+texts[i], images=images[i], return_tensors="pt").to("cuda:0")
            # autoregressively complete prompt
            output = model.generate(**input,max_length=100)
            outnpy=output.to("cpu").numpy()
            answers.append(processor.decode(outnpy[0], skip_special_tokens=True))
        del model
        torch.cuda.empty_cache()
    elif model_name.lower()=="minigpt4":
        pass
    else:
        raise Exception("unrecognised model_name, please choose from llava,blip,minigpt4")
    t_gen = time.time()-t_start
    print(f"generation part takes {t_gen:.2f}s")
    return answers


# helper functions for miniGPT-4
def upload_img(chat,img):
    CONV_VISION = CONV_VISION_Vicuna0
    chat_state = CONV_VISION.copy()
    img_list = []
    chat.upload_img(img, chat_state, img_list)
    chat.encode_img(img_list)
    return chat_state, img_list


def ask(chat,user_message, chat_state):
    chat.ask(user_message, chat_state)
    return chat_state


def answer(chat,chat_state, img_list, num_beams=1, temperature=1.0):
    llm_message  = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]

    return llm_message, chat_state, img_list


def query_minigpt(question,img,chat):
    with torch.no_grad():
        chat_state, img_list = upload_img(chat,img)
        chat_state = ask(chat,question, chat_state)
        llm_message, chat_state, img_list = answer(chat,chat_state, img_list)
    return llm_message