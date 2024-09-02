from transformers import AutoModelForPreTraining, AutoTokenizer, AutoImageProcessor
# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image, ImageOps
import os, csv
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time
from math import ceil
import json


from denoiser.imagenet.DRM import DiffusionRobustModel
from defender import Defender

# target models
from transformers import AutoProcessor, AutoModelForPreTraining
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer # qwen

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat,CONV_VISION_Vicuna0

from openai import OpenAI
import httpx
import base64
from io import BytesIO

from skimage import img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma


openai_settings = json.load(open("../openai_env_var.json"))
os.environ["OPENAI_API_KEY"] = openai_settings["api_key"]
os.environ["OPENAI_BASE_URL"] = openai_settings["url"]

def resize_keep_ratio(img:Image.Image):
    if img.height>img.width:
        nw = img.width*224//img.height
        nimg = img.resize((nw,224))
        lappend = (224-nw)//2
        rappend = 224-nw-lappend
        nimg = ImageOps.expand(nimg,(lappend,0,rappend,0))
    else:
        nh = img.height*224//img.width
        nimg = img.resize((224,nh))
        uappend = (224-nh)//2
        bappend = 224-nh-uappend
        nimg = ImageOps.expand(nimg,(0,uappend,0,bappend))
    return nimg

# REMINDER: gpt4 only support png,jp(e)g,webp,gif now
def encode_image(image_path):
    """convert image to jpeg and encode with base64"""
    img = Image.open(image_path).convert("RGB")
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    return base64.b64encode(im_bytes).decode('utf-8')

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
    image_dir = "/data1/qxy/MLM/temp/denoised_img"
    text_file = "harmbench_behaviors_text_val.csv"
    denoise_checkpoint_num = 8 # (0,350,50)
    pair_mode = "combine"
    denoise_batch = 50
    # dir for output images
    output_dir = "./output"
    # dir for internal variables
    temp_dir = "./temp"
    out_text_file = f"{temp_dir}/text.csv"
    # args for minigpt4
    cfg_path = "./minigpt4/minigpt4_eval.yaml"    
    batch_size=1
    image_num=100
    input_res=224
    alpha=1.0
    epsilon=8
    steps=8
    output="temp"
    image_path="temp"
    tgt_text_path="temp.txt"
    question_path="temp.txt"
    start_idx=0
    num_query=10
    sigma=8
    # --options args, with nargs="+", for override
    options=None

def defence(args, a:Args):
    """
    resize the input image if needed
    denoise, calculate cossim and choose the safe images
    return (image_num, images)
    """
    # denoise the image
    print("processing images... ",end="")
    # count the number of images in args.img
    image_num = 0
    if os.path.isdir(args.img):
        image_num = len(os.listdir(args.img))
    else:
        image_num = 1
    generate_denoised_img(args.img,a.image_dir,8,batch_size=50,device=a.device)
    # generate_denoised_img_with_NLM(args.img,a.image_dir,a.denoise_checkpoint_num,batch_size=50,device=a.device)
    print("Done")
    # compute cosine similarity
    print("computing cossim... ",end="")
    sim_matrix = get_similarity_list(a, save_internal=True)
    print("Done")
    # for each row, check with detector
    d = Defender(threshold=args.threshold)
    adv_idx = []
    for i in range(sim_matrix.shape[0]):
        adv_idx.append(d.get_lowest_idx(sim_matrix[i]))

    # save the result to csv as new column
    # this don't work for conbine mode
    # df = pd.read_csv(args.text,header=None)
    # df["input_adv_img"] = adv_idx
    # df.to_csv(a.out_text_file,header=False,index=False)

    images = []
    # save the image to output folder
    denoised = os.listdir(a.image_dir)
    denoised.sort()
    for i in range(sim_matrix.shape[0]):
        # find specific image and denoise time
        idx = i%image_num*a.denoise_checkpoint_num+adv_idx[i] 
        images.append(Image.open(f"{a.image_dir}/{denoised[idx]}")) # store the image for later generation of response
        # save to disk
        # os.system(
        #     f"cp {a.image_dir}/{denoised[idx]} {a.output_dir}/img/{denoised[idx]}")
    # print(f"filtered images are saved to {a.output_dir}/img")
    return (image_num, images)

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
    image_num = len(img_embed_list)//args.denoise_checkpoint_num
    if args.pair_mode=="combine":
        cossims = np.zeros((len(text_embed_list)*len(img_embed_list), args.denoise_checkpoint_num))
        for i in range(len(text_embed_list)):
            for j in range(image_num):
                for k in range(args.denoise_checkpoint_num):
                    text_embed = text_embed_list[i]
                    img_embed = img_embed_list[j*args.denoise_checkpoint_num+k]
                    cossims[i*image_num+j, k] = compute_cosine(img_embed, text_embed)
    else: # injection
        cossims = np.zeros((len(text_embed_list), args.denoise_checkpoint_num))
        for i in range(len(text_embed_list)):
            for j in range(args.denoise_checkpoint_num):
                text_embed = text_embed_list[i]
                img_embed = img_embed_list[i*args.denoise_checkpoint_num+j]
                # FIXME IndexError: list index out of range
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

def generate_denoised_img(path:str, save_path, cps:int , step=50, DEVICE="cuda:0", batch_size = 50):
    """
    read all img file under given dir, and convert to RGB
    copy the original size image as denoise000,
    then resize to 224*224,
    denoise for multiple iterations and save them to save_path
    
    :path: path of dir or file of image(s)
    :cps: denoise range(0,step*cps,step) times
            step=50 by default
    """
    
    assert os.path.isdir(path)
    resized_imgs = []
    img_names = []
    dir1 = os.listdir(path) # list of img names
    dir1.sort()
    for filename in dir1:
        img = Image.open(path+"/"+filename).convert("RGB")
        # save the original image
        savename=os.path.splitext(filename)[0]+"_denoised_000times.jpg"
        img.save(os.path.join(save_path,savename))
        # resize for denoise
        resized_imgs.append(
            resize_keep_ratio(img=img)
        )
        img_names.append(filename)
    if cps<=1:
        return
    
    model = DiffusionRobustModel()
    iterations = range(step,cps*step,step)
    b_num = ceil(len(resized_imgs)/batch_size) # how man runs do we need
    for b in tqdm.tqdm(range(b_num),desc="denoise batch"):
        l = b*batch_size
        r = (b+1)*batch_size if b<b_num-1 else len(resized_imgs)
        # denoise for each part between l and r
        part = resized_imgs[l:r]
        partname = img_names[l:r]
        for it in iterations:
            # project value between -1,1
            ary = [np.array(_,dtype=np.float32)/255*2-1 for _ in part]
            ary = np.array(ary)
            ary = torch.tensor(ary).permute(0,3,1,2).to(DEVICE)
            denoised_ary=np.array(model.denoise(ary, it).to("cpu"))
            denoised_ary=denoised_ary.transpose(0,2,3,1)
            denoised_ary = (denoised_ary+1)/2*255
            denoised_ary = denoised_ary.astype(np.uint8)
            for i in range(denoised_ary.shape[0]):
                img=Image.fromarray(denoised_ary[i])
                sn = os.path.splitext(partname[i])[0]+"_denoised_{:0>3d}times.jpg".format(it)
                img.save(os.path.join(save_path,sn))
    del model
    torch.cuda.empty_cache()
    return

def generate_denoised_img_with_NLM(path:str, save_path, cps:int , step=1, device="cuda:1", batch_size = 50):
    """
    read all img file under given dir, and convert to RGB
    copy the original size image as denoise000,
    then resize to 224*224,
    denoise for multiple iterations and save them to save_path
    using Non Local Means (NLM)
    
    :path: path of dir or file of image(s)
    :cps: denoise range(0,step*cps,step) times
            step=1 by default
    """
    
    assert os.path.isdir(path)
    resized_imgs = []
    img_names = []
    dir1 = os.listdir(path) # list of img names
    dir1.sort()
    for filename in dir1:
        img = Image.open(path+"/"+filename).convert("RGB")
        # save the original image
        savename=os.path.splitext(filename)[0]+"_denoised_000times.jpg"
        img.save(os.path.join(save_path,savename))
        # resize for denoise
        resized_imgs.append(
            resize_keep_ratio(img=img)
        )
        img_names.append(filename)
    if cps<=1:
        return
    
    # model = DiffusionRobustModel(device=int(device[-1]))
    iterations = range(step,cps*step,step)
    b_num = ceil(len(resized_imgs)/batch_size) # how man runs do we need
    for b in tqdm.tqdm(range(b_num),desc="denoise batch"):
        l = b*batch_size
        r = (b+1)*batch_size if b<b_num-1 else len(resized_imgs)
        # denoise for each part between l and r
        part = resized_imgs[l:r]
        partname = img_names[l:r]
        for it in iterations:
            # project value between -1,1
            ary = [np.array(_,dtype=np.float32)/255*2-1 for _ in part]
            ary = np.array(ary)
            denoised_ary = []
            for img_temp in ary:
                sigma_est = np.mean(estimate_sigma(img_temp, channel_axis=-1))
                patch_kw = dict(
                    patch_size=5, patch_distance=6, channel_axis=-1  # 5x5 patches  # 13x13 search area
                )
                for _ in range(it):
                    img_temp = denoise_nl_means(
                        img_temp, h=0.8 * sigma_est, sigma=sigma_est, fast_mode=False, **patch_kw
                    )
                denoised_ary.append(img_temp)
            denoised_ary = np.array(denoised_ary)
            denoised_ary = (denoised_ary+1)/2*255
            denoised_ary = denoised_ary.astype(np.uint8)
            for i in range(denoised_ary.shape[0]):
                img=Image.fromarray(denoised_ary[i])
                sn = os.path.splitext(partname[i])[0]+"_denoised_{:0>3d}times.jpg".format(it)
                img.save(os.path.join(save_path,sn))
    # del model
    torch.cuda.empty_cache()
    return


def get_response(model_name, texts, images, a=Args()):
    """
    get different model response with given texts and images pairs
    :model_name: on of "llava", "minigpt4", "blip","gpt4"...
    """
    answers = []
    t_start=time.time()
    # TODO use batch encode and decode
    if model_name in ["llava", "blip", "llava1.6"]:
        if model_name=="llava":
            processor = AutoProcessor.from_pretrained("/home/xuyue/Model/llava-1.5-7b-hf")
            model = AutoModelForPreTraining.from_pretrained("/home/xuyue/Model/llava-1.5-7b-hf") 
        elif model_name=="llava1.6":
            processor = LlavaNextProcessor.from_pretrained("/home/xuyue/Model/llava_1.6_7b_hf")
            model = LlavaNextForConditionalGeneration.from_pretrained("/home/xuyue/Model/llava_1.6_7b_hf") 
        else: # blip
            processor = InstructBlipProcessor.from_pretrained("/home/xuyue/Model/Instructblip-vicuna-7b")
            model = InstructBlipForConditionalGeneration.from_pretrained("/home/xuyue/Model/Instructblip-vicuna-7b") 
        model.to(a.DEVICE) # type: ignore
        for i in tqdm.tqdm(range(len(texts)),desc="generating response"):
            if "000" not in images[i].filename and "denoise" in images[i].filename:
                answers.append("sorry,  I can not assist with that.")
                continue
            input = processor(text=f"<image>\n{texts[i]}\n", images=images[i], return_tensors="pt").to("cuda:0")
            # autoregressively complete prompt
            output = model.generate(**input,max_new_tokens=512) # type: ignore
            outnpy=output.to("cpu").numpy()
            answer = processor.decode(outnpy[0], skip_special_tokens=True)
            answers.append(answer.replace(f"\n{texts[i]}\n","").strip())
        del model
        torch.cuda.empty_cache()
    elif model_name.lower()=="minigpt4":
        print(f"Loading MiniGPT-4 models...",end="")
        # load models for i2t
        cfg = Config(a)
        model_config = cfg.model_cfg
        model_config.device_8bit = int(a.DEVICE[-1]) # get gpu_id
        model_cls = registry.get_model_class(model_config.arch)  # model_config.arch: minigpt-4
        model = model_cls.from_config(model_config).to(a.DEVICE)

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor     = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)       
        num_beams = 1
        temperature = 1.0
        chat = Chat(model, vis_processor, device=a.DEVICE)
        print("Done")
        # start querying
        for i in tqdm.tqdm(range(len(texts)),desc="generating response"):
            if "000" not in images[i].filename and "denoise" in images[i].filename: # refuse to answer adv image query
                answers.append("sorry,  I can not assist with that.")
                continue
            answers.append(query_minigpt(
                question="<image>\n"+texts[i], img=images[i],chat=chat
            ))
        del chat,model,vis_processor
        torch.cuda.empty_cache()
    elif model_name=="qwen":
        tokenizer = AutoTokenizer.from_pretrained("/home/xuyue/Model/Qwen_VL_Chat",trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("/home/xuyue/Model/Qwen_VL_Chat",trust_remote_code=True,fp32=True).eval()
        model.to(a.DEVICE)
        for i in tqdm.tqdm(range(len(texts)),desc="generating response"):
            if "000" not in images[i].filename and "denoise" in images[i].filename:
                answers.append("sorry,  I can not assist with that.")
                continue
            input = tokenizer.from_list_format([{"image":images[i].filename},{"text":f"{texts[i]}"}]) # type: ignore
            # autoregressively complete prompt
            answer, history = model.chat(tokenizer, query=input, history=None ,max_new_tokens=512)
            answers.append(answer)
        del model
        torch.cuda.empty_cache()
    elif model_name=="gpt4":
        client = OpenAI(
        http_client=httpx.Client(
            base_url="https://api.xiaoai.plus/v1",
            follow_redirects=True,
            ),
        )
        
        for i in tqdm.tqdm(range(len(texts)),desc="generating response"):
            if "000" not in images[i].filename and "denoise" in images[i].filename:
                answers.append("sorry,  I can not assist with that.")
                continue
            base64_image = encode_image(images[i].filename)
            errorcnt=1
            while errorcnt>0:
                try: # api may raise error
                    response = client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=[
                            {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": texts[i]},
                                {"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}",
                                            "detail": "auto"},
                                }
                            ],
                            }
                        ],
                        max_tokens=300
                    )
                except Exception as e:
                    errorcnt+=1
                    time.sleep(3)
                    if errorcnt>6:
                        print(f"exception encountered with query{i}")
                        print(e)
                        answer=""
                        break
                else:
                    errorcnt=0 # successfully generated, break loop
                    answer = response.choices[0].message.content
            answers.append(answer)
            time.sleep(1)
    else:
        raise Exception("unrecognised model_name, please choose from llava,blip,minigpt4,qwen")
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
                              max_new_tokens=512,
                              max_length=2000)[0]

    return llm_message, chat_state, img_list


def query_minigpt(question,img,chat):
    with torch.no_grad():
        chat_state, img_list = upload_img(chat,img)
        chat_state = ask(chat,question, chat_state)
        llm_message, chat_state, img_list = answer(chat,chat_state, img_list)
    return llm_message

def get_img_embedding(image_path:list):
    model_path = "/home/xuyue/Model/llava-1.5-7b-hf"
    DEVICE = "cuda:1"
    ########################
    model = AutoModelForPreTraining.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to(DEVICE)
    imgprocessor = AutoImageProcessor.from_pretrained(model_path)

    # model.to(DEVICE)
    images = []
    for img in image_path:
        images.append(Image.open(img))
    
    res = []
    for image in images:
        # img embedding
        pixel_value = imgprocessor(image, return_tensors="pt").pixel_values.to(DEVICE)
        image_outputs = model.vision_tower(pixel_value, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[model.config.vision_feature_layer]
        selected_image_feature = selected_image_feature[:, 1:] # by default
        image_features = model.multi_modal_projector(selected_image_feature)
        # calculate average to compress the 2th dimension
        image_features = torch.mean(image_features, dim=1).detach().to("cpu")
        res.append(image_features)
    del pixel_value, image_outputs, selected_image_feature
    # model.to("cpu")
    torch.cuda.empty_cache()
    return res