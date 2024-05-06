"""
This file works as an Integrated Block ahead of LLMs to filter potential 
adversarial and pass harmless image with origin text to LLM

input fromat:
- a single csv file with each line containing a text input
- a folder or a single file `.bmp` image(s), should be of same number with text\
    and we assume that they are in the same order

output:
- n bmp images under `/output` with same file name(s)
- add a column "input_adv_img" to the csv file denoting classification result
"""

import argparse,os
import pandas as pd

from model_tools import *
from defender import Defender
import sys
import denoiser.imagenet.denoise
import time

DELETE_TEMP_FILE = True

if __name__=="__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser(description='detect adversarial images and pass harmless images to LLMs')
    parser.add_argument('--text', type=str, required=True, help='path to the text input file')
    parser.add_argument('--img', type=str, required=True, help='path to the image input folder or file')
    parser.add_argument('--threshold', type=float, default=-0.0025, help='threshold for adversarial detection')
    args = parser.parse_args()

    a = Args()
    # deal with directory
    if not os.path.exists(a.temp_dir):
        os.makedirs(a.temp_dir)
    else:
        os.system(f"rm -rf {a.temp_dir}/*")
    a.image_dir = a.temp_dir+"/denoised_img"
    os.mkdir(a.image_dir)
    #
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
    else:
        os.system(f"rm -rf {a.output_dir}/*")
    os.mkdir(a.output_dir+"/img")

    # denoise the image
    print("processing images... ",end="")
    generate_denoised_img(args.img,a.image_dir,8)
    print("Done")
    # compute cosine similarity
    a.text_file = args.text
    a.denoise_checkpoint_num = 8
    a.model_path = "/home/xuyue/Model/llava-1.5-7b-hf"
    print("computing cossim... ",end="")
    sim_matrix = get_similarity_list(a, save_internal=True)
    print("Done")
    # for each row, check with detector
    d = Defender(threshold=args.threshold)
    adv_idx = []
    for i in range(sim_matrix.shape[0]):
        adv_idx.append(d.get_lowest_idx(sim_matrix[i]))

    # save the result to csv as new column
    df = pd.read_csv(args.text,header=None)
    df["input_adv_img"] = adv_idx
    df.to_csv(a.out_text_file,header=False,index=False)

    images = []
    # save the image to output folder
    denoised = os.listdir(a.image_dir)
    denoised.sort()
    for i in range(sim_matrix.shape[0]):
        idx = i*a.denoise_checkpoint_num+adv_idx[i]
        images.append(Image.open(f"{a.image_dir}/{denoised[idx]}")) # store the image for later generation of response
        os.system(
            f"cp {a.image_dir}/{denoised[idx]} {a.output_dir}/img/{denoised[idx]}")
    t_finish = time.time()
    print(f"filtered images are saved to {a.output_dir}/img")
    print(f"processed {sim_matrix.shape[0]} inputs in {t_finish-t0:.2f}s")
    if DELETE_TEMP_FILE:
        os.system(f"rm -rf {a.temp_dir}")
    pass

    # generate response
    from transformers import AutoProcessor, AutoModelForPreTraining
    processor = AutoProcessor.from_pretrained("/home/xuyue/Model/llava-1.5-7b-hf")
    model = AutoModelForPreTraining.from_pretrained("/home/xuyue/Model/llava-1.5-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    model.to("cuda:0")

    # read the prompts from csv to a list
    df = pd.read_csv(a.text_file,header=None)
    texts = df[0].tolist()
    behaviours = df[5].tolist() # for harmbench eval

    # generate response
    outputs = []
    for i in tqdm.tqdm(range(len(texts))):
        input = processor("<image>"+texts[i], images[i], return_tensors="pt").to("cuda:0")
        # autoregressively complete prompt
        output = model.generate(**input, max_new_tokens=100)
        outputs.append(output.to("cpu").numpy())
    del model
    torch.cuda.empty_cache()
    # save QA pairs to json
    import json
    res = {}
    for i in range(len(texts)):
        response = processor.decode(outputs[i][0], skip_special_tokens=True)
        res[behaviours[i]] = [{"test_case":[images[i].filename,texts[i]],"generation":response}]
    with open(a.output_dir+"/response.json","w") as f:
        json.dump(res,f)
    
    t_generate = time.time()
    print(f"Full generation completed in {t_generate-t0:.2f}s")

    # evaluate
    # this can be done separately to avoid Out of Memory Error
    result_path = os.path.abspath(a.output_dir)
    os.chdir("/home/xuyue/HarmBench")
    os.system(f"./scripts/evaluate_completions.sh \
        /home/xuyue/Model/HarmBench-Llama-2-13b \
        /home/xuyue/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
        {result_path}/response.json \
        {result_path}/response_eval.json")