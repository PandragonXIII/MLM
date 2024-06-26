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
import time
import json

DELETE_TEMP_FILE = True

if __name__=="__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser(description='detect adversarial images and pass harmless images to LLMs')
    parser.add_argument('--text', type=str, required=True, help='path to the text input file')
    parser.add_argument('--img', type=str, required=True, help='path to the image input folder or file')
    parser.add_argument('--threshold', type=float, default=-0.0005046081542968749, help='threshold for adversarial detection')
    parser.add_argument('--model', type=str, default="llava", help='model to use for generation')
    parser.add_argument('--pair_mode', type=str, default="combine", help='relation between imgs and texts, should be "combine" or "injection"')
    parser.add_argument('--no_eval', action="store_true", help="disable harmbench evaluation phase" )
    parser.add_argument('--no_detect', action="store_true", help="disable detector phase" )
    parser.add_argument('--multirun', type=int, default=1, help="run the precess nultiple times")
    args = parser.parse_args()

    a = Args()
    a.pair_mode=args.pair_mode
    a.text_file = args.text
    a.denoise_checkpoint_num = 8
    a.model_path = "/home/xuyue/Model/llava-1.5-7b-hf"

    # create output dir
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
    else:
        os.system(f"rm -rf {a.output_dir}/*")
    os.mkdir(a.output_dir+"/img")

    for _ in range(args.multirun):
        # create temp dir
        if not os.path.exists(a.temp_dir):
            os.makedirs(a.temp_dir)
        else:
            os.system(f"rm -rf {a.temp_dir}/*")
        a.image_dir = a.temp_dir+"/denoised_img"
        os.mkdir(a.image_dir)
        

        if not args.no_detect:
            image_num, images = defence(args=args, a=a)
        else: # generate answers directly without denoise and classify
            if os.path.isdir(args.img):
                # read the images
                dir1 = os.listdir(args.img)
                # count the number of images in args.img
                image_num = len(dir1)
                dir1.sort()
                images = []
                for f in dir1:
                    if os.path.isfile(f"{args.img}/{f}"):
                        images.append(Image.open(f"{args.img}/{f}"))
            else:
                image_num = 1
                images = [Image.open(args.img)]

        # read the prompts from csv to a list
        df = pd.read_csv(a.text_file,header=None)
        texts = df[0].tolist()
        behaviours = df[df.shape[1]-1].tolist() # read behave for harmbench eval(v1_x for mmvet)
        # if args.eval_performance: # mm-vet do not have behav
        #     behaviours = df[1].tolist() # read 'v1_x'
        # else:
        #     behaviours = df[5].tolist() # for harmbench eval

        t_finish = time.time()
        print(f"processed {image_num} images, {len(texts)} texts in {t_finish-t0:.2f}s")

        # duplicate the texts and behaviours if pair_mode is combine
        if args.pair_mode=="combine":
            new_texts = []
            new_behaviours = []
            for i in range(len(texts)):
                new_texts.extend([texts[i]]*image_num)
                new_behaviours.extend([behaviours[i]]*image_num)
            text_len = len(texts)
            texts = new_texts
            behaviours = new_behaviours
            new_imgs = images*text_len
            images = new_imgs
        # generate responses
        print(f"using {args.model} for generation")
        responses = get_response(args.model, texts, images)

        # init dict in the first loop
        if _==0:
            res = {}
            for behav in set(behaviours): # init each behaviour class
                res[behav]=[]
        # save QA pairs to dict
        for i in range(len(texts)):
            res[behaviours[i]].append({"test_case":[images[i].filename,texts[i]],"generation":responses[i]}) # type: ignore
    
    # save dict to json
    with open(a.output_dir+"/response.json","w") as f:
        json.dump(res,f)

    if DELETE_TEMP_FILE:
            os.system(f"rm -rf {a.temp_dir}")
    
    t_generate = time.time()
    print(f"Full generation completed in {t_generate-t0:.2f}s")
    if not args.no_eval:
        # evaluate
        print("evaluating...")
        # this can be done separately to avoid Out of Memory Error
        result_path = os.path.abspath(a.output_dir)
        os.chdir("/home/xuyue/HarmBench")
        os.system(f"./scripts/evaluate_completions.sh \
            /home/xuyue/Model/HarmBench-Llama-2-13b \
            /home/xuyue/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
            {result_path}/response.json \
            {result_path}/response_eval.json")