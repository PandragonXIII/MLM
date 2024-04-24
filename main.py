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

from model_tools import *

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='detect adversarial images and pass harmless images to LLMs')
    parser.add_argument('--text', type=str, required=True, help='path to the text input file')
    parser.add_argument('--img', type=str, required=True, help='path to the image input folder or file')
    args = parser.parse_args()

    # denoise the image

    # compute cosine similarity
    a = Args()
    a.image_dir = ""
    a.text_file = ""
    a.denoise_checkpoint_num = 8
    a.MODEL_PATH = "/data1/qxy/models/llava-1.5-7b-hf"
    sim_matrix = get_similarity_list(a, save_internal=False)
    # check with detector

    # save the result to csv

    # save the image to output folder

    pass