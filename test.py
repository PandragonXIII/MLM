from transformers import AutoProcessor, AutoModelForPreTraining
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import time,tqdm,torch
from PIL import Image
import os, csv
import numpy as np

model_name="llava"
answers = []
t_start=time.time()
# TODO use batch encode and decode
if model_name=="llava" or model_name=="blip":
    if model_name=="llava":
        processor = AutoProcessor.from_pretrained("/home/xuyue/Model/llava-1.5-7b-hf")
        model = AutoModelForPreTraining.from_pretrained("/home/xuyue/Model/llava-1.5-7b-hf") 
    else: # blip
        processor = InstructBlipProcessor.from_pretrained("/home/xuyue/Model/Instructblip-vicuna-7b")
        model = InstructBlipForConditionalGeneration.from_pretrained("/home/xuyue/Model/Instructblip-vicuna-7b") 
    model.to("cuda:0")
    for i in tqdm.tqdm(range(len(texts)),desc="generating response"):
        input = processor(text=f"### Human: {texts[i]} <image> ### Assistant: ", images=images[i], return_tensors="pt").to("cuda:0")
        # autoregressively complete prompt
        output = model.generate(**input,max_length=300)
        outnpy=output.to("cpu").numpy()
        answer = processor.decode(outnpy[0], skip_special_tokens=True)
        answers.append(answer.split('Assistant: ')[-1].strip())
    del model
    torch.cuda.empty_cache()