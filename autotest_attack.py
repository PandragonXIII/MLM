"""
automatically test all models with/without defence using mm-vet dataset, 
and reformat the output and save separately to ./gen
"""
import os,json,time

def reformat(filename:str,read_dir:str,save_dir:str):
    """
    read data from read_dir and save the reformated to save_dir
    filename: saved file name
    the file to read is default as response.json
    """
    ndic = dict()
    with open(f"{read_dir}/response.json") as f:
        dic = json.load(f)
        print(f"query number: {len(dic.keys())}")
        for k,v in dic.items():
            ndic[k]=[conversation["generation"] for conversation in v]
    dp = os.path.join(save_dir,filename.split(".")[0])
    if not os.path.exists(dp):
        os.mkdir(dp)
    
    with open(f"{dp}/{filename}","w") as f1:
        json.dump(ndic, f1)
    # save original response json
    with open(f"{dp}/response.json","w") as f:
        json.dump(dic,f)
    if os.path.exists(f"{read_dir}/response_eval.json"):
        os.system(f"cp {read_dir}/response_eval.json {dp}/")


# imgdir = "/home/xuyue/QXYtemp/mm-vet-data/images218"
# textfile = "/home/xuyue/QXYtemp/mm-vet-data/mmvet_query.csv"
imgdir = "/home/xuyue/QXYtemp/MLM/src/image/testset"
textfile = "/home/xuyue/QXYtemp/MLM/src/text/testset_malicious.csv"
# imgdir = "/home/xuyue/QXYtemp/MLM/block_input/test_img"
# textfile = "/home/xuyue/QXYtemp/MLM/block_input/test_text.csv"

models = ["qwen"]
for model in models:
    os.system(f"""nohup python ./main.py --text {textfile} \
    --outdir /home/xuyue/QXYtemp/MLM/gen_attack/{model}_with_defence \
    --img {imgdir} --model {model} \
    --pair_mode combine  --threshold -0.003936767578125 \
    --no_eval --cuda 1 &""")
    time.sleep(600)
    while not os.path.exists(f"/home/xuyue/QXYtemp/MLM/gen_attack/{model}_with_defence/response.json"):
        time.sleep(60)
    # reformat(f"{model}_with_defence.json","/home/xuyue/QXYtemp/MLM/output","/home/xuyue/QXYtemp/MLM/gen_mmvet")
    print(f"-------{model} with defence generation complete-------")
    time.sleep(10)
    os.system(f"""nohup python ./main.py --text {textfile} --no_detect \
    --img {imgdir} --model {model} \
    --outdir /home/xuyue/QXYtemp/MLM/gen_attack/{model}_no_defence \
    --pair_mode combine  --threshold -0.003936767578125 \
    --no_eval --cuda 1 &""")
    time.sleep(600)
    while not os.path.exists(f"/home/xuyue/QXYtemp/MLM/gen_attack/{model}_no_defence/response.json"):
        time.sleep(60)
    # reformat(f"{model}_no_defence.json","/home/xuyue/QXYtemp/MLM/output","/home/xuyue/QXYtemp/MLM/gen_mmvet")
    print(f"-------{model} without defence generation complete-------")
    time.sleep(5)