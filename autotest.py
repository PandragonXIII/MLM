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


imgdir = "/home/xuyue/QXYtemp/mm-vet-data/images218"
textfile = "/home/xuyue/QXYtemp/mm-vet-data/mmvet_query.csv"

models = ["minigpt4"]
for model in models:
    os.system(f"""nohup python ./main.py --text {textfile} \
    --img {imgdir} --model {model} \
    --pair_mode injection  --threshold -0.0005046081542968749 \
    --no_eval --multirun 3 &""")
    time.sleep(600)
    while not os.path.exists("/home/xuyue/QXYtemp/MLM/output/response.json"):
        time.sleep(30)
    reformat(f"{model}_with_defence.json","/home/xuyue/QXYtemp/MLM/output","/home/xuyue/QXYtemp/MLM/gen")
    print(f"-------{model} with defence generation complete-------")
    time.sleep(10)
    os.system(f"""nohup python ./main.py --text {textfile} \
    --img {imgdir} --model {model} \
    --pair_mode injection  --threshold -0.0005046081542968749 \
    --no_eval --no_detect --multirun 3 &""")
    time.sleep(600)
    while not os.path.exists("/home/xuyue/QXYtemp/MLM/output/response.json"):
        time.sleep(30)
    reformat(f"{model}_no_defence.json","/home/xuyue/QXYtemp/MLM/output","/home/xuyue/QXYtemp/MLM/gen")
    print(f"-------{model} without defence generation complete-------")
    time.sleep(5)