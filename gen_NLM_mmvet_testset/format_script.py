import os,json

def reformat(filename:str,read_dir:str,save_dir:str):
    """
    read data from read_dir and save the reformated to save_dir for mmvst test
    filename: saved file name
    the file to read is default as response.json
    """
    ndic = dict()
    with open(f"{read_dir}/response.json") as f:
        dic = json.load(f)
        print(f"query number: {len(dic.keys())}")
        for k,v in dic.items():
            # ndic[k]=[conversation["generation"] for conversation in v]
            ndic[k]=v[0]["generation"]
    dp = os.path.join(save_dir,filename.split(".")[0])
    if not os.path.exists(dp):
        os.mkdir(dp)
    
    with open(f"{dp}/{filename}","w") as f1:
        json.dump(ndic, f1)
    # save original response json
    # with open(f"{dp}/response.json","w") as f:
    #     json.dump(dic,f)
    # if os.path.exists(f"{read_dir}/response_eval.json"):
    #     os.system(f"cp {read_dir}/response_eval.json {dp}/")

print(os.listdir("/home/xuyue/QXYtemp/MLM/gen_NLM_mmvet_testset"))
dirs = os.listdir("/home/xuyue/QXYtemp/MLM/gen_NLM_mmvet_testset")
dirs.remove("format_script.py")
for d in dirs:
    reformat("reformatted.json",f"/home/xuyue/QXYtemp/MLM/gen_NLM_mmvet_testset/{d}",f"/home/xuyue/QXYtemp/MLM/gen_NLM_mmvet_testset/{d}")