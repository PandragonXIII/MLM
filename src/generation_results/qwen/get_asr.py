import os, json

dirs = ["./qwen_adv_eval.json"]
for d in dirs:
    with open(d) as f:
        data = json.load(f)
    clean_count,clean_break,adv_count,adv_break=0,0,0,0
    for v in data.values():
        for case in v:
            if "clean" in case["test_case"][0]:
                clean_count+=1
                if case["label"]==1:
                    clean_break+=1
            else: 
                adv_count+=1
                if case["label"]==1:
                    adv_break+=1
    c_asr = clean_break/clean_count
    a_asr = adv_break/adv_count
    print(f"{d}:\nclean asr: {c_asr}\nadv asr: {a_asr}")