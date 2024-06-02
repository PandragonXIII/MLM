import os, time


dirs = os.listdir("./")
dirs.remove("shorten.py")
dirs.remove("eval.py")
for d in dirs:
    print("evaluating...")
    # this can be done separately to avoid Out of Memory Error
    result_path = os.path.abspath(d)
    os.chdir("/home/xuyue/HarmBench")
    os.system(f"./scripts/evaluate_completions.sh \
        /home/xuyue/Model/HarmBench-Llama-2-13b \
        /home/xuyue/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
        {result_path}/response2.json \
        {result_path}/response2_eval.json")

    while not os.path.exists(f"{result_path}/response2_eval.json"):
        time.sleep(30)
    os.chdir("/home/xuyue/QXYtemp/MLM/gen_attack")