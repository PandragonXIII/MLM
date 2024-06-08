import os, time


dirs = os.listdir("./")
dirs.sort()
for d in dirs:
    if os.path.isfile(d):
        continue
    print("evaluating...")
    # this can be done separately to avoid Out of Memory Error
    result_path = os.path.abspath(d)
    os.chdir("/home/xuyue/HarmBench")
    os.system(f"./scripts/evaluate_completions.sh \
        /home/xuyue/Model/HarmBench-Llama-2-13b \
        /home/xuyue/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
        {result_path}/response.json \
        {result_path}/response_eval.json")

    while not os.path.exists(f"{result_path}/response_eval.json"):
        time.sleep(5)
    os.chdir("/home/xuyue/QXYtemp/MLM/gen_JP_attack")