import csv, os
import math
import random

def get_binign_validation():
    # for each of the .csv file in ./MMLU, read the first line and store in ./MMLU/first_line.csv
    for file in os.listdir("./MMLU/val"):
        with open(f"./MMLU/val/{file}", "r", encoding="utf-8") as fr:
            reader = csv.reader(fr)
            for row in reader:
                with open(f"./first_line.csv", "a", encoding="utf-8") as fw:
                    writer = csv.writer(fw, lineterminator="\n")
                    writer.writerow(row)
                break

def get_benign_test(file_name:str):
    """get 200 samples from MMLU test dataset evenly(29*4+28*3)"""
    # delete the file if it already exists
    if os.path.exists(f"./{file_name}"):
        os.remove(f"./{file_name}")
    cnt=0
    dir = os.listdir("./MMLU/test")
    dir.sort()
    for i in range(len(dir)):
        with open(f"./MMLU/test/{dir[i]}", "r", encoding="utf-8") as fr:
            # calculate muw many rows to sample in this file
            sample_num = math.ceil((200-cnt)/(len(dir)-i))
            for row in csv.reader(fr):
                with open(f"./{file_name}", "a", encoding="utf-8") as fw:
                    writer = csv.writer(fw, lineterminator="\n")
                    writer.writerow(row)
                cnt+=1
                sample_num-=1
                if sample_num<=0:
                    break
            # if cnt==200:
            #     break

def get_malicious_test(file_name:str):
    """get 200 samples from Hambench test dataset with FunctionalCategory=standard """
    # delete the file if it already exists
    if os.path.exists(f"./{file_name}"):
        os.remove(f"./{file_name}")
    
    cnt=0
    with open(f"./Harmbench/harmbench_behaviors_text_test.csv", "r", encoding="utf-8") as fr:
        with open(f"./{file_name}", "a", encoding="utf-8") as fw:
            for row in csv.reader(fr):
                if row[1]=="standard":
                    writer = csv.writer(fw, lineterminator="\n")
                    writer.writerow(row)
                    cnt+=1
                if cnt==200:
                    break
    

if __name__ == "__main__":
    # get_benign_test("testset_benign.csv")
    # get_malicious_test("testset_malicious.csv")
    # with open(f"./src/text/RedTeam_2K.csv", "r", encoding="utf-8") as fr:
    #         reader = csv.reader(fr)
    #         origin_data = [r for r in reader]
    # data = random.sample(origin_data,k=200)
    # with open(f"./src/text/RedTeam_200.csv", "w", encoding="utf-8") as fw:
    #     writer = csv.writer(fw, lineterminator="\n")
    #     for r in data:
    #         writer.writerow(r)
    with open(f"./src/text/RedTeam_1800.csv", "r", encoding="utf-8") as fr:
            reader = csv.reader(fr)
            origin_data = [r for r in reader]
    data = random.sample(origin_data,k=200)
    with open(f"./src/text/RedTeam_test_200.csv", "w", encoding="utf-8") as fw:
        writer = csv.writer(fw, lineterminator="\n")
        for r in data:
            writer.writerow(r)
    data2 = random.sample(data,k=100)
    with open(f"./src/text/RedTeam_test_100.csv", "w", encoding="utf-8") as fw:
        writer = csv.writer(fw, lineterminator="\n")
        for r in data2:
            writer.writerow(r)
    pass