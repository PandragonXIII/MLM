import os,json

dirs = os.listdir("./")
dirs.remove("shorten.py")
for d in dirs:
    with open(f"{d}/response.json") as f:
        data = json.load(f)
        newdata = dict()
        for k,v in data.items():
            v = v[:8]
            newdata[k]=v
    with open(f"{d}/response2.json","a") as f2:
        json.dump(newdata,f2)
