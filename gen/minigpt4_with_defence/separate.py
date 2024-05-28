import json, os

fn="minigpt4_with_defence.json"
with open(fn) as f:
    data = json.load(f)
    d1,d2,d3=dict(),dict(),dict()
    for k,v in data.items():
        d1[k]=v[0]
        d2[k]=v[1]
        d3[k]=v[2]
d = [d1,d2,d3]
for i in range(3):
    with open(f"../separate/{fn[:-5]}_{i}(new).json","w") as fs:
        json.dump(d[i],fs)
        
