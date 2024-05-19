import os, csv
import numpy

"""read data of multiple generations and corresponding evaluation from 
`./performance_evaluation/**/.*-cap-score-1runs.csv` 
and calculate average score for each model"""

srcdir = "./performance_evaluation"
dirs = os.listdir(srcdir)
dirs.sort()
data = dict()
for dir in dirs:
    with open(f"{srcdir}/{dir}/{dir}_gpt-4-32k-0613-cap-score-1runs.csv") as f:
        reader = csv.reader(f)
        reader.__next__()
        for row in reader:
            # print(row)
            if row[0][:-2] not in data:
                data[row[0][:-2]] = []
            run = [float(j) for j in row[1:8]]
            data[row[0][:-2]].append(run)
# calculate average
for k in data.keys():
    std = numpy.std([data[k][i][-1] for i in range(3)])
    data[k] = numpy.average(data[k],axis=0)
    data[k] = numpy.round(data[k],2)
    data[k] =  data[k].tolist()
    data[k].append(round(std,2))
print(data)