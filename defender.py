"""
a defender class that classify adversarial and clean image with 
a decline threshold
Should contains methods:
1. train: calculate threshold with given data and specified ratio
2. predict: pass
3. validate: given validation dataset, call predict function and 
return statistics (e.g. confusion matrix)
"""

import numpy as np
import pandas as pd

class Defender():
    """
    given text and image, if any decreace of cosine similarity greater than threshold 
    (the delta values are smaller than threshold) occurs during denoise, 
    then consider it as adversarial
    """
    def __init__(self, threshold = None):
        self.threshold = threshold
        print(f"Defender initialized with threshold={self.threshold}")
    
    def train(self, data:pd.DataFrame, ratio=0.9):
        """
        calculate threshold with given data and specified conservative ratio
        :data: malicious query(text) v.s. clean image cosine similarity
            cols: denoise times
            rows: different text
        :ratio: the ratio of clean image cosine similarity decrease value that should be lower than threshold
        """
        # get the first col (origin image) as the base cosine similarity
        base = data.iloc[:40,0]
        # get the decrease value
        decrease = data.iloc[:40,1:].sub(base, axis=0)
        # get the ratio of decrease value that is lower than threshold
        self.threshold = np.percentile(decrease, (1-ratio)*100)
        print(f"Threshold updated to {self.threshold}")

    def predict(self, cossims):
        """
        given a series of cosine similarity , return whether it is adversarial
        by calculating the interpolation between each and the first one

        :cossims: a nD array of cosine similarity (m*n)
            cols: denoise times n
            row: m text
        return: a nD array of boolean (m*1)
            True means adversarial
        """
        if len(cossims.shape) == 1 or cossims.shape[1] == 1:
            return True in (np.array(cossims[:,1:]) - cossims[:,0] < self.threshold)
        else:
            ret = []
            for r in range(cossims.shape[0]):
                row = cossims.iloc[r]
                ret.append(True in (np.array(row[1:]) - row[0] < self.threshold))
            return ret

import csv
def test_defender_on_validation_set():
    """
    test the defender on validation set and save the results
    """
    f = "./src/analysis/similarity_matrix_validation.csv"
    df = pd.read_csv(f)
    # get the clean image data
    clean_header = [col for col in df.columns if "clean" in col]
    if len(clean_header)>8:
        clean_header = clean_header[:8]
    clean_data = df[clean_header]

    results = {
        "data percentage":[],
        "constraint":[],
        "accuracy":[],
        "recall":[],
        "precision":[],
        "f1":[],
        "classification threshold":[]
    }
    for threshold in [0.95, 0.975, 0.99, 0.995]:
        # train the defender on clean data and keep 95%
        d = Defender()
        d.train(clean_data, threshold)

        # test the defender on adversarial data
        adv16_header = [col for col in df.columns if "prompt_constrained_16" in col]
        adv32_header = [col for col in df.columns if "prompt_constrained_32" in col]
        adv64_header = [col for col in df.columns if "prompt_constrained_64" in col]
        advucon_header = [col for col in df.columns if "prompt_unconstrained" in col]
        # get data with denoise time leq than 350
        adv_data = []
        for adv_header in [adv16_header, adv32_header, adv64_header, advucon_header]:
            h_list=[]
            for h in adv_header:
                t = int(h.split("_")[-1].rstrip("times"))
                if t <= 350:
                    h_list.append(h)
            # get adversarial data of 16, 32, 64 constraint and unconstrained
            adv_data.append(df[h_list])

        constraint_names = [
            "16","32","64","unconstrainted"
        ]
        # predict whether it is adversarial
        for j in range(len(adv_data)):
            adv_predict = d.predict(adv_data[j])
            adv_predict = np.array(adv_predict)
            tp = sum(adv_predict[:40])
            fp = sum(adv_predict[40:])
            tn = 40 - fp
            fn = 40 - tp
            acc = (tp+tn)/80
            recall = tp/(tp+fn)
            precision = tp/(tp+fp)
            f1 = 2*precision*recall/(precision+recall)
            # print(f"Adversarial group: {constraint_names[j]}")
            # print(f"Accuracy: {acc}, Recall: {recall}, Precision: {precision}, F1: {f1}")
            results["data percentage"].append(threshold)
            results["constraint"].append(constraint_names[j])
            results["accuracy"].append(acc)
            results["recall"].append(recall)
            results["precision"].append(precision)
            results["f1"].append(f1)
            results["classification threshold"].append(d.threshold)
    # save the results
    results = pd.DataFrame(results)
    results.to_csv("./src/analysis/defender_validation_results.csv", index=False)
    print("Results saved to ./src/analysis/defender_validation_results.csv")
    # print 4 tables separately
    print("data percentage: 0.95")
    print(results[results["data percentage"]==0.95])
    print("data percentage: 0.975")
    print(results[results["data percentage"]==0.975])
    print("data percentage: 0.99")
    print(results[results["data percentage"]==0.99])
    print("data percentage: 0.995")
    print(results[results["data percentage"]==0.995])

if __name__ == "__main__":
    test_defender_on_validation_set()