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
from typing import List
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class Defender():
    """
    given text and image, if any decreace of cosine similarity greater than threshold 
    (the delta values are smaller than threshold) occurs during denoise, 
    then consider it as adversarial
    """
    def __init__(self, threshold = None):
        self.threshold = threshold
        print(f"Defender initialized with threshold={self.threshold}")
    
    def train(self, data:pd.DataFrame| List[pd.DataFrame], ratio=0.9):
        """
        calculate threshold with given data and specified conservative ratio
        :data: malicious query(text) v.s. clean image cosine similarity
            cols: denoise times
            rows: different text
        :ratio: the ratio of clean image cosine similarity decrease value that should be lower than threshold
        """
        if type(data) == list:
            for df in data:
                names = [i for i in range(df.shape[1])]
                df.columns = names
            data = pd.concat(data, axis=0, ignore_index=True)
        # get the first col (origin image) as the base cosine similarity
        base = data.iloc[:,0]
        # get the decrease value
        decrease = data.iloc[:,1:].sub(base, axis=0)
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
        if len(cossims.shape) == 1:
            return True in (np.array(cossims[1:]) - cossims[0] < self.threshold)
        else:
            ret = []
            for r in range(cossims.shape[0]):
                row = cossims.iloc[r]
                ret.append(True in (np.array(row.iloc[1:]) - row.iloc[0] < self.threshold))
            return ret
        
    def get_lowest_idx(self, cossims):
        """
        given a series of cosine similarity , return whether it is adversarial
        for adversarial data, return the index of lowest cosine similarity

        :cossims: a nD array of cosine similarity (1*n)
            cols: denoise times n
            row: 1 text
        return: a nD array of int (1*1)
            0 is clean, positive int is the index of lowest cosine similarity
        """
        # reshape the data
        cossims = np.array(cossims).reshape(1,-1)
        delta = cossims - cossims[0,0]
        if np.min(delta)<self.threshold:
            return np.argmin(delta)
        else:
            return 0
    
    def get_confusion_matrix(self, datapath:str,checkpt_num=8,group=True):
        """test the defender with given cosine similarity data, 
        save the statics to savepath\n
        only consider malicious text\n
        the result contains 4 rows: for each adversarial image, 
        consider it as positive and clean as negative, 
        output the results
        checkpt_num: the maximum number of denoise times checkpoint to consider
        group: if true, output separate matrix for different adv image"""
        df = pd.read_csv(datapath)
        df = df[df["is_malicious"]==1] # only consider malicious text input
        results = {
            "constraint":[],
            "accuracy":[],
            "recall":[],
            "precision":[],
            "f1":[],
            "classification threshold":[],
            "fpr":[]
        }
        fp,tn=0,0
        # get the Test Set clean image data for prediction
        all_clean_header = [col for col in df.columns if "clean_" in col]
        clean_classes_names = set(["_".join(h.split("_")[:2]) for h in all_clean_header])
        for clean_class in clean_classes_names:
            clean_header = [col for col in df.columns if clean_class+"_" in col]
            if len(clean_header)>checkpt_num:
                clean_header = clean_header[:checkpt_num]
            clean_data = df[clean_header]
            # predict with clean image
            clean_predict = np.array(self.predict(clean_data))
            fp += sum(clean_predict[:])
            tn += sum(~clean_predict[:])

        tot_tp,tot_fn=0,0
        # get the adversarial image data
        all_adv_header = [col for col in df.columns if "prompt_" in col]
        adv_classes_names = set(["_".join(h.split("_")[:3]) for h in all_adv_header])
        for adv_class in adv_classes_names:
            # list the headers of adv_class constraint
            adv_header = [col for col in df.columns if adv_class+"_" in col]
            if len(adv_header)>checkpt_num:
                adv_header = adv_header[:checkpt_num]
            # get data
            adv_data = df[adv_header]
            # predict
            adv_predict = np.array(self.predict(adv_data))
            tp = sum(adv_predict[:])
            fn = sum(~adv_predict[:])
            tot_tp+=tp
            tot_fn+=fn

            if not group: # calculate together later
                continue
            # if group, calculate matrix for the class now
            # check num positive = negative
            assert tp+fn == fp+tn

            acc = (tp+tn)/(tp+fn+fp+tn)
            recall = tp/(tp+fn)
            precision = tp/(tp+fp)
            f1 = 2*precision*recall/(precision+recall)
            fpr = fp/(fp+tn)
            results["constraint"].append(adv_class.split("_")[-1])
            results["accuracy"].append(acc)
            results["recall"].append(recall)
            results["precision"].append(precision)
            results["f1"].append(f1)
            results["classification threshold"].append(self.threshold)
            results["fpr"].append(fpr)  
        if not group:
            assert tot_tp+tot_fn==fp+tn
            tp = tot_tp
            fn = tot_fn
            acc = (tp+tn)/(tp+fn+fp+tn)
            recall = tp/(tp+fn)
            precision = tp/(tp+fp)
            f1 = 2*precision*recall/(precision+recall)
            fpr = fp/(fp+tn)
            results["constraint"].append("all")
            results["accuracy"].append(acc)
            results["recall"].append(recall)
            results["precision"].append(precision)
            results["f1"].append(f1)
            results["classification threshold"].append(self.threshold)
            results["fpr"].append(fpr)  
        results = pd.DataFrame(results)
        return results

'''
def test_defender_on_validation_set():
    """
    test the defender on validation set and save the results
    """
    f = "MLM/src/intermediate-data/similarity_matrix_clean_test.csv"
    df = pd.read_csv(f)
    # get the clean image data
    clean_header = [col for col in df.columns if "clean_resized" in col]
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
        cleantest_header = [col for col in df.columns if "clean_test" in col]
        # get data with denoise time leq than 350
        adv_data = []
        for adv_header in [adv16_header, adv32_header, adv64_header, advucon_header, cleantest_header]:
            h_list=[]
            for h in adv_header:
                t = int(h.split("_")[-1].rstrip("times"))
                if t <= 350:
                    h_list.append(h)
                else:
                    break
            # get adversarial data of 16, 32, 64 constraint and unconstrained
            adv_data.append(df[h_list])

        constraint_names = [
            "16","32","64","unconstrainted","clean test"
        ]
        # predict whether it is adversarial **image**
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
    p = "MLM/src/analysis/defender_clean_test_results.csv"
    results.to_csv(p, index=False)
    print(f"Results saved to {p}")
    # print 4 tables separately
    print("data percentage: 0.95")
    print(results[results["data percentage"]==0.95])
    print("data percentage: 0.975")
    print(results[results["data percentage"]==0.975])
    print("data percentage: 0.99")
    print(results[results["data percentage"]==0.99])
    print("data percentage: 0.995")
    print(results[results["data percentage"]==0.995])

def test_effect_on_clean_image():
    """
    test the defender with a new clean image(effect on performance) with malicious text
    result only include accuracy
    """
    f = "MLM/src/intermediate-data/similarity_matrix_clean_test.csv"
    df = pd.read_csv(f)
    # get the clean image data
    clean_header = [col for col in df.columns if "clean_resized" in col]
    if len(clean_header)>8:
        clean_header = clean_header[:8]
    clean_data = df[clean_header]

    results = {
        "data percentage":[],
        "constraint":[],
        "accuracy":[],
        "classification threshold":[]
    }
    for threshold in [0.95, 0.975, 0.99, 0.995]:
        # train the defender on clean data and keep 95%
        d = Defender()
        d.train(clean_data, threshold)

        # test the defender on adversarial data
        cleantest_header = [col for col in df.columns if "clean_test" in col]
        # get data with denoise time leq than 350
        adv_data = []
        for adv_header in [cleantest_header]:
            h_list=[]
            for h in adv_header:
                t = int(h.split("_")[-1].rstrip("times"))
                if t <= 350:
                    h_list.append(h)
                else:
                    break
            # get adversarial data of 16, 32, 64 constraint and unconstrained
            adv_data.append(df[h_list])

        constraint_names = [
            "clean test"
        ]
        # predict whether it is adversarial
        for j in range(len(adv_data)):
            adv_predict = d.predict(adv_data[j])
            adv_predict = np.array(adv_predict)
            fp = sum(adv_predict[:40])
            acc = (40-fp)/40
            # print(f"Adversarial group: {constraint_names[j]}")
            # print(f"Accuracy: {acc}, Recall: {recall}, Precision: {precision}, F1: {f1}")
            results["data percentage"].append(threshold)
            results["constraint"].append(constraint_names[j])
            results["accuracy"].append(acc)
            results["classification threshold"].append(d.threshold)
    # save the results
    results = pd.DataFrame(results)
    p = "MLM/src/analysis/defender_clean_test_results.csv"
    results.to_csv(p, index=False)
    print(f"Results saved to {p}")
    # print 4 tables separately
    print("data percentage: 0.95")
    print(results[results["data percentage"]==0.95])
    print("data percentage: 0.975")
    print(results[results["data percentage"]==0.975])
    print("data percentage: 0.99")
    print(results[results["data percentage"]==0.99])
    print("data percentage: 0.995")
    print(results[results["data percentage"]==0.995])

def test_clean_vs_adversarial_on_malicious():
    """
    test the defender on another clean img v.s. adversarial image with malicious text 
    to determin its performance (with malicious text input)
    true positive: adv, true negative: clean image
    """
    f = "MLM/src/intermediate-data/similarity_matrix_clean_test.csv"
    df = pd.read_csv(f)
    df = df[df["is_malicious"]==1] # only test on malicious text
    # get the clean image data
    clean_header = [col for col in df.columns if "clean_resized" in col]
    if len(clean_header)>8:
        clean_header = clean_header[:8]
    clean_data = df[clean_header]

    results = {
        "data percentage":[],
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
        cleantest_header = [col for col in df.columns if "clean_test" in col]
        # get data with denoise time leq than 350
        adv_data = []
        for adv_header in [adv16_header, adv32_header, adv64_header, advucon_header, cleantest_header]:
            h_list=[]
            for h in adv_header:
                t = int(h.split("_")[-1].rstrip("times"))
                if t <= 350:
                    h_list.append(h)
                else:
                    break
            # get adversarial data of 16, 32, 64 constraint and unconstrained
            adv_data.append(df[h_list])

        # predict whether it is adversarial with real adversarial image
        tp,fn=0,0
        for j in range(len(adv_data)-1):
            adv_predict = d.predict(adv_data[j])
            adv_predict = np.array(adv_predict)
            tp += sum(adv_predict[:])
            fn += sum(~adv_predict[:])
        # predict with clean image
        clean_predict = np.array(d.predict(adv_data[-1]))
        fp = sum(clean_predict[:])
        tn = sum(~clean_predict[:])
        
        acc=(tp+tn)/(tp+fn+fp+tn)
        recall=tp/(tp+fn)
        precision=tp/(tp+fp)
        f1=2*precision*recall/(precision+recall)
        # print(f"Adversarial group: {constraint_names[j]}")
        # print(f"Accuracy: {acc}, Recall: {recall}, Precision: {precision}, F1: {f1}")
        results["data percentage"].append(threshold)
        results["accuracy"].append(acc)
        results["recall"].append(recall)
        results["precision"].append(precision)
        results["f1"].append(f1)
        results["classification threshold"].append(d.threshold)
    # save the results
    results = pd.DataFrame(results)
    p="MLM/src/analysis/defender_clean_test_results.csv"
    results.to_csv(p, index=False)
    print(f"Results saved to {p}")
    # print 4 tables separately
    print("data percentage: 0.95")
    print(results[results["data percentage"]==0.95])
    print("data percentage: 0.975")
    print(results[results["data percentage"]==0.975])
    print("data percentage: 0.99")
    print(results[results["data percentage"]==0.99])
    print("data percentage: 0.995")
    print(results[results["data percentage"]==0.995])

def test_defender_on_test_set():
    """
    test the defender on test set and save the results
    """
    # get data from validation to train the model
    f = "MLM/src/intermediate-data/similarity_matrix_validation.csv"
    train_df = pd.read_csv(f)
    # get the clean image data
    clean_header = [col for col in train_df.columns if "clean_resized" in col]
    if len(clean_header)>8:
        clean_header = clean_header[:8]
    clean_data = train_df[clean_header]

    # read test data
    f2 = "MLM/src/intermediate-data/similarity_matrix_test.csv"
    df = pd.read_csv(f2)

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
        cleantest_header = [col for col in df.columns if "clean_test" in col]
        # get data header with denoise time leq than 350
        adv_data = []
        for adv_header in [adv16_header, adv32_header, adv64_header, advucon_header, cleantest_header]:
            h_list=[]
            for h in adv_header:
                t = int(h.split("_")[-1].rstrip("times"))
                if t <= 350:
                    h_list.append(h)
                else:
                    break
            # get data
            adv_data.append(df[h_list])

        constraint_names = [
            "16","32","64","unconstrainted","clean test"
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
    p="MLM/src/analysis/defender_TestSet_results.csv"
    results.to_csv(p, index=False)
    print(f"Results saved to {p}")
    # print 4 tables separately
    print("data percentage: 0.95")
    print(results[results["data percentage"]==0.95])
    print("data percentage: 0.975")
    print(results[results["data percentage"]==0.975])
    print("data percentage: 0.99")
    print(results[results["data percentage"]==0.99])
    print("data percentage: 0.995")
    print(results[results["data percentage"]==0.995])

def test_defender_on_malicious_test_set():
    """
    test the defender on new clean img v.s. adversarial image with test set malicious text 
    to determin its performance (with malicious text input) of distinguishing images
    true positive: adv image, true negative: clean image
    """
    # get data from validation to train the model
    f = "MLM/src/intermediate-data/similarity_matrix_validation.csv"
    df = pd.read_csv(f)
    df = df[df["is_malicious"]==1] # only test on malicious text
    # get the clean image data
    clean_header = [col for col in df.columns if "clean_resized" in col]
    if len(clean_header)>8:
        clean_header = clean_header[:8]
    clean_data = df[clean_header]

    # read test data
    f2 = "MLM/src/intermediate-data/similarity_matrix_test.csv"
    df = pd.read_csv(f2)

    results = {
        "data percentage":[],
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
        cleantest_header = [col for col in df.columns if "clean_test" in col]
        # get data header with denoise time leq than 350
        adv_data = []
        for adv_header in [adv16_header, adv32_header, adv64_header, advucon_header, cleantest_header]:
            h_list=[]
            for h in adv_header:
                t = int(h.split("_")[-1].rstrip("times"))
                if t <= 350:
                    h_list.append(h)
                else:
                    break
            # get data
            adv_data.append(df[h_list])

        # predict whether the image is adversarial
        tp,fn=0,0
        # predict on adversarial image
        for j in range(len(adv_data)-1): 
            adv_predict = d.predict(adv_data[j])
            adv_predict = np.array(adv_predict)
            tp += sum(adv_predict[:])
            fn += sum(~adv_predict[:])
        # predict on clean image
        clean_predict = np.array(d.predict(adv_data[-1]))
        fp = sum(clean_predict[:])
        tn = sum(~clean_predict[:])
        
        acc=(tp+tn)/(tp+fn+fp+tn)
        recall=tp/(tp+fn)
        precision=tp/(tp+fp)
        f1=2*precision*recall/(precision+recall)
        # print(f"Adversarial group: {constraint_names[j]}")
        # print(f"Accuracy: {acc}, Recall: {recall}, Precision: {precision}, F1: {f1}")
        results["data percentage"].append(threshold)
        results["accuracy"].append(acc)
        results["recall"].append(recall)
        results["precision"].append(precision)
        results["f1"].append(f1)
        results["classification threshold"].append(d.threshold)
    # save the results
    results = pd.DataFrame(results)
    p="MLM/src/analysis/defender_TestSet_results.csv"
    results.to_csv(p, index=False)
    print(f"Results saved to {p}")
    # print 4 tables separately
    print("data percentage: 0.95")
    print(results[results["data percentage"]==0.95])
    print("data percentage: 0.975")
    print(results[results["data percentage"]==0.975])
    print("data percentage: 0.99")
    print(results[results["data percentage"]==0.99])
    print("data percentage: 0.995")
    print(results[results["data percentage"]==0.995])

'''

def test_imgdetector(datapath:str,savepath:str,cpnum=8,data_rate=[0.95, 0.975, 0.99, 0.995]):
    # path = "./src/intermediate-data/similarity_matrix_validation.csv" # load training data
    path = "./src/intermediate-data/10clean_similarity_matrix_val.csv"
    data = pd.read_csv(path)
    data = data[data["is_malicious"]==1] # only consider malicious text
    # train_data = data[[col for col in data.columns if "clean_resized" in col]][:cpnum]
    train_data = []
    for i in range(10):
        train_data.append(data.iloc[:,i*8:i*8+8])
    detector = Defender()
    results = []
    for i in data_rate:
        detector.train(train_data,ratio=i)
        # print(f"Threshold: {detector.threshold}")
        # predict on test data and return results
        confusion = detector.get_confusion_matrix(datapath,checkpt_num=cpnum,group=False)
        confusion["data percentage"] = i
        print(confusion)
        results.append(confusion)
    results = pd.concat(results).sort_values(by=["data percentage","constraint"])
    results.to_csv(savepath, index=False)
    
def plot_tpr_fpr(datapath:str, savepath:str, cpnum=8):
    # training data
    path = "./src/intermediate-data/10clean_similarity_matrix_val.csv"
    data = pd.read_csv(path)
    data = data[data["is_malicious"]==1] # only consider malicious text
    train_data = []
    for i in range(10):
        train_data.append(data.iloc[:,i*8:i*8+8])
    detector = Defender()

    datapoints = []
    for i in range(80,101):
        detector.train(train_data,ratio=i/100)
        # predict on test data and return results
        confusion = detector.get_confusion_matrix(datapath,checkpt_num=cpnum,group=False)
        # delete the row of inf
        confusion = confusion[confusion["constraint"]!="inf"]
        point = confusion[["recall","fpr"]].mean(axis=0)
        if i==95:
            chosen_pt=point
        else:
            datapoints.append(point)
    datapoints = pd.concat(datapoints, axis=1, ignore_index=True)

    # plot
    plt.clf()
    plt.figure(figsize=(4.4,3.2))

    # plot the curve 
    func = lambda x: 1.08131*x**0.14027
    x = np.arange(0,0.5,0.005)
    y = [func(x1) for x1 in x]
    plt.plot(x,y,color="gray",alpha=1,linestyle='--',linewidth=4,zorder=1)


    # plt.xlim((0,1))
    # plt.ylim((0,1))
    plt.scatter(datapoints.loc["fpr"],datapoints.loc["recall"],s=80,c="#23bac5",alpha=0.7,zorder=2)
    plt.scatter(chosen_pt[1],chosen_pt[0],s=150,c="#fd763f",marker="X",
                linewidth=1,edgecolors='w',zorder=3)
    plt.annotate("95%",(chosen_pt[1],chosen_pt[0]),xytext=(chosen_pt[1]-0.02,chosen_pt[0]+0.05),xycoords="data")
    # for i in range(80,101):
    #     plt.annotate(str(i),(datapoints.loc["fpr"][i-80],datapoints.loc["recall"][i-80]))
    plt.xlabel("False Positive Rate",fontsize=11)
    plt.ylabel("True Positive Rate",fontsize=11)

    plt.yticks(np.arange(0,1,0.2),['{:.0%}'.format(_) for _ in np.arange(0,1,0.2)])
    plt.xticks(np.arange(0,0.5,0.1),['{:.0%}'.format(_) for _ in np.arange(0,0.5,0.1)])

    

    plt.tight_layout()
    plt.savefig(savepath)
    plt.savefig(savepath.rstrip(".jpg")+".pdf")
    print(datapoints.loc["fpr"],datapoints.loc["recall"])

    

if __name__ == "__main__":
    plot_tpr_fpr(datapath="./src/intermediate-data/4clean_similarity_matrix_val.csv",
                    savepath="./src/analysis/ValSet_tpr-fpr_plot(new).jpg")
    # plot_tpr_fpr(datapath="./src/intermediate-data/4clean_similarity_matrix_test.csv",
    #                 savepath="./src/analysis/TestSet_tpr-fpr_plot.jpg")
    # test_imgdetector(datapath="./src/intermediate-data/4clean_similarity_matrix_val.csv",
    #                  savepath="./src/analysis/10clean_imgdetector_ValSet_results.csv",
    #                  data_rate=[0.95, 0.975, 0.99, 0.995])
    # test_imgdetector(datapath="./src/intermediate-data/4clean_similarity_matrix_test.csv",
    #                  savepath="./src/analysis/10clean_imgdetector_TestSet_results.csv",
    #                  data_rate=[0.95, 0.975, 0.99, 0.995])
