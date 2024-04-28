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
    
    def get_confusion_matrix(self, datapath:str,checkpt_num=8):
        """test the defender with given cosine similarity data, 
        save the statics to savepath\n
        only consider malicious text\n
        the result contains 4 rows: for each adversarial image, 
        consider it as positive and clean as negative, 
        output the results
        checkpt_num: the maximum number of denoise times checkpoint to consider"""
        df = pd.read_csv(datapath)
        df = df[df["is_malicious"]==1] # only consider malicious text input
        results = {
            "constraint":[],
            "accuracy":[],
            "recall":[],
            "precision":[],
            "f1":[],
            "classification threshold":[]
        }
        # get the Test Set clean image data for prediction
        clean_header = [col for col in df.columns if "clean_test" in col]
        if len(clean_header)>checkpt_num:
            clean_header = clean_header[:checkpt_num]
        clean_data = df[clean_header]
        # predict with clean image
        clean_predict = np.array(self.predict(clean_data))
        fp = sum(clean_predict[:])
        tn = sum(~clean_predict[:])
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
            acc = (tp+tn)/(tp+fn+fp+tn)
            recall = tp/(tp+fn)
            precision = tp/(tp+fp)
            f1 = 2*precision*recall/(precision+recall)
            results["constraint"].append(adv_class.split("_")[-1])
            results["accuracy"].append(acc)
            results["recall"].append(recall)
            results["precision"].append(precision)
            results["f1"].append(f1)
            results["classification threshold"].append(self.threshold)
        results = pd.DataFrame(results)
        return results


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

def test_imgdetector(datapath:str,savepath:str,cpnum=8):
    path = "./src/intermediate-data/similarity_matrix_validation.csv" # load training data
    data = pd.read_csv(path)
    data = data[data["is_malicious"]==1] # only consider malicious text
    train_data = data[[col for col in data.columns if "clean_resized" in col]][:cpnum]
    detector = Defender()
    results = []
    for i in [0.95, 0.975, 0.99, 0.995]:
        detector.train(train_data,ratio=i)
        # print(f"Threshold: {detector.threshold}")
        # predict on test data and return results
        confusion = detector.get_confusion_matrix(datapath,checkpt_num=cpnum)
        confusion["data percentage"] = i
        print(confusion)
        results.append(confusion)
    results = pd.concat(results).sort_values(by=["data percentage","constraint"])
    results.to_csv(savepath, index=False)
    

if __name__ == "__main__":
    test_imgdetector(datapath="./src/intermediate-data/similarity_matrix_test.csv",
                     savepath="./src/analysis/imgdetector_TestSet_results.csv")
    test_imgdetector(datapath="./src/intermediate-data/similarity_matrix_validation.csv",
                     savepath="./src/analysis/imgdetector_ValSet_results.csv")
