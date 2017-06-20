import pickle
import sys
import numpy as np
from sklearn.cluster import KMeans

#Global Variables
featureAll, model, predictedOutput = None, None, None
outputAll = None

def loadData(folderName):
    '''
    method load the Data Sets
    :param folderName: folder path for Data Sets
    :return: None
    '''
    global featureAll, outputAll
    featureTest = pickle.load(open(folderName+"\\Test\\TestFeatureWOBOW.p", "rb"))
    featureTrain = pickle.load(open(folderName+"\\Train\\TrainFeatureWOBOW.p", "rb"))
    outputTest = pickle.load(open(folderName + "\\Test\\TestOutput.p", "rb"))
    outputTrain = pickle.load(open(folderName + "\\Train\\TrainOutput.p", "rb"))
    outputAll = outputTest + outputTrain
    featureAll = np.vstack((featureTest,featureTrain))


def model():
    '''
    method to build the  clustering model
    :return: None
    '''
    global model, predictedOutput
    model = KMeans(n_clusters=2)
    predictedOutput = model.fit_predict(featureAll)
    pickle.dump(model, open("MiniBatchKMeans.model", "wb"))

def evaluate():
    '''
    method to generate Precision, Recall and F-1 Measure
    :return: None
    '''
    global outputAll, predictedOutput
    tp, fn, fp, tn = 0, 0, 0, 0
    for i in range(len(outputAll)):
        if (outputAll[i] == 1 and predictedOutput[i] == 0):
            tp += 1
        elif (outputAll[i] == 1 and predictedOutput[i] == 1):
            fn += 1
        elif (outputAll[i] == -1 and predictedOutput[i] == 0):
            fp += 1
        else:
            tn += 1
    print("precision")
    precision = tp / (tp + fp)
    print(precision)
    print("recall")
    recall = tp / (tp + fn)
    print(recall)
    print("f1-measure")
    f1 = (2 * tp) / (2 * tp + fp + fn)
    print(f1)

def main():
    '''
    main method begins here
    :return: None
    '''
    if len(sys.argv) != 2:
        print("Invalid Arguments")
        return
    else:
        folderName = sys.argv[1]
        print("Reading in the Data ...")
        loadData(folderName)
        model()
        evaluate()
main()