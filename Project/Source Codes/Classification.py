import sys
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

#Global Variables
featuresTrain , outputTrain, featuresTest, modelName = None, None, None, None
predictedOut, outputTest, model, modelFit = None, None, None, None

def loadData(folderName, fileType = ""):
    '''
    method to load the data sets
    :param folderName: Folder Path containing Data Set
    :param fileType: Data Set with or without Bad of Words
    :return None
    '''
    global featuresTrain, outputTrain, featuresTest, outputTest
    featuresTrain = pickle.load(open(folderName+"\\Train\\TrainFeature"+fileType+".p","rb"))
    outputTrain = pickle.load(open(folderName+"\\Train\\TrainOutput"+".p","rb"))
    featuresTest = pickle.load(open(folderName+"\\Test\\TestFeature"+fileType+".p","rb"))
    outputTest = pickle.load(open(folderName+"\\Test\\TestOutput"+".p","rb"))

def modelFit():
    '''
    method to perform classification
    :return None
    '''
    global model, featuresTrain, outputTrain, modelFit, modelName
    modelFit = model.fit(featuresTrain,outputTrain)
    pickle.dump(modelFit, open(modelName+".model", "wb"))

def modelPredict(feature):
    '''
    method to predict from the learned model
    :param feature: The testing Data to predict from
    :return None
    '''
    global modelFit, featuresTest, predictedOut
    predictedOut = modelFit.predict(feature)

def model(modelOption):
    '''
    method to load the data sets
    :param folderName: Folder Path containing Data Set
    :param fileType:  Data Set with or without Bad of Words
    :return None
    '''
    global model, modelName
    if(modelOption == 1):
        print("Enter the number of n_estimators")
        n_estimators = int(input())
        model = RandomForestClassifier(n_estimators=n_estimators)
        modelName="RandomForestClassifier"+str(n_estimators)
    elif(modelOption == 2):
        model = DecisionTreeClassifier()
        modelName = "DecisionTreeClassifier"
    else:
        model  = AdaBoostClassifier(
                    DecisionTreeClassifier(max_depth=2),
                    algorithm="SAMME")
        modelName = "AdaBoostClassifier"

def modelResult(testOption):
    '''
    method to print the Precision, Recall and F-1 Measure
    :param testOption: flag to run the tests
    :return: None
    '''
    global outputTest, outputTrain, predictedOut
    if(testOption==1):
        outputTest = outputTrain
    tp, fn, fp, tn = 0, 0, 0, 0
    for i in range(len(outputTest)):
        if (outputTest[i] == 1 and predictedOut[i] == 1):
            tp += 1
        elif (outputTest[i] == 1 and predictedOut[i] == -1):
            fn += 1
        elif (outputTest[i] == -1 and predictedOut[i] == 1):
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * tp) / (2 * tp + fp + fn)
    print("Precision: {} Recall: {} F-1 Measure: {}".format(precision*100,recall*100,f1*100))

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
        print("Choose the Data Set \n" +
              " 1: With All Features \n" +
              " 2: With Out Bag Of Words Features")
        dataOption = int(input())

        print("Reading in the Data ...")
        if(dataOption == 1):
            loadData(folderName)
        else:
            loadData(folderName, "WOBOW")

        print("Choose the Model \n 1: Random Forest \n"+
              " 2: Decision Tree \n 3: AdaBoost ")
        modelOption = int(input())
        print("Generating Model ...")

        model(modelOption)
        print("Fitting the Model ...")
        modelFit()
        print("Choose Testing and Traning Models \n"+
              " 1: Train as Test Data \n"+
              " 2: Test as Test Data")
        testOption = int(input())

        print("Predicting the Output")
        if(testOption == 1):
            modelPredict(featuresTrain)
        else:
            modelPredict(featuresTest)
        print("Generating Results")
        modelResult(testOption)
main()