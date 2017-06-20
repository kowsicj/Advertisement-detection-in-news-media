import glob
import sys
import random
import os
import pickle

def dataSplitter(folderName, folderTrain, folderTest, fileList):
    '''
    method to split the Data Sets for Traing and Testing
    :param folderName: folder path for the Data Sets
    :param folderTrain: folder path for Traning Data
    :param folderTest: folder path for Testing Data
    :param fileList: List of files
    :return: None
    '''
    if not os.path.exists(folderTrain):
        os.mkdir(folderTrain)
    if not os.path.exists(folderTest):
        os.mkdir(folderTest)
    for file in fileList:
        fileName = (os.path.basename(file))
        with open(file, 'r') as fread:
            read_data = fread.readlines()
            positiveIndex = []
            negativeIndex = []

            for index in range(len(read_data)):
                if (read_data[index][0] is '1'):
                    positiveIndex.append(index)
                else:
                    negativeIndex.append(index)
            positiveIndex = list(positiveIndex)
            negativeIndex = list(negativeIndex)
            testPositiveIndex = [positiveIndex[i] for i in
                                 sorted(random.sample(range(len(positiveIndex)), int(len(positiveIndex) * 0.30)))]
            testNegativeIndex = [negativeIndex[i] for i in
                                 sorted(random.sample(range(len(negativeIndex)), int(len(negativeIndex) * 0.30)))]
            trainPostiveIndex = list(set(positiveIndex) - set(testPositiveIndex))
            trainNegativeIndex = list(set(negativeIndex) - set(testNegativeIndex))
            testIndices = testPositiveIndex + testNegativeIndex
            trainIndices = trainPostiveIndex + trainNegativeIndex
            fwrite = open(folderTest + "\\" + fileName.split('.')[0] + '.txt', 'w')
            for i in testIndices:
                fwrite.write(read_data[i])
            fwrite.close();
            fwrite = open(folderTrain + "\\" + fileName.split('.')[0] + '.txt', 'w')
            for i in trainIndices:
                fwrite.write(read_data[i])
            fwrite.close();

def readFeatureFile(folderName, featureFileName, outputFileName, flag=False):
    '''
    method to read the features from the files
    :param folderName: folder path for the Data Sets
    :param featureFileName: file name for the features file
    :param outputFileName: file name for the output file
    :param flag: flag to read Bag of Words
    :return: None
    '''
    fileList = glob.glob(folderName + "/*.txt")
    featureVector = []
    outputVector = []
    for file in fileList:
        with open(file, 'r') as fread:
            read_data = fread.readlines()
            for index in range(len(read_data)):
                line = read_data[index].strip()
                if flag:
                    lineFeature = [0] * 124
                else:
                    lineFeature = [0]*4125
                features = line.split(' ')
                counter = 1
                if(line[0] is '1'):
                    outputVector.append(1)
                else:
                    outputVector.append(-1)
                for fIndex in range(1,len(features)):
                    if(flag and (counter <= 122 or counter>=4124)):
                        if(features[fIndex] is not ''):
                            featureAttr = features[fIndex].split(':')
                            lineFeature[counter] = float(featureAttr[1])
                            counter += 1
                    elif(not flag):
                        if (features[fIndex] is not ''):
                            featureAttr = features[fIndex].split(':')
                            lineFeature[int(featureAttr[0]) - 1] = float(featureAttr[1])
                featureVector.append(lineFeature)
    pickle.dump(featureVector, open(folderName+'\\'+featureFileName, "wb"))
    if(not flag):
        pickle.dump(outputVector, open(folderName+'\\'+outputFileName, "wb"))

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
        fileList = glob.glob(folderName+"/*.txt")
        folderTrain = folderName + "\\Train"
        folderTest = folderName + "\\Test"
        print("Reading Files ...")
        dataSplitter(folderName,folderTrain,folderTest,fileList)
        print("Writing Training Features ...")
        readFeatureFile(folderTrain,'TrainFeature.p', 'TrainOutput.p')
        print("Writing Testing Features ...")
        readFeatureFile(folderTest, 'TestFeature.p', 'TestOutput.p')
        print("Writing Training Features WithOut Bag Of Words ...")
        readFeatureFile(folderTrain, 'TrainFeatureWOBOW.p', 'TrainOutput.p',True)
        print("Writing Testing Features WithOut Bag Of Words ...")
        readFeatureFile(folderTest, 'TestFeatureWOBOW.p', 'TestOutput.p',True)
main()