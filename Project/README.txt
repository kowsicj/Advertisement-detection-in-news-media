README

Introduction
There are three programs:- 
    1) DataSplitter.py - Program to split the Data Set into Training and Testing Folders
    2) Classsification.py - Program to Classify the Data Set
    3) Clustering.py - Program to Cluster the Data Set

	[Location of dataset, split dataset (for training and testing), pickle files : https://drive.google.com/open?id=0B-48PuIzSLhsVFBmOVNFaG5UTEk]
	
To Run DataSplitter.py
    python DataSplitter.py <folder containing Data Set with path>
    eg:- python DataSplitter.py "C:\Users\srina\Desktop\TV_News_Channel_Commercial_Detection_Dataset"

Sample Output
    Reading Files ...
    Writing Training Features ...
    Writing Testing Features ...
    Writing Training Features WithOut Bag Of Words ...
    Writing Testing Features WithOut Bag Of Words ...

To Run Classification.py
    python Classification.py <folder containing Data Set with path>
    eg:- python DataSplitter.py "C:\Users\srina\Desktop\TV_News_Channel_Commercial_Detection_Dataset"

Note: Either run DataSplitter.py before running this program or download the splitted data from the shared google drive

Sample Output
    Choose the Data Set 
     1: With All Features 
     2: With Out Bag Of Words Features
    2
    Reading in the Data ...
    Choose the Model 
     1: Random Forest 
     2: Decision Tree 
     3: AdaBoost 
    1
    Generating Model ...
    Enter the number of n_estimators
    10
    Fitting the Model ...
    Choose Testing and Traning Models 
     1: Train as Test Data 
     2: Test as Test Data
    2
    Predicting the Output
    Generating Results
    Precision: 92.9146027342976 Recall: 95.05412089025825 F-1 Measure: 93.97218548354775

To Run Clustering.py
    python Clustering.py <folder containing Data Set with path>
    eg:- python DataSplitter.py "C:\Users\srina\Desktop\TV_News_Channel_Commercial_Detection_Dataset"

Note: Either run DataSplitter.py before running this program or download the splitted data from the shared google drive

Sample Output
    Reading in the Data ...
    precision
    0.6449805843783818
    recall
    0.9857109849083678
    f1-measure
    0.7797482479810296
