import csv
import sys
import random
import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
#from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
#from sklearn.metrics import mean_squared_error

TEST_SIZE = 0.4

def main():     
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predicti  ons
    for m in ['knn','svm','LogisticRegression','DecisionTree','RandomForest','NaiveBayes','Bagging']:    
        model = train_model(X_train, y_train,method = m)
        predictions = model.predict(X_test)
        sensitivity, specificity = evaluate(y_test, predictions)
        # Print results
        print('With training model: ',m)
        print(f"Correct: {(y_test == predictions).sum()}")
        print(f"Incorrect: {(y_test != predictions).sum()}")
        print(f"Accuracy: {(y_test == predictions).sum()/len(y_test)}")
        print(f"True Positive Rate: {100 * sensitivity:.2f}%")
        print(f"True Negative Rate: {100 * specificity:.2f}%")

    for m in ['knn','svm','LogisticRegression','DecisionTree','RandomForest','NaiveBayes','Bagging']:            
        if m == 'knn':
            #k
            ps = range(1,10)
        elif m == 'DecisionTree':
            #max_depth
            ps = range(1,10)
        elif m == 'RandomForest':
            #random
            ps = range(1,10)         
        elif m == 'NaiveBayes':
            ps = ['Gaussian','Bernouli','Multinomial']
        elif m == 'Bagging':
            ps = range(1,10)
        elif m == 'svm':
            #kernel
            ps = ['rbf','poly','sigmoid']
        elif m == 'LogisticRegression':
            #C
            ps = np.logspace(-2,4,10)
        
        accuracy_scores = []
        difference_scores = []
        minimumD = 100000
        maximalA = 0
        for p in ps:
            scores = Cross_Validation(evidence, labels, m, p)
            accuracy_scores.append(scores.mean())
            print('check:',p)
            print(np.mean(Plot_learning_curve(evidence, labels, m ,p,plot = False)))
            difference_scores.append(np.mean(Plot_learning_curve(evidence, labels, m ,p,plot = False)))
            if np.mean(Plot_learning_curve(evidence, labels, m ,p,plot = False)) <= minimumD:
                minimumD = np.mean(Plot_learning_curve(evidence, labels, m ,p,plot = False))
                miniP = p
            if scores.mean()>= maximalA:
                maximalA = scores.mean()
                maxiP = p
        # Print results
        print('With cross-validation checked by accuracy and learning_cerve: ',m)
        print("Accuracy:",accuracy_scores)
        print(f"Best: {100*maximalA:.2f}% at {maxiP}")
        print("Error:",difference_scores)
        print(f"Best: {minimumD:.2f} at {miniP}")

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
      0 int64 - Administrative, an integer
      1 float64 - Administrative_Duration, a floating point number
      2 int64 - Informational, an integer
      3 float64 - Informational_Duration, a floating point number
      4 int64 - ProductRelated, an integer
      5 float64 - ProductRelated_Duration, a floating point number
      6 float64 - BounceRates, a floating point number
      7 float64 - ExitRates, a floating point number
      8 float64 - PageValues, a floating point number
      9 float64 - SpecialDay, a floating point number
     *10 0 - Month, an index from 0 (January) to 11 (December)
      11 int64 - OperatingSystems, an integer
      12 int64 - Browser, an integer
      13 int64 - Region, an integer
      14 int64 - TrafficType, an integer
     *15 0 - VisitorType, an integer 0 (not returning) or 1 (returning)
     *16 bool - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    *is 1 if Revenue is true, and 0 otherwise. 17 bool
    """
    try:
        with open(filename) as f:
            '''reader = csv.reader(f)
            next(reader)
        
            data = []
            for row in reader:
                data.append({
                    "evidence": [float(cell) for cell in row[:16]],
                    "label": 0 if row[17] == "false" else 1
                })'''
            df = pd.read_csv(f)
            df1 = df.copy()
        evidence = []
        label = []
        Month2Num = {'Jan':0,'Feb':1,'Mar':2,'Apr':3,'May':4,'June':5,'Jul':6,'Aug':7,'Sep':8,'Oct':9,'Nov':10,'Dec':11}
#        print(np.shape(df1))
        for nRow in range(np.shape(df1)[0]):
            temp = []            
            for nCol in range(np.shape(df1)[1]):
                if nCol == 10:
                    temp.append(int(Month2Num[df1.iloc[nRow,nCol]]))
#                    print('tempcheck',nCol,'month',df1.iloc[nRow,nCol],int(Month2Num[df1.iloc[nRow,nCol]]),temp[-1])
                elif nCol == 15:
                    temp.append(int(1 if df1.iloc[nRow,nCol]== "Returning_Vistor" else 0))
                elif nCol == 16:
                    temp.append(int(0 if df1.iloc[nRow,nCol] == False else 1))
                elif nCol == 17:
                    label.append(int(0 if df1.iloc[nRow,nCol] == False else 1))
                else:
                    temp.append(df1.iloc[nRow,nCol])
            evidence.append(temp)
 #           print(evidence[-1])
        return (evidence,label)
    except:    
        raise NotImplementedError

def train_model(evidence, labels,method = 'knn'):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    
    To compapre, there can be knn(k=5),svm,
    """
    try:
        model = ''
        if method == 'knn':
            model = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
        elif method == 'svm':                  
            model = svm.SVC(kernel = 'rbf',gamma='auto') #‘linear’
        elif method == 'LogisticRegression':
            model = LogisticRegression(C=1.0,n_jobs = -1)
        elif method == 'DecisionTree':
            model = DecisionTreeClassifier(criterion ='gini',splitter = 'best', max_depth = 5)
        elif method == 'RandomForest':    
            model = RandomForestClassifier(n_estimators=10, random_state=4,n_jobs=-1)
        elif method == 'NaiveBayes':
            model = GaussianNB()
        elif method == 'Bagging':
            model = BaggingClassifier(n_estimators=10, random_state=4 ,n_jobs=-1)    
        return model.fit(evidence, labels)
    except:
        raise NotImplementedError

def Cross_Validation(evidence, labels, method ,parameter):
    """
    Given a list of evidence lists and a list of labels, return a score_list with different parameters
    cv = 10, scoring = accuracy
    To compapre, there can be knn(k=),svm,
    """
    try:
        if method == 'knn':
            model = KNeighborsClassifier(parameter,n_jobs =  -1)
        elif method == 'svm':                  
            model = svm.SVC(kernel = parameter) #‘linear’
        elif method == 'LogisticRegression':
            model = LogisticRegression(C=parameter,n_jobs =  -1)
        elif method == 'DecisionTree':
            model = DecisionTreeClassifier(criterion ='gini',splitter = 'best', max_depth = parameter,n_jobs = -1)
        elif method == 'RandomForest':    
            model = RandomForestClassifier(n_estimators=parameter, random_state=4,n_jobs = -1)
        elif method == 'NaiveBayes':
            if parameter == 'Gaussian':
                model = GaussianNB()
            if parameter == 'Bernouli':
                model = BernoulliNB()
            if parameter == 'Multinomia':
                model = MultinomialNB()
        elif method == 'Bagging':
            model = BaggingClassifier(n_estimators=parameter, random_state=4,n_jobs =  -1)    
        scores = cross_val_score(model, evidence, labels, cv = 10, scoring = 'accuracy',n_jobs =  -1)
        return scores
    except:
        raise NotImplementedError

def Plot_learning_curve(evidence, labels, method ,parameter,plot = True):
    """
    Given a list of evidence lists and a list of labels, return a score_list with different parameters
    cv = 10, scoring = accuracy
    To compapre, there can be knn(k=),svm,
    """
    try:
        if method == 'knn':
            model = KNeighborsClassifier(n_neighbors=parameter,n_jobs=-1)
        elif method == 'svm':                  
            model = svm.SVC(kernel = parameter,gamma = 'auto') #‘linear’
        elif method == 'LogisticRegression':
            model = LogisticRegression(C=parameter,n_jobs = -1)
        elif method == 'DecisionTree':
            model = DecisionTreeClassifier(criterion ='gini',splitter = 'best', max_depth = parameter,n_jobs =  -1)
        elif method == 'RandomForest':    
            model = RandomForestClassifier(n_estimators=parameter, random_state=4,n_jobs =  -1)
        elif method == 'NaiveBayes':
            if parameter == 'Gaussian':
                model = GaussianNB()
            if parameter == 'Bernouli':
                model = BernoulliNB()
            if parameter == 'Multinomia':
                model = MultinomialNB()
        elif method == 'Bagging':
            model = BaggingClassifier(n_estimators=parameter, random_state=4,n_jobs =  -1)    
        train_sizes, train_scores, test_scores = learning_curve(model, evidence, labels, cv=10, train_sizes=np.linspace(0.5,0.95,10),n_jobs =  -1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        differences = abs(train_scores_mean-test_scores_mean)
        if plot:
            plt.figure()
            plt.title(method)
            plt.xlabel(u"train_sample")
            plt.ylabel(u"score")
    #        plt.gca().invert_yaxis()
    #        plt.grid()
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,alpha=0.1, color="b")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="r")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"train_score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"cross_validation_score")
            plt.legend(loc="best")
            plt.draw()
            plt.show()
            plt.gca().invert_yaxis()
            plt.savefig(method+"with parameter as"+str(parameter)+"learn_curve.jpg") 
        return differences
    except:
        raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    try: 
         sensitivity = 0
         specificity = 0
         for i in range(len(labels)):
             if labels[i] == 1 and predictions[i] == 1:
                 sensitivity += 1
             elif labels[i] == 0 and predictions[i] == 0:
                 specificity += 1
         return(sensitivity/len(labels),specificity/len(labels))
    except:
        raise NotImplementedError


if __name__ == "__main__":
    main()
