# adaboost binary classifier
# dataset: cancer.csv

# libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
#Visualisation 
import matplotlib.pyplot as plt
import seaborn as sns

#import csv file
data = pd.read_csv("cancer.csv")

data.head()
data.shape
data.info()

# check the Y-distribution
data.diagnosis.value_counts()

#EDA 

#id column is not seignificant drop column "id"

data.drop(columns = ["id"],inplace = True)
data.columns

#Check for the Nulls
data.isnull().sum()  #No Null Present

#Check for Zeros("0")

for c in data:
    print(c, " == ", data[c][data[c] == 0])
    
#Fixing Zeros

#concavity_mean
data.shape
data.concavity_mean.describe()

data.concavity_mean[data.concavity_mean == 0] = np.mean(data.concavity_mean)

#concave points_mean
data["concave points_mean"].describe()
data["concave points_mean"][data["concave points_mean"] == 0] = np.mean(data["concave points_mean"])


#concavity_se
data.concavity_se.describe()
data.concavity_se[data.concavity_se == 0] = concavity_worst

#concavity_worst
data.concavity_worst.describe()
data.concavity_worst[data.concavity_worst == 0] = np.mean(data.concavity_worst)

#concave points_se
data["concave points_se"].describe()
data["concave points_se"][data["concave points_se"] == 0] = np.mean(data["concave points_se"])

#concave points_worst
data["concave points_worst"].describe()

data["concave points_worst"][data["concave points_worst"] == 0] = np.mean(data["concave points_worst"])

#===================================================================
#Split data into numeri and factor
#b) Generate the description for numeric variab
nc = data.select_dtypes(exclude = "object").columns.values
fc = data.select_dtypes(include = "object").columns.values


#Check for Colinearity and Outlier
# function to plot the histogram, correlation matrix, boxplot based on the chart-type
def plotdata(data,nc,ctype):
    if ctype not in ['h','c','b']:
        msg='Invalid Chart Type specified'
        return(msg)
    
    if ctype=='c':
        cor = data[nc].corr()
        cor = np.tril(cor)
        sns.heatmap(cor,vmin=-1,vmax=1,xticklabels=nc,
                    yticklabels=nc,square=False,annot=True,linewidths=1)
    else:
        COLS = 2
        ROWS = np.ceil(len(nc)/COLS)
        POS = 1
        
        fig = plt.figure() # outer plot
        for c in nc:
            fig.add_subplot(ROWS,COLS,POS)
            if ctype=='b':
                sns.boxplot(data[c],color='yellow')
            else:
                sns.distplot(data[c],bins=20,color='green')
            
            POS+=1
    return(1)

#Correlation Matrix
plotdata(data,nc,'c')

#Features ShownigHigh colinearity
data.columns
fetr_drop = ['perimeter_mean','area_mean','concave points_mean','perimeter_se','area_se','radius_worst','texture_worst','perimeter_worst','area_worst','compactness_worst','concavity_worst','concave points_worst'] 


#make a copy of data
data_1 = data.copy()
#Drop columns having igher colinearity
data_1.drop(columns = fetr_drop,inplace =True)

# in data 'Unnamed: 32' feature has singualirity so drop that columns

data_1.drop(columns = ["Unnamed: 32"],inplace =True)

#refresh spliting f columns in data_1
nc = data_1.select_dtypes(exclude = "object").columns.values
fc = data_1.select_dtypes(include = "object").columns.values

len(data.columns)
len(data_1.columns)

#Check for outliers
plotdata(data,nc,'b') #data look fluctuatimg so there can be exrtrim alues present in data so all are valid outliers


# split data into train / test
trainx,testx,trainy,testy = train_test_split(data_1.drop('diagnosis',1),
                                             data.diagnosis,
                                             test_size=0.25)

print(trainx.shape,trainy.shape)
print(testx.shape, testy.shape)

# build the AdaBoost classifier model
# number of trees to build in the boosting process

trees = 50

m1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=trees).fit(trainx,trainy)

#Predict p1
p1 = m1.predict(testx)
p1

#Function for Confucion Matrix
def cm(actual,pred):
    
    # accuracy score
    print("Model Accuracy = {}".format(accuracy_score(actual,pred)))
    print("\n")
    
    # confusion matrix
    df = pd.DataFrame({'actual':actual, 'predicted':pred})
    print(pd.crosstab(df.actual, df.predicted, margins=True))
    print("\n")
    
    # classification report
    print(classification_report(actual,pred))
    
# model 1 evaluation
cm(testy,p1)

#=====================================================================

#Model M2 using feature selection 

#Important features
feat = pd.DataFrame({"feature":trainx.columns,"score":m1.feature_importances_})

feat = feat.sort_values('score',ascending = False)
print(feat)

#Selected Top 8 Features(Drop other featurs) 

data_1.drop(columns = ["compactness_mean","symmetry_worst","smoothness_mean","compactness_se","concave points_se","fractal_dimension_worst","fractal_dimension_mean","symmetry_mean","concavity_se","texture_se","smoothness_se"],inplace=True)

data_1.columns


# split data into train / test

trainx1,testx1,trainy1,testy1 = train_test_split(data_1.drop('diagnosis',1),
                                             data.diagnosis,
                                             test_size=0.25)

print(trainx1.shape,trainy1.shape)
print(testx1.shape, testy1.shape)

# build the AdaBoost classifier model
# number of trees to build in the boosting process

trees = 50

m2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=trees).fit(trainx1,trainy1)

#Predict p1
p2 = m2.predict(testx1)
p2

# model 2 evaluation
cm(testy1,p2)


#=====================================================================

#CONCLUSION : - #Ensemble Technique

'''
We 2 model model split ratiotaken for each model will be 25/75 

#Model 1 = Raw model (Model building perform after performing EDA and cleaning all the data) 

we got M1 = 94% accuracy 

              precision    recall  f1-score   support

           B       0.93      0.96      0.95        83
           M       0.95      0.90      0.92        60

    accuracy                           0.94       143
   macro avg       0.94      0.93      0.93       143
weighted avg       0.94      0.94      0.94       143



#Model 2 = perform after selecting only important features
we hae selected top 8 features 
we got M2 = 97% accuracy

              precision    recall  f1-score   support

           B       0.97      0.99      0.98        95
           M       0.98      0.94      0.96        48

    accuracy                           0.97       143
   macro avg       0.97      0.96      0.97       143
weighted avg       0.97      0.97      0.97       143


#After performing 2 Model we come to the conclusion that model 2 (M2) which is featire selection model is Optimum model because M2 has higher accuracy than M1 and in M2 there is slight improvement in classes. 
'''








