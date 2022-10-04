# Load necessary packages
from os import SEEK_END
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

# set seed
seed = 12

# Read in appropriate data
data1 = pd.read_csv("../admissionNew.csv")
matchedData2 = pd.read_csv("../matching/matched.csv")

# Add matching column to know which rows were matched
data1["matched"] = matchedData2["Unnamed: 0"]

# Drop rows that were not able to find matches
missingReplaceValue = max(data1["matched"])
data1["matched"] = data1["matched"].replace(missingReplaceValue,np.nan)
data1 = data1.dropna()
matchedData2 = matchedData2.dropna()

# Drop columns that were used for matching
realData = data1.drop(["matched"], axis = 1)
idealData = matchedData2.drop(["Unnamed: 0"], axis = 1)

# Seperate data for modeling
realX = realData.iloc[:,:-1]
realY = realData.iloc[:,-1]
idealX = idealData.iloc[:,:-1]
idealY = idealData.iloc[:,-1]

# Encode data into numerical for modeling
category_names = dict()
categories = ["Sex", "Race"]
for category in categories:
    onehot = pd.get_dummies(realX[category])
    realX = realX.drop(category,axis = 1)
    onehot.columns = onehot.columns.values + "_" + category
    realX = realX.join(onehot)
    onehot = pd.get_dummies(idealX[category])
    idealX = idealX.drop(category,axis = 1)
    onehot.columns = onehot.columns.values + "_" + category
    idealX = idealX.join(onehot)
    # Store category names for future use
    category_names[category] = onehot.columns.values
# Encode label into categorical where Yes is 1 and No is 0
realY = realY.astype("category")
realY = realY.cat.codes
idealY = idealY.astype("category")
idealY = idealY.cat.codes

# Train logistic regression
log_reg_ideal = LogisticRegression(random_state=seed).fit(idealX,idealY)
log_reg_real = LogisticRegression(random_state=seed).fit(realX,realY)

# Get logistic regression predictions
ideal_predictions = log_reg_ideal.predict(idealX)
real_predictions = log_reg_real.predict(realX)

# Get metric helper functions
def get_precision(true,predicted):
    return precision_score(true,predicted,average = 'binary')

def get_recall(true,predicted):
    return recall_score(true,predicted,average = 'binary')

def get_auc(true,predicted):
    fpr, tpr, thresholds = metrics.roc_curve(true, predicted)
    return metrics.auc(fpr,tpr)

def get_FPR_FNR(true,predicted):
    tn, fp, fn, tp = confusion_matrix(true, predicted).ravel()
    fpr = fp/(fp+tn)
    fnr = fn/(tp+fn)
    return (fpr,fnr)

def get_selectionRate(predicted):
    return np.sum(predicted == 1)/len(predicted)

# Get race metrics comparing to ideal Y given training data used
def metric_report(train_data, predicted_labels):
    # Metrics for Race
    race_column_names = category_names["Race"]
    for col in race_column_names:
        print(col[:-5] + ":")
        subideal = idealY[train_data[col] == 1]
        sub_predictions = predicted_labels[train_data[col] == 1]
        print("  Precision=" + str(round(get_precision(subideal,sub_predictions),4)))
        print("  Recall=" + str(round(get_recall(subideal,sub_predictions),4)))
        fpr,fnr = get_FPR_FNR(subideal,sub_predictions)
        print("  FPR=" + str(round(fpr,4)))
        print("  FNR=" + str(round(fnr,4)))
        print("  AUC=" + str(round(get_auc(subideal,sub_predictions),4)))
        print("  Selection Rate=" + str(round(get_selectionRate(sub_predictions),4)))
    # Metrics for Sex
    sex_column_names = category_names["Sex"]
    for col in sex_column_names:
        print(col[:-4]+":")
        subideal = idealY[train_data[col] == 1]
        sub_predictions = predicted_labels[train_data[col] == 1]
        print("  Precision=" + str(round(get_precision(subideal,sub_predictions),4)))
        print("  Recall=" + str(round(get_recall(subideal,sub_predictions),4)))
        fpr,fnr = get_FPR_FNR(subideal,sub_predictions)
        print("  FPR=" + str(round(fpr,4)))
        print("  FNR=" + str(round(fnr,4)))
        print("  AUC=" + str(round(get_auc(subideal,sub_predictions),4)))
        print("  Selection Rate=" + str(round(get_selectionRate(sub_predictions),4)))
    # Metrics for Income
    income_categories = [0, 25000, 50000, 75000, 100000, 150000, 200000, 300000]
    for idx in range(len(income_categories)-1):
        low_inclusive = income_categories[idx]
        high_exclusive = income_categories[idx+1]
        print("Income between "+str(low_inclusive)+" and "+str(high_exclusive)+":")
        subideal = idealY[(train_data["Income"] >= low_inclusive) & (train_data["Income"] < high_exclusive)]
        sub_predictions = predicted_labels[(train_data["Income"] >= low_inclusive) & (train_data["Income"] < high_exclusive)]
        print("  Precision=" + str(round(get_precision(subideal,sub_predictions),4)))
        print("  Recall=" + str(round(get_recall(subideal,sub_predictions),4)))
        fpr,fnr = get_FPR_FNR(subideal,sub_predictions)
        print("  FPR=" + str(round(fpr,4)))
        print("  FNR=" + str(round(fnr,4)))
        print("  AUC=" + str(round(get_auc(subideal,sub_predictions),4)))
        print("  Selection Rate=" + str(round(get_selectionRate(sub_predictions),4)))
        

metric_report(idealX,ideal_predictions)
print("------------------------------")
metric_report(realX,real_predictions)