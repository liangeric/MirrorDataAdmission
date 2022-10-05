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
    metrics = []
    # Metrics for Race
    race_column_names = category_names["Race"]
    for col in race_column_names:
        print(col[:-5] + ":")
        subideal = idealY[train_data[col] == 1]
        sub_predictions = predicted_labels[train_data[col] == 1]
        precision = round(get_precision(subideal,sub_predictions),4)
        metrics += [precision]
        print("  Precision=" + str(precision))
        recall = round(get_recall(subideal,sub_predictions),4)
        metrics += [recall]
        print("  Recall=" + str(recall))
        auc = round(get_auc(subideal,sub_predictions),4)
        metrics += [auc]
        print("  AUC=" + str(auc))
        fpr,fnr = get_FPR_FNR(subideal,sub_predictions)
        metrics += [round(fpr,4)]
        metrics += [round(fnr,4)]
        print("  FPR=" + str(round(fpr,4)))
        print("  FNR=" + str(round(fnr,4)))
        selection = round(get_selectionRate(sub_predictions),4)
        metrics += [selection]
        print("  Selection Rate=" + str(selection))
    # Metrics for Sex
    sex_column_names = category_names["Sex"]
    for col in sex_column_names:
        print(col[:-4]+":")
        subideal = idealY[train_data[col] == 1]
        sub_predictions = predicted_labels[train_data[col] == 1]
        precision = round(get_precision(subideal,sub_predictions),4)
        metrics += [precision]
        print("  Precision=" + str(precision))
        recall = round(get_recall(subideal,sub_predictions),4)
        metrics += [recall]
        print("  Recall=" + str(recall))
        auc = round(get_auc(subideal,sub_predictions),4)
        metrics += [auc]
        print("  AUC=" + str(auc))
        fpr,fnr = get_FPR_FNR(subideal,sub_predictions)
        metrics += [round(fpr,4)]
        metrics += [round(fnr,4)]
        print("  FPR=" + str(round(fpr,4)))
        print("  FNR=" + str(round(fnr,4)))
        selection = round(get_selectionRate(sub_predictions),4)
        metrics += [selection]
        print("  Selection Rate=" + str(selection))
        print("  Selection Rate=" + str(round(get_selectionRate(sub_predictions),4)))
    # Metrics for Income
    income_categories = [0, 25000, 50000, 75000, 100000, 150000, 200000, 300000]
    for idx in range(len(income_categories)-1):
        low_inclusive = income_categories[idx]
        high_exclusive = income_categories[idx+1]
        print("Income between "+str(low_inclusive)+" and "+str(high_exclusive)+":")
        subideal = idealY[(train_data["Income"] >= low_inclusive) & (train_data["Income"] < high_exclusive)]
        sub_predictions = predicted_labels[(train_data["Income"] >= low_inclusive) & (train_data["Income"] < high_exclusive)]
        precision = round(get_precision(subideal,sub_predictions),4)
        metrics += [precision]
        print("  Precision=" + str(precision))
        recall = round(get_recall(subideal,sub_predictions),4)
        metrics += [recall]
        print("  Recall=" + str(recall))
        auc = round(get_auc(subideal,sub_predictions),4)
        metrics += [auc]
        print("  AUC=" + str(auc))
        fpr,fnr = get_FPR_FNR(subideal,sub_predictions)
        metrics += [round(fpr,4)]
        metrics += [round(fnr,4)]
        print("  FPR=" + str(round(fpr,4)))
        print("  FNR=" + str(round(fnr,4)))
        selection = round(get_selectionRate(sub_predictions),4)
        metrics += [selection]
        print("  Selection Rate=" + str(selection))

    return metrics

# Get disparity for different groups above
def get_disparity(experiment,comparison):
    # compute disparities
    disparities = []
    for i in range(int(len(experiment)/len(comparison))):
        subExperiment = experiment[i*6:i*6+6]
        disparities += list(np.round(np.array(subExperiment)/np.array(comparison),4))

    # print out disparities
    race_column_names = category_names["Race"]
    for idx in range(len(race_column_names)):
        col = race_column_names[idx]
        print(col[:-5] + ":")
        print("  Precision Disparity=" + str(disparities[0+6*idx]))
        print("  Recall Disparity=" + str(disparities[1+6*idx]))
        print("  AUC Disparity=" + str(disparities[2+6*idx]))
        print("  FPR Disparity=" + str(disparities[3+6*idx]))
        print("  FNR Disparity=" + str(disparities[4+6*idx]))
        print("  Selection Rate Disparity=" + str(disparities[5+6*idx]))
    # Metrics for Sex
    sex_column_names = category_names["Sex"]
    for idx in range(len(sex_column_names)):
        col = sex_column_names[idx]
        print(col[:-4]+":")
        print("  Precision Disparity=" + str(disparities[36+6*idx]))
        print("  Recall Disparity=" + str(disparities[37+6*idx]))
        print("  AUC Disparity=" + str(disparities[38+6*idx]))
        print("  FPR Disparity=" + str(disparities[39+6*idx]))
        print("  FNR Disparity=" + str(disparities[40+6*idx]))
        print("  Selection Rate Disparity=" + str(disparities[41+6*idx]))
    # Metrics for Income
    income_categories = [0, 25000, 50000, 75000, 100000, 150000, 200000, 300000]
    for idx in range(len(income_categories)-1):
        low_inclusive = income_categories[idx]
        high_exclusive = income_categories[idx+1]
        print("Income between "+str(low_inclusive)+" and "+str(high_exclusive)+":")
        print("  Precision Disparity=" + str(disparities[54+6*idx]))
        print("  Recall Disparity=" + str(disparities[55+6*idx]))
        print("  AUC Disparity=" + str(disparities[56+6*idx]))
        print("  FPR Disparity=" + str(disparities[57+6*idx]))
        print("  FNR Disparity=" + str(disparities[58+6*idx]))
        print("  Selection Rate Disparity=" + str(disparities[59+6*idx]))
        

experiment_a = metric_report(idealX,ideal_predictions)
print("------------------------------")
experiment_b = metric_report(realX,real_predictions)
print("------------------------------")
comparisonGroup = [0.6721,0.8542,0.5634,0.7273,0.1458,0.8079]
get_disparity(experiment_b,comparisonGroup)