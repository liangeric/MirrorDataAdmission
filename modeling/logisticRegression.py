# Load necessary packages
from nis import match
from os import SEEK_END
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

# set seed
seed = 12
np.random.seed(seed)

# Read in appropriate data
data1 = pd.read_csv("../admissionNew.csv")
matchedData2 = pd.read_csv("../matching/matched.csv")

# Add matching column to know which rows were matched
data1["matched"] = matchedData2["Unnamed: 0"]

# Drop rows that were not able to find matches
missingReplaceValue = max(data1["matched"])
data1["matched"] = data1["matched"].replace(missingReplaceValue,np.nan)
matchedData2["Unnamed: 0"] = matchedData2["Unnamed: 0"].replace(missingReplaceValue,np.nan)
data1 = data1.dropna()
matchedData2 = matchedData2.dropna()

# Drop columns that were used for matching
realData = data1.drop(["matched"], axis = 1)
idealData = matchedData2.drop(["Unnamed: 0"], axis = 1)

# List of columns to drop that we don't want to include in model
drop_list = ["Opportunities","Academic Qualification","Non-Academic Qualification",
             "Diversity", "Intrinsic Abilities"]
realData = realData.drop(drop_list, axis = 1)
idealData = idealData.drop(drop_list, axis = 1)

# Seperate data for modeling
realX = realData.iloc[:,:-1]
realY = realData.iloc[:,-1]
idealX = idealData.iloc[:,:-1]
idealY = idealData.iloc[:,-1]

# Normalize non categorical data
category_names = dict()
categories = ["Sex", "Race"]
temp_categories = []
for category in categories:
    if category not in drop_list:
        temp_categories.append(category)
categories = temp_categories
realXcategorical = realX[categories]
idealXcategorical = idealX[categories]
realX = realX.drop(categories,axis = 1)
idealX = idealX.drop(categories,axis=1)
realXStd = realX.apply(lambda x: (x - x.mean())/np.std(x),axis = 0)
idealXStd = idealX.apply(lambda x: (x - x.mean())/np.std(x),axis = 0)

# Encode categorical data into one hot
for category in categories:
    onehot = pd.get_dummies(realXcategorical[category])
    onehot.columns = onehot.columns.values + "_" + category
    realX = realX.join(onehot)
    realXStd = realXStd.join(onehot)
    onehot = pd.get_dummies(idealXcategorical[category])
    onehot.columns = onehot.columns.values + "_" + category
    idealX = idealX.join(onehot)
    idealXStd = idealXStd.join(onehot)
    # Store category names for future use
    category_names[category] = onehot.columns.values
# Encode label into categorical where Yes is 1 and No is 0
realY = realY.astype("category")
realY = realY.cat.codes
idealY = idealY.astype("category")
idealY = idealY.cat.codes

# Set up cross validation
realX = realX.reset_index(drop = True)
idealX = idealX.reset_index(drop = True)
realXStd = realXStd.reset_index(drop = True)
idealXStd = idealXStd.reset_index(drop = True)
realY = realY.reset_index(drop = True)
idealY = idealY.reset_index(drop = True)
validation_percentage = 0.2
validation_size = round(len(realX) * validation_percentage)
validation_idx = np.random.choice(list(range(len(realX))), validation_size, replace = False)

realX_valid = realX.iloc[validation_idx]
realX_train = realX.drop(validation_idx,axis = 0)
idealX_valid = idealX.iloc[validation_idx]
idealX_train = idealX.drop(validation_idx,axis = 0)

realXStd_valid = realXStd.iloc[validation_idx]
realY_valid = realY.iloc[validation_idx]
realXStd_train = realXStd.drop(validation_idx,axis = 0)
realY_train = realY.drop(validation_idx, axis = 0)

idealXStd_valid = idealXStd.iloc[validation_idx]
idealY_valid = idealY.iloc[validation_idx]
idealXStd_train = idealXStd.drop(validation_idx,axis = 0)
idealY_train = idealY.drop(validation_idx, axis = 0)

# Train logistic regression
log_reg_ideal = LogisticRegression(random_state=seed).fit(idealXStd_train,idealY_train)
log_reg_real = LogisticRegression(random_state=seed).fit(realXStd_train,realY_train)

# Get coefficients
def get_coefficients():
    col_names = idealX.columns.values
    print("Model (a):")
    model_coefs = zip(col_names,log_reg_ideal.coef_[0])
    for name,coef in model_coefs:
        print("  "+name+": "+str(coef))
    print("Model (b):")
    model_coefs = zip(col_names,log_reg_real.coef_[0])
    for name,coef in model_coefs:
        print("  "+name+": "+str(coef))

# Get logistic regression prediction probabilities
ideal_prediction_probs = pd.DataFrame(log_reg_ideal.predict_proba(idealXStd_valid))
real_prediction_probs = pd.DataFrame(log_reg_real.predict_proba(realXStd_valid))

# Get predictions based on threshold
threshold_list = [0.5,0.7]
ideal_predictions_list = []
real_predictions_list = []
for threshold in threshold_list:
    ideal_predictions_list += [(ideal_prediction_probs.applymap(lambda x: 1 if x>threshold else 0)).iloc[:,1]]
    real_predictions_list += [(real_prediction_probs.applymap(lambda x: 1 if x>threshold else 0)).iloc[:,1]]

ideal_predictions = np.array(ideal_predictions_list[0])
real_predictions = np.array(real_predictions_list[0])

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
        subideal = idealY_valid[train_data[col] == 1]
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
        subideal = idealY_valid[train_data[col] == 1]
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
    # Metrics for Income
    income_categories = [0, 25000, 50000, 75000, 100000, 150000, 200000, 300000]
    for idx in range(len(income_categories)-1):
        low_inclusive = income_categories[idx]
        high_exclusive = income_categories[idx+1]
        print("Income between "+str(low_inclusive)+" and "+str(high_exclusive)+":")
        subideal = idealY_valid[(train_data["Income"] >= low_inclusive) & (train_data["Income"] < high_exclusive)]
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
    # Metric for specific groups
    print("White, Male, Income 200000 to 300000:")
    filter = (train_data["Income"] >= 200000) & (train_data["Income"] < 300000) & (train_data["White_Race"] == 1) & (train_data["Male_Sex"] == 1)
    subideal = idealY_valid[filter]
    sub_predictions = predicted_labels[filter]
    if len(sub_predictions) != 0:
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
    else:
        print("Warning: Specific group size is 0!")

    return metrics

# Get disparity for different groups above
def get_disparity(experiment,comparison):
    # compute disparities
    disparities = []
    for i in range(int(len(experiment)/len(comparison))):
        subExperiment = experiment[i*6:i*6+6]
        disparities += list(np.round(np.array(subExperiment)/np.array(comparison),4))

    # print out disparities for Race
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
    # print our disparities for Sex
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
    # print out disparities for specific groups
    print("White, Male, Income 200000 to 300000:")
    print("  Precision Disparity=" + str(disparities[96]))
    print("  Recall Disparity=" + str(disparities[97]))
    print("  AUC Disparity=" + str(disparities[98]))
    print("  FPR Disparity=" + str(disparities[99]))
    print("  FNR Disparity=" + str(disparities[100]))
    print("  Selection Rate Disparity=" + str(disparities[101]))
        
get_coefficients()
print("------------------------------")
#experiment_a = metric_report(idealX_valid,ideal_predictions)
print("------------------------------")
#experiment_b = metric_report(realX_valid,real_predictions)
print("------------------------------")
comparisonGroup = [0.6721,0.8542,0.5634,0.7273,0.1458,0.8079]
#get_disparity(experiment_b,comparisonGroup)