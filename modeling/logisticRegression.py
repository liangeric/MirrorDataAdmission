# Load necessary packages
import numpy as np
import pandas as pd

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
realY = realY.astype("category")
realY = realY.cat.codes
idealY = idealY.astype("category")
idealY = idealY.cat.codes

# Train logistic regression

# Get logistic regression predictions