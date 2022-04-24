# Load necessary packages
import numpy as np
import pandas as pd
import random
from bisect import bisect

def roundNonCat(dataSource, noncategories):
    newData = dataSource.copy()
    for (col,decimalsRound,buckets) in noncategories:
        # given column name and number of decimal places to round to
        if decimalsRound != None:
            newData[col] = np.round(newData[col], decimalsRound)
        # given column name and buckets to match to
        else:
            newData[col] = newData[col].apply(lambda x: bisect(buckets,x))
    return newData

# Finds matches of the first dataset in the second dataset (according to order in argument)
def find_match(dataOne,dataTwo):
    matches = []
    for i in range(dataOne.shape[0]):
        # Shuffle second dataset search order so that if there are ties, ties are broken randomly
        searchOrder = list(range(dataTwo.shape[0]))
        random.shuffle(searchOrder)
        # Look for an exact match in second Dataset
        for j in searchOrder:
            if (np.sum(dataOne.iloc[i] == dataTwo.iloc[j]) == dataOne.shape[1]):
                matches +=[j]
                break
            if (j == (dataTwo.shape[0]-1)):
                matches +=[dataTwo.shape[0]]
    return matches
    

if __name__ == '__main__':
    # Load in two data files
    data1 = pd.read_csv("../MirrorDataAdmission/matching/sample1.csv")
    data2 = pd.read_csv("../MirrorDataAdmission/matching/sample2.csv")

    # List of row names that are categorical
    categories = ["diversity", "admission"]
    # List of row names that are non-categorical in the format of (name,decimalsRound,buckets)
    # name: name of column
    # decimalsRound: how many decimal places to keep or None if using buckets
    # buckets: Buckets to do rounding, make sure that decimalsRound is None, format of buckets example below:
    #          E.g. [20, 45, 65] defines 4 buckets: [-, 20), [20, 45), [45, 65), [65, -)
    noncategories= [("TOEFL",None,[70,80,90,100])]

    # round the noncategorical columns
    data1Round = roundNonCat(data1,noncategories)
    data2Round = roundNonCat(data2,noncategories)
    
    # Match the datasets
    matches = find_match(data1Round,data2Round)

    # add a row of NA's for matches not found
    extraRow = data2.iloc[-1]
    data2 = data2.append(extraRow)
    data2.iloc[-1] = np.nan

    # Generate matched dataset
    matchedData = data2.iloc[matches]
    matchedData.to_csv("matching/matched.csv")