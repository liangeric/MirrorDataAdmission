# Load necessary packages
from re import search
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
        # print out for progress check
        if i%100 == 0:
            print(i)
        # Shuffle second dataset search order so that if there are ties, ties are broken randomly
        searchOrder = list(range(dataTwo.shape[0]))
        random.shuffle(searchOrder)
        # Look for an exact match in second Dataset
        for idx in range(len(searchOrder)):
            j = searchOrder[idx]
            if (np.sum(dataOne.iloc[i] == dataTwo.iloc[j]) == dataOne.shape[1]):
                matches +=[j]
                break
            if (idx == (len(searchOrder)-1)):
                matches +=[dataTwo.shape[0]]
    return matches
    

if __name__ == '__main__':
    # Load in two data files
    data1 = pd.read_csv("../admissionNew.csv")
    data2 = pd.read_csv("../idealAdmission.csv")

    # List of column names that are categorical
    categories = ["Sex", "Race"]
    # List of column names that are non-categorical in the format of (name,decimalsRound,buckets)
    # name: name of column
    # decimalsRound: how many decimal places to keep or None if using buckets
    # buckets: Buckets to do rounding, make sure that decimalsRound is None, format of buckets example below:
    #          E.g. [20, 45, 65] defines 4 buckets: [-, 20), [20, 45), [45, 65), [65, -)
    noncategories= [("Intrinsic Abilities",None,[0.2,0.35,0.5,0.65,0.8]),
                    ("Income",None,[25000,50000,75000,100000,150000,200000])]

    # get the subdatasets for matching
    allColumns = []
    allColumns += categories
    for col in noncategories:
        allColumns += [col[0]]
    subData1 = data1[allColumns]
    subData2 = data2[allColumns]

    # round the noncategorical columns
    data1Round = roundNonCat(subData1,noncategories)
    data2Round = roundNonCat(subData2,noncategories)

    # add index as a column for reference
    data2Round.reset_index(inplace=True)
    
    # Match the datasets
    data2Round = data2Round.drop_duplicates(subset = allColumns)
    matches = pd.merge(data1Round,data2Round,how = 'left', on = allColumns)
    matches = matches['index']
    matches = matches.fillna(len(matches))
    #matches = find_match(data1Round,data2Round)

    # add a row of NA's for matches not found
    extraRow = data2.iloc[-1]
    data2 = data2.append(extraRow)
    data2.iloc[-1] = np.nan

    # Generate matched dataset
    matchedData = data2.iloc[matches]
    matchedData.to_csv("./matched.csv")

    # Print out matched data shape for verification
    print(matchedData.shape)