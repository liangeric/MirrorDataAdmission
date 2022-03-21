# Load necessary packages
from cgitb import small
import numpy as np
import pandas as pd

def calculate_score(dataSource, categories, noncategories):
    # convert categorical to one-hot
    for col in categories:
        oneHotCols = pd.get_dummies(dataSource[col])
        dataSource = pd.concat([dataSource,oneHotCols], axis = 1)
        dataSource = dataSource.drop(col,1)

    # given mean, normalize all rows of non-categorical data
    for (col,mu,sigma) in noncategories:
        dataSource[col] = (dataSource[col] - mu)/sigma

    return dataSource

# Finds matches of the first dataset in the second dataset (according to order in argument)
def find_match(scores1,scores2):
    matches = []
    for i in range(scores1.shape[0]):
        smallestDist = np.inf
        indexMatch = 0
        for j in range(scores2.shape[0]):
            # Find Euclidean distance between rows (L2 norm on difference)
            currentDistance = np.linalg.norm(scores1.iloc[i] - scores2.iloc[j])
            if currentDistance < smallestDist:
                smallestDist = currentDistance
                indexMatch = j
        matches += [indexMatch]
    return matches
    

if __name__ == '__main__':
    # Load in two data files
    data1 = pd.read_csv("../MirrorDataAdmission/matching/sample1.csv")
    data2 = pd.read_csv("../MirrorDataAdmission/matching/sample2.csv")

    # List of row names that are categorical
    categories = ["diversity", "admission"]
    # List of row names that are non-categorical in the format of (name,mean,std) for both datasets
    noncategoriesOne = [("TOEFL",90,10)]
    noncategoriesTwo = [("TOEFL",90,10)]

    # Get scores for datasets
    scores1 = calculate_score(data1, categories, noncategoriesOne)
    scores2 = calculate_score(data2, categories, noncategoriesTwo)
    
    # Match the datasets
    matches = find_match(scores1,scores2)

    # Generate matched dataset
    matchedData = data2.iloc[matches]
    matchedData.to_csv("matching/matched.csv")