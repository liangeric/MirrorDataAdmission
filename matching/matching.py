# Load necessary packages
import numpy as np
import pandas as pd

def calculate_score(dataSource, noncategories):
    # given mean, normalize all rows of non-categorical data
    for (col,mu,sigma) in noncategories:
        dataSource[col] = (dataSource[col] - mu)/sigma
    return dataSource

# Finds matches of the first dataset in the second dataset (according to order in argument)
def find_match(scores1,scores2, noncategories, categories):
    matches = []
    for i in range(scores1.shape[0]):
        smallestDist = np.inf
        indexMatch = 0
        scores1nonCat = scores1[noncategories]
        scores2nonCat = scores2[noncategories]
        scores1Cat = scores1[categories]
        scores2Cat = scores2[categories]
        for j in range(scores2.shape[0]):
            # Find Euclidean distance between rows for non-categorical (L2 norm on difference)
            currentDistance = np.linalg.norm(scores1nonCat.iloc[i] - scores2nonCat.iloc[j])
            # Indicator for each categorical to check if it's the same
            currentDistance += len(categories) - np.sum(scores1Cat.iloc[i] == scores2Cat.iloc[j])
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
    noncategories = []
    for (col,mu,sigma) in noncategories:
        noncategories += [col]

    # Get scores for datasets
    scores1 = calculate_score(data1, noncategoriesOne)
    scores2 = calculate_score(data2, noncategoriesTwo)
    
    # Match the datasets
    matches = find_match(scores1,scores2, noncategories, categories)

    # Generate matched dataset
    matchedData = data2.iloc[matches]
    matchedData.to_csv("matching/matched.csv")