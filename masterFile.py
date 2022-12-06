import numpy as np
import pandas as pd
from tqdm import tqdm
import idealGeneration
import newGeneration
import matching.matching_generator as matcher
import modeling.logisticRegression as modeler
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

# Column names
column_names = ["American Indian Precision",
                    "American Indian Recall",
                    "American Indian AUC",
                    "American Indian FPR",
                    "American Indian FNR",
                    "American Indian Selection Rate",
                    "Asian Precision",
                    "Asian Recall",
                    "Asian AUC",
                    "Asian FPR",
                    "Asian FNR",
                    "Asian Selection Rate",
                    "Black Precision",
                    "Black Recall",
                    "Black AUC",
                    "Black FPR",
                    "Black FNR",
                    "Black Selection Rate",
                    "Other Race Precision",
                    "Other Race Recall",
                    "Other Race AUC",
                    "Other Race FPR",
                    "Other Race FNR",
                    "Other Race Selection Rate",
                    "Pacific Islander Precision",
                    "Pacific Islander Recall",
                    "Pacific Islander AUC",
                    "Pacific Islander FPR",
                    "Pacific Islander FNR",
                    "Pacific Islander Selection Rate",
                    "White Precision",
                    "White Recall",
                    "White AUC",
                    "White FPR",
                    "White FNR",
                    "White Selection Rate",
                    "Female Precision",
                    "Female Recall",
                    "Female AUC",
                    "Female FPR",
                    "Female FNR",
                    "Female Selection Rate",
                    "Male Precision",
                    "Male Recall",
                    "Male AUC",
                    "Male FPR",
                    "Male FNR",
                    "Male Selection Rate",
                    "Other Sex Precision",
                    "Other Sex Recall",
                    "Other Sex AUC",
                    "Other Sex FPR",
                    "Other Sex FNR",
                    "Other Sex Selection Rate",
                    "Income 0 Precision",
                    "Income 0 Recall",
                    "Income 0 AUC",
                    "Income 0 FPR",
                    "Income 0 FNR",
                    "Income 0 Selection Rate",
                    "Income 25000 Precision",
                    "Income 25000 Recall",
                    "Income 25000 AUC",
                    "Income 25000 FPR",
                    "Income 25000 FNR",
                    "Income 25000 Selection Rate",
                    "Income 50000 Precision",
                    "Income 50000 Recall",
                    "Income 50000 AUC",
                    "Income 50000 FPR",
                    "Income 50000 FNR",
                    "Income 50000 Selection Rate",
                    "Income 75000 Precision",
                    "Income 75000 Recall",
                    "Income 75000 AUC",
                    "Income 75000 FPR",
                    "Income 75000 FNR",
                    "Income 75000 Selection Rate",
                    "Income 100000 Precision",
                    "Income 100000 Recall",
                    "Income 100000 AUC",
                    "Income 100000 FPR",
                    "Income 100000 FNR",
                    "Income 100000 Selection Rate",
                    "Income 150000 Precision",
                    "Income 150000 Recall",
                    "Income 150000 AUC",
                    "Income 150000 FPR",
                    "Income 150000 FNR",
                    "Income 150000 Selection Rate",
                    "Income 200000 Precision",
                    "Income 200000 Recall",
                    "Income 200000 AUC",
                    "Income 200000 FPR",
                    "Income 200000 FNR",
                    "Income 200000 Selection Rate",]

# Generate the experiment results for multiple iterations
def get_data(iters):
    idealTable = np.zeros((iters,96))
    realTable = np.zeros((iters,96))

    for i in tqdm(range(iters), leave=False):
        # Run data generation
        total_n = 10000
        seed = i
        idealGeneration.generate_data(total_n,seed, verbose = False)
        newGeneration.generate_data(total_n,seed, verbose = False)
        #print("Generation Done!")

        # Run matching process
        data1path = "admissionNew.csv"
        data2path = "idealAdmission.csv"
        categories = ["Sex", "Race"]
        noncategories= [("Intrinsic Abilities",None,[0.2,0.35,0.5,0.65,0.8]),
                        ("Income",None,[25000,50000,75000,100000,150000,200000])]
        matcher.run_match(categories,noncategories,data1path,data2path, verbose = False)
        #print("Matching Done!")

        # Run modeling process
        data2MatchedPath = "matching/matched.csv"
        seed = i
        idealResults, realResults = modeler.runModel(data1path, data2MatchedPath, seed, verbose = False)
        idealTable[i] = idealResults
        realTable[i] = realResults
        #print("Modeling Done!")

    # Label column names
    idealTable = pd.DataFrame(idealTable,columns=column_names)
    realTable = pd.DataFrame(realTable,columns=column_names)
    realTable.to_csv("results/realTable.csv", index = False)
    idealTable.to_csv("results/idealTable.csv", index = False)

# Generate appropriate figure
def generateFigure(targetCategory, save = False):
    realTable = pd.read_csv("results/realTable.csv")
    idealTable = pd.read_csv("results/idealTable.csv")
    if targetCategory not in column_names:
        print("Not a valid category to generate a figure!")
    else:
        idealResult = idealTable[targetCategory]
        realResult = realTable[targetCategory]
        idealMean = np.mean(idealResult)
        realMean = np.mean(realResult)
        idealStd = np.std(idealResult)
        realStd = np.std(realResult)
        plt.figure()
        plt.errorbar(["Ideal","Real"],[idealMean,realMean],[idealStd*2,realStd*2],
                     capsize = 3, marker = '.', linestyle='None')
        plt.title(targetCategory + "(95% C.I.)")
        plt.savefig("figures/"+targetCategory)

# Comment out if regenerating data is not necessary
get_data(50)
# Get figures based on experiments
for columnName in tqdm(column_names, leave = False):
    generateFigure(columnName)