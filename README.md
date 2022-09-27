# Data Admission Generator
Using the code in this repository, users are able to specify a graphical model and generate synthetic data from it. To see an example of a graphical model and how to generate data please see the idealGeneration.py or newGeneration.py (the only difference is that idealGeneration.py generates data from an ideal world that we will specify later). Note that this repository is based off/used [this repository](https://github.com/DataResponsibly/MirrorDataGenerator) and expands on it with more features.

## Architecture of nodes/edges in graphical model
To see the code on how to create nodes/edges or how to generate data, all the code is under the mirror subdirectory. Note that the file erasers.py is not used in our current code and was kept for consistency from the original repository as mentioned above. To add new edge types of node types, users should be able to directly edit edges.py and nodes.py as specified in the comments of these files.

## Generating our synthetic data
In the main repository, as mentioned previously we generate data from two different graphical models. The code representing these graphical models is under idealGeneration.py and newGeneration.py, and the respective generated synthetic data files are called idealAdmission.csv and admissionNew.csv. To see details about how to generate or create your own graphical model, please see the details/comments in either of these files. To simpliy edit the amount of data generated, users can edit the "n" parameter in either file as sepcified in the comments. Also note that for easy visual representation of the models, users can see the models and details of the model [here](https://docs.google.com/presentation/d/104YyZtvxNQyOVcIE9kXj3wTkH71K1nqNqf6h7myk1c8/edit?usp=sharing). Slide 1 represents newGeneration.py and slide 8 represents idealGeneration.py, slides 2-7 represent details about the edges in both models.

To generate data, users need to run the command in the following format: "python newGeneration.py".

## Analysis of data
We have also added code to carry out some basic analysis and breakdown of the generated synthetic data. The jupyter notebook that outputs the analysis can be found under "dataAnalysis.ipynb".

## Matching data
We also created code in this repository under matching, where given two data files and specifications, will try to match the data in one file to the other. This does exact matching for categorical columns, and for continuous columns, users can specify buckets where the algorithm will recognize a match as long as the two values fall within the same bucket (please see comments in the file on how to specify these parameters). The outputted matching file is called matched.csv, and shows the matches it found for each row in sample1.csv using sample2.csv (it also shows the index of the match in sample2.csv and the row will be blank if no match is found).

To match data, users need to run the command in the following format: "python matching.py".

## Updated Meeting Notes
To stay up to date with planned changes and future edits please see [here](https://docs.google.com/document/d/1zaEJ0MDEkJjHLa6OlDWh4RHyN6VrYZwqWaz3UHDV9hs/edit?usp=sharing).
