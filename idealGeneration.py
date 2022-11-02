from mirror.nodes import *
from mirror.edges import *
from mirror.generator import Mirror
import pandas as pd

# size of the data
total_n = 5000
seed = 12

# initialize nodes
node_in_abil = GaussianNode("Intrinsic Abilities",total_n,0.5,0.3**2,0,1)
node_sex = CategoricalNode("Sex", {"Male": 0.335, "Female": 0.335, "Other":0.33}, total_n)
node_race = CategoricalNode("Race", 
                            {"White": 0.17,
                             "Black": 0.17, 
                             "American Indian": 0.17,
                             "Asian":0.17, 
                             "Pacific Islander": 0.16,
                             "Other": 0.16}, 
                            total_n)
node_income = OrdinalLocalNode("Income", 
                               {"bound": [0, 25000, 50000, 75000, 100000, 150000, 200000, 300000], 
                                "probability": [0.18, 0.2, 0.165,0.125,0.15,0.08,0.1]},
                               total_n)

node_opportunities = GaussianNode("Opportunities",total_n,0.5,0.05**2,0,1)
node_AQ = GaussianNode("Academic Qualification",total_n,0.5,0.05**2,0,1)
node_NAQ = GaussianNode("Non-Academic Qualification",total_n,0.5,0.05**2,0,1)
node_diversity = GaussianNode("Diversity",total_n,0.5,0.15**2,0,1)

# Note SAT scaled down by 10, so scores range from 40 to 160 by 1, which is equivalent to 400 to 1600 by 10
node_SAT = GaussianRoundNode("SAT",total_n,105,0**2,4,160,0)
node_GPA = GaussianRoundNode("GPA",total_n,3.0,0**2,0.0,4.0,2)
node_numAPs = GaussianRoundNode("Number of APs",total_n,4,0**2,0,8,0)
node_meanAPs = GaussianRoundNode("Mean AP Score",total_n,3,0**2,0,5,0)
node_EC = GaussianNode("Extracurriculars",total_n,0.5,0**2,0,1)
node_letters = GaussianNode("Letters of Rec",total_n,0.5,0**2,0,1)
node_essay = GaussianNode("Essay",total_n,0.5,0**2,0,1)

node_admission = CategoricalNode("Admission", {"Yes": 0.5, "No": 0.5}, sample_n=total_n)

# initialize edges
edge_ability_AQ = NtoN("Intrinsic Abilities", "Academic Qualification", [0.35,0.5,0.65],[["Gaussian",0.43,0.05**2,0,1,None],
                                                                                         ["Gaussian",0.50,0.05**2,0,1,None],
                                                                                         ["Gaussian",0.57,0.05**2,0,1,None],
                                                                                         ["Gaussian",0.64,0.05**2,0,1,None]])
edge_ability_NAQ = NtoN("Intrinsic Abilities", "Non-Academic Qualification", [0.35,0.65,0.8],[["Gaussian",0.43,0.05**2,0,1,None],
                                                                                              ["Gaussian",0.50,0.05**2,0,1,None],
                                                                                              ["Gaussian",0.57,0.05**2,0,1,None],
                                                                                              ["Gaussian",0.64,0.05**2,0,1,None]])
edge_op_AQ = NtoN("Opportunities", "Academic Qualification", [0.4,0.5,0.6],[["Gaussian",0.43,0.05**2,0,1,None],
                                                                              ["Gaussian",0.50,0.05**2,0,1,None],
                                                                              ["Gaussian",0.57,0.05**2,0,1,None],
                                                                              ["Gaussian",0.64,0.05**2,0,1,None]])
edge_op_NAQ = NtoN("Opportunities", "Non-Academic Qualification", [0.4,0.5,0.6],[["Gaussian",0.43,0.05**2,0,1,None],
                                                                                   ["Gaussian",0.50,0.05**2,0,1,None],
                                                                                   ["Gaussian",0.57,0.05**2,0,1,None],
                                                                                   ["Gaussian",0.64,0.05**2,0,1,None]])
edge_sex_con = CtoN("Sex","Diversity",{"Male":["Gaussian",0.35,0.15**2,0,1,None],
                                       "Female":["Gaussian",0.5,0.15**2,0,1,None],
                                       "Other":["Gaussian",0.7,0.15**2,0,1,None]})
edge_race_con = CtoN("Race","Diversity",{"White":["Gaussian",0.25,0.15**2,0,1,None],
                                         "Black":["Gaussian",0.75,0.15**2,0,1,None],
                                         "American Indian":["Gaussian",0.75,0.15**2,0,1,None],
                                         "Asian":["Gaussian",0.5,0.15**2,0,1,None],
                                         "Pacific Islander":["Gaussian",0.5,0.15**2,0,1,None],
                                         "Other":["Gaussian",0.75,0.15**2,0,1,None]})
edge_income_con = NtoN("Income", "Diversity", [25000,50000,75000,100000,150000,200000],[["Gaussian",0.40,0.01**2,0,1,None],
                                                                                        ["Gaussian",0.45,0.01**2,0,1,None],
                                                                                        ["Gaussian",0.50,0.01**2,0,1,None],
                                                                                        ["Gaussian",0.55,0.01**2,0,1,None],
                                                                                        ["Gaussian",0.60,0.01**2,0,1,None],
                                                                                        ["Gaussian",0.65,0.01**2,0,1,None],
                                                                                        ["Gaussian",0.70,0.01**2,0,1,None]])


edge_AQ_SAT = NtoN("Academic Qualification", "SAT", [0.3,0.5,0.7],[["Gaussian",925,100**2,400,1600,0],
                                                                     ["Gaussian",1050,100**2,400,1600,0],
                                                                     ["Gaussian",1175,100**2,400,1600,0],
                                                                     ["Gaussian",1300,100**2,400,1600,0]])
edge_AQ_GPA = NtoN("Academic Qualification", "GPA", [0.3,0.5,0.7],[["Gaussian",2.75,0.15**2,0,4,2],
                                                                     ["Gaussian",3.00,0.15**2,0,4,2],
                                                                     ["Gaussian",3.25,0.15**2,0,4,2],
                                                                     ["Gaussian",3.50,0.15**2,0,4,2]])  
edge_AQ_numAPs = NtoN("Academic Qualification", "Number of APs", [0.3,0.5,0.7],[["Gaussian",3,0.5**2,0,8,0],
                                                                                  ["Gaussian",4,0.5**2,0,8,0],
                                                                                  ["Gaussian",5,0.5**2,0,8,0],
                                                                                  ["Gaussian",6,0.5**2,0,8,0]]) 
edge_AQ_meanAPs = NtoN("Academic Qualification", "Mean AP Score", [0.3,0.5,0.7],[["Gaussian",2.75,01.5**2,0,5,1],
                                                                                   ["Gaussian",3.00,0.15**2,0,5,1],
                                                                                   ["Gaussian",3.25,0.15**2,0,5,1],
                                                                                   ["Gaussian",3.50,0.15**2,0,5,1]]) 


edge_AQ_letters = NtoN("Academic Qualification", "Letters of Rec", [0.3,0.5,0.7],[["Gaussian",0.2,0.05**2,0,1,None],
                                                                                    ["Gaussian",0.4,0.05**2,0,1,None],
                                                                                    ["Gaussian",0.6,0.05**2,0,1,None],
                                                                                    ["Gaussian",0.8,0.05**2,0,1,None]])
edge_AQ_essay = NtoN("Academic Qualification", "Essay", [0.3,0.5,0.7],[["Gaussian",0.2,0.05**2,0,1,None],
                                                                         ["Gaussian",0.4,0.05**2,0,1,None],
                                                                         ["Gaussian",0.6,0.05**2,0,1,None],
                                                                         ["Gaussian",0.8,0.05**2,0,1,None]])
edge_AQ_EC = NtoN("Academic Qualification", "Extracurriculars", [0.3,0.5,0.7],[["Gaussian",0.2,0.05**2,0,1,None],
                                                                                  ["Gaussian",0.4,0.05**2,0,1,None],
                                                                                  ["Gaussian",0.6,0.05**2,0,1,None],
                                                                                  ["Gaussian",0.8,0.05**2,0,1,None]])       
edge_NAQ_EC = NtoN("Non-Academic Qualification", "Extracurriculars", [0.3,0.5,0.7],[["Gaussian",0.2,0.05**2,0,1,None],
                                                                                      ["Gaussian",0.4,0.05**2,0,1,None],
                                                                                      ["Gaussian",0.6,0.05**2,0,1,None],
                                                                                      ["Gaussian",0.8,0.05**2,0,1,None]])                                                                                                                                                                                                                                                                                                                
edge_NAQ_letters = NtoN("Non-Academic Qualification", "Letters of Rec", [0.3,0.5,0.7],[["Gaussian",0.2,0.05**2,0,1,None],
                                                                                         ["Gaussian",0.4,0.05**2,0,1,None],
                                                                                         ["Gaussian",0.6,0.05**2,0,1,None],
                                                                                         ["Gaussian",0.8,0.05**2,0,1,None]])
edge_NAQ_essay = NtoN("Non-Academic Qualification", "Essay", [0.3,0.5,0.7],[["Gaussian",0.2,0.05**2,0,1,None],
                                                                             ["Gaussian",0.4,0.05**2,0,1,None],
                                                                             ["Gaussian",0.6,0.05**2,0,1,None],
                                                                             ["Gaussian",0.8,0.05**2,0,1,None]])   
edge_con_essay = NtoN("Diversity", "Essay", [0.4,0.5,0.6],[["Gaussian",0.2,0.05**2,0,1,None],
                                                             ["Gaussian",0.4,0.05**2,0,1,None],
                                                             ["Gaussian",0.6,0.05**2,0,1,None],
                                                             ["Gaussian",0.8,0.05**2,0,1,None]])   


edge_SAT_admission = NtoC("SAT","Admission",[1000,1100,1200],[{"Yes": 0.1, "No": 0.9}, 
                                                             {"Yes": 0.2, "No": 0.8},
                                                             {"Yes": 0.8, "No": 0.2},
                                                             {"Yes": 0.9, "No": 0.1}])
edge_GPA_admission = NtoC("GPA","Admission",[3.0,3.2,3.4],[{"Yes": 0.1, "No": 0.9}, 
                                                             {"Yes": 0.2, "No": 0.8},
                                                             {"Yes": 0.8, "No": 0.2},
                                                             {"Yes": 0.9, "No": 0.1}])
edge_numAPs_admission = NtoC("Number of APs","Admission",[4,5,6],[{"Yes": 0.1, "No": 0.9}, 
                                                             {"Yes": 0.2, "No": 0.8},
                                                             {"Yes": 0.8, "No": 0.2},
                                                             {"Yes": 0.9, "No": 0.1}])
edge_meanAPs_admission = NtoC("Mean AP Score","Admission",[3.0,3.2,3.4],[{"Yes": 0.1, "No": 0.9}, 
                                                             {"Yes": 0.2, "No": 0.8},
                                                             {"Yes": 0.8, "No": 0.2},
                                                             {"Yes": 0.9, "No": 0.1}])
edge_EC_admission = NtoC("Extracurriculars","Admission",[0.40,0.50,0.60],[{"Yes": 0.1, "No": 0.9}, 
                                                             {"Yes": 0.2, "No": 0.8},
                                                             {"Yes": 0.8, "No": 0.2},
                                                             {"Yes": 0.9, "No": 0.1}])
edge_letters_admission = NtoC("Letters of Rec","Admission",[0.40,0.50,0.60],[{"Yes": 0.1, "No": 0.9}, 
                                                             {"Yes": 0.2, "No": 0.8},
                                                             {"Yes": 0.8, "No": 0.2},
                                                             {"Yes": 0.9, "No": 0.1}])
edge_essay_admission = NtoC("Essay","Admission",[0.40,0.50,0.60],[{"Yes": 0.1, "No": 0.9}, 
                                                             {"Yes": 0.2, "No": 0.8},
                                                             {"Yes": 0.8, "No": 0.2},
                                                             {"Yes": 0.9, "No": 0.1}])
edge_con_admission = NtoC("Diversity","Admission",[0.50,0.55,0.60],[{"Yes": 0.1, "No": 0.9}, 
                                                             {"Yes": 0.2, "No": 0.8},
                                                             {"Yes": 0.8, "No": 0.2},
                                                             {"Yes": 0.9, "No": 0.1}])



# Create DAG
nodes = [node_in_abil, node_sex, node_race, node_income, node_opportunities, node_AQ, node_NAQ, 
         node_diversity, node_SAT, node_GPA, node_numAPs, node_meanAPs, node_EC, node_letters, 
         node_essay, node_admission]
edge_relation = {"Academic Qualification": ([edge_ability_AQ,edge_op_AQ],
                                            [0.5,0.5]),
                 "Non-Academic Qualification": ([edge_ability_NAQ,edge_op_NAQ],
                                                [0.5,0.5]),
                 "Diversity": ([edge_sex_con,edge_race_con,edge_income_con],
                               [0.33,0.34,0.33]),
                 "SAT": edge_AQ_SAT,
                 "GPA": edge_AQ_GPA,
                 "Number of APs": edge_AQ_numAPs,
                 "Mean AP Score": edge_AQ_meanAPs,
                 "Extracurriculars": ([edge_AQ_EC,edge_NAQ_EC],
                                      [0.5,0.5]),
                 "Letters of Rec": ([edge_AQ_letters,edge_NAQ_letters],
                                    [0.5,0.5]),
                 "Essay": ([edge_AQ_essay,edge_NAQ_essay,edge_con_essay],
                           [0.335,0.335,0.33]),
                 "Admission": ([edge_SAT_admission,edge_GPA_admission,edge_numAPs_admission,edge_meanAPs_admission,
                                edge_EC_admission,edge_letters_admission,edge_essay_admission,edge_con_admission],
                               [0.1,0.1,0.1,0.1,0.15,0.15,0.15,0.15])}



# generate data
mirror = Mirror(seed=seed)
mirror.generate_csv(nodes, edge_relation)
mirror.save_to_disc("idealAdmission.csv", excluded_cols=["group","C_SAT","C_GPA","C_Number of APs",
                                                       "C_Mean AP Score", "C_Extracurriculars",
                                                       "C_Letters of Rec","C_Essay","C_Diversity"])                               
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  