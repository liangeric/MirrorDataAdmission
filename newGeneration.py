from mirror.nodes import *
from mirror.edges import *
from mirror.generator import Mirror
import pandas as pd

# size of the data
total_n = 30000

# initialize nodes
node_in_abil = GaussianNode("Intrinsic Abilities",total_n,0.5,0.15**2,0,1)
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

node_opportunities = GaussianNode("Opportunities",total_n,0.5,0.15**2,0,1)
node_AQ = GaussianNode("Academic Qualification",total_n,0.5,0.15**2,0,1)
node_NAQ = GaussianNode("Non-Academic Qualification",total_n,0.5,0.15**2,0,1)
node_diversity = GaussianNode("Diversity",total_n,0.5,0.15**2,0,1)

# Note SAT scaled down by 10, so scores range from 40 to 160 by 1, which is equivalent to 400 to 1600 by 10
node_SAT = GaussianRoundNode("SAT",total_n,105,25**2,4,160,0)
node_GPA = GaussianRoundNode("GPA",total_n,3.0,0.5**2,0.0,4.0,2)
node_numAPs = GaussianRoundNode("Number of APs",total_n,4,1**2,0,8,0)
node_meanAPs = GaussianRoundNode("Mean AP Score",total_n,3,0.5**2,0,5,0)
node_EC = GaussianNode("Extracurriculars",total_n,0.5,0.15**2,0,1)
node_letters = GaussianNode("Letters of Rec",total_n,0.5,0.15**2,0,1)
node_essay = GaussianNode("Essay",total_n,0.5,0.15**2,0,1)

node_admission = CategoricalNode("Admission", {"Yes": 0.5, "No": 0.5}, sample_n=total_n)

# initialize edges
edge_race_income = CtoN("Race","Income",{"White":["OrdLocal",[0, 25000, 50000, 75000, 100000, 150000, 200000, 300000],[0.18, 0.1, 0.165,0.225,0.15,0.08,0.1],None,None,None],
                                         "Black":["OrdLocal",[0, 25000, 50000, 75000, 100000, 150000, 200000, 300000],[0.18, 0.2, 0.165,0.125,0.15,0.08,0.1],None,None,None],
                                         "American Indian":["OrdLocal",[0, 25000, 50000, 75000, 100000, 150000, 200000, 300000],[0.18, 0.2, 0.165,0.125,0.15,0.08,0.1],None,None,None],
                                         "Asian":["OrdLocal",[0, 25000, 50000, 75000, 100000, 150000, 200000, 300000],[0.18, 0.2, 0.165,0.125,0.15,0.08,0.1],None,None,None],
                                         "Pacific Islander":["OrdLocal",[0, 25000, 50000, 75000, 100000, 150000, 200000, 300000],[0.18, 0.2, 0.165,0.125,0.15,0.08,0.1],None,None,None],
                                         "Other":["OrdLocal",[0, 25000, 50000, 75000, 100000, 150000, 200000, 300000],[0.18, 0.2, 0.165,0.125,0.15,0.08,0.1],None,None,None]})
edge_ability_op = NtoN("Intrinsic Abilities", "Opportunities", [0.35,0.65,0.8],[["Gaussian",0.43,0.15**2,0,1,None],
                                                                                ["Gaussian",0.50,0.15**2,0,1,None],
                                                                                ["Gaussian",0.57,0.15**2,0,1,None],
                                                                                ["Gaussian",0.64,0.15**2,0,1,None]])
edge_sex_op = CtoN("Sex","Opportunities",{"Male":["Gaussian",0.6,0.15**2,0,1,None],
                                          "Female":["Gaussian",0.5,0.15**2,0,1,None]})
edge_race_op = CtoN("Race","Opportunities",{"White":["Gaussian",0.5,0.15**2,0,1,None],
                                            "Black":["Gaussian",0.35,0.15**2,0,1,None],
                                            "American Indian":["Gaussian",0.35,0.15**2,0,1,None],
                                            "Asian":["Gaussian",0.35,0.15**2,0,1,None],
                                            "Pacific Islander":["Gaussian",0.35,0.15**2,0,1,None],
                                            "Other":["Gaussian",0.35,0.15**2,0,1,None]})
edge_income_op = NtoN("Income", "Opportunities", [25000,50000,75000,100000,150000,200000],[["Gaussian",0.40,0.01**2,0,1,None],
                                                                                           ["Gaussian",0.45,0.01**2,0,1,None],
                                                                                           ["Gaussian",0.50,0.01**2,0,1,None],
                                                                                           ["Gaussian",0.54,0.01**2,0,1,None],
                                                                                           ["Gaussian",0.60,0.01**2,0,1,None],
                                                                                           ["Gaussian",0.65,0.01**2,0,1,None],
                                                                                           ["Gaussian",0.70,0.01**2,0,1,None]])


edge_ability_AQ = NtoN("Intrinsic Abilities", "Academic Qualification", [0.35,0.65,0.8],[["Gaussian",0.43,0.15**2,0,1,None],
                                                                                         ["Gaussian",0.50,0.15**2,0,1,None],
                                                                                         ["Gaussian",0.57,0.15**2,0,1,None],
                                                                                         ["Gaussian",0.64,0.15**2,0,1,None]])
edge_ability_NAQ = NtoN("Intrinsic Abilities", "Non-Academic Qualification", [0.35,0.65,0.8],[["Gaussian",0.43,0.15**2,0,1,None],
                                                                                              ["Gaussian",0.50,0.15**2,0,1,None],
                                                                                              ["Gaussian",0.57,0.15**2,0,1,None],
                                                                                              ["Gaussian",0.64,0.15**2,0,1,None]])
edge_op_AQ = NtoN("Opportunities", "Academic Qualification", [0.35,0.65,0.8],[["Gaussian",0.43,0.15**2,0,1,None],
                                                                              ["Gaussian",0.50,0.15**2,0,1,None],
                                                                              ["Gaussian",0.57,0.15**2,0,1,None],
                                                                              ["Gaussian",0.64,0.15**2,0,1,None]])
edge_op_NAQ = NtoN("Opportunities", "Non-Academic Qualification", [0.35,0.65,0.8],[["Gaussian",0.43,0.15**2,0,1,None],
                                                                                   ["Gaussian",0.50,0.15**2,0,1,None],
                                                                                   ["Gaussian",0.57,0.15**2,0,1,None],
                                                                                   ["Gaussian",0.64,0.15**2,0,1,None]])
edge_sex_con = CtoN("Sex","Diversity",{"Male":["Gaussian",0.5,0.15**2,0,1,None],
                                       "Female":["Gaussian",0.35,0.15**2,0,1,None]})
edge_race_con = CtoN("Race","Diversity",{"White":["Gaussian",0.35,0.15**2,0,1,None],
                                         "Black":["Gaussian",0.5,0.15**2,0,1,None],
                                         "American Indian":["Gaussian",0.5,0.15**2,0,1,None],
                                         "Asian":["Gaussian",0.5,0.15**2,0,1,None],
                                         "Pacific Islander":["Gaussian",0.5,0.15**2,0,1,None],
                                         "Other":["Gaussian",0.5,0.15**2,0,1,None]})
edge_income_con = NtoN("Income", "Diversity", [25000,50000,75000,100000,150000,200000],[["Gaussian",0.40,0.01**2,0,1,None],
                                                                                        ["Gaussian",0.45,0.01**2,0,1,None],
                                                                                        ["Gaussian",0.50,0.01**2,0,1,None],
                                                                                        ["Gaussian",0.54,0.01**2,0,1,None],
                                                                                        ["Gaussian",0.60,0.01**2,0,1,None],
                                                                                        ["Gaussian",0.65,0.01**2,0,1,None],
                                                                                        ["Gaussian",0.70,0.01**2,0,1,None]])


edge_AQ_SAT = NtoN("Academic Qualification", "SAT", [0.35,0.65,0.8],[["Gaussian",925,250**2,0,1,None],
                                                                     ["Gaussian",1050,250**2,0,1,None],
                                                                     ["Gaussian",1175,250**2,0,1,None],
                                                                     ["Gaussian",1300,250**2,0,1,None]])
edge_AQ_GPA = NtoN("Academic Qualification", "GPA", [0.35,0.65,0.8],[["Gaussian",2.75,0.5**2,0,1,None],
                                                                     ["Gaussian",3.00,0.5**2,0,1,None],
                                                                     ["Gaussian",3.25,0.5**2,0,1,None],
                                                                     ["Gaussian",3.50,0.5**2,0,1,None]])  
edge_AQ_numAPs = NtoN("Academic Qualification", "Number of APs", [0.35,0.65,0.8],[["Gaussian",3,1**2,0,1,None],
                                                                               ["Gaussian",4,1**2,0,1,None],
                                                                               ["Gaussian",5,1**2,0,1,None],
                                                                               ["Gaussian",6,1**2,0,1,None]]) 
edge_AQ_meanAPs = NtoN("Academic Qualification", "Mean AP Score", [0.35,0.65,0.8],[["Gaussian",2.75,0.5**2,0,1,None],
                                                                                   ["Gaussian",3.00,0.5**2,0,1,None],
                                                                                   ["Gaussian",3.25,0.5**2,0,1,None],
                                                                                   ["Gaussian",3.50,0.5**2,0,1,None]]) 


edge_AQ_letters = NtoN("Academic Qualification", "Letters of Rec", [0.35,0.65,0.8],[["Gaussian",0.43,0.15**2,0,1,None],
                                                                                    ["Gaussian",0.50,0.15**2,0,1,None],
                                                                                    ["Gaussian",0.57,0.15**2,0,1,None],
                                                                                    ["Gaussian",0.64,0.15**2,0,1,None]])
edge_AQ_essay = NtoN("Academic Qualification", "Essay", [0.35,0.65,0.8],[["Gaussian",0.43,0.15**2,0,1,None],
                                                                         ["Gaussian",0.50,0.15**2,0,1,None],
                                                                         ["Gaussian",0.57,0.15**2,0,1,None],
                                                                         ["Gaussian",0.64,0.15**2,0,1,None]])
edge_NAQ_EC = NtoN("Non-Academic Qualification", "Extracurriculars", [0.35,0.65,0.8],[["Gaussian",0.43,0.15**2,0,1,None],
                                                                                      ["Gaussian",0.50,0.15**2,0,1,None],
                                                                                      ["Gaussian",0.57,0.15**2,0,1,None],
                                                                                      ["Gaussian",0.64,0.15**2,0,1,None]])                                                                                                                                                                                                                                                                                                                
edge_NAQ_letters = NtoN("Non-Academic Qualification", "Letters of Rec", [0.35,0.65,0.8],[["Gaussian",0.43,0.15**2,0,1,None],
                                                                                         ["Gaussian",0.50,0.15**2,0,1,None],
                                                                                         ["Gaussian",0.57,0.15**2,0,1,None],
                                                                                         ["Gaussian",0.64,0.15**2,0,1,None]])
edge_NAQ_essay = NtoN("Non-Academic Qualification", "Essay", [0.35,0.65,0.8],[["Gaussian",0.43,0.15**2,0,1,None],
                                                                             ["Gaussian",0.50,0.15**2,0,1,None],
                                                                             ["Gaussian",0.57,0.15**2,0,1,None],
                                                                             ["Gaussian",0.64,0.15**2,0,1,None]])   
edge_con_essay = NtoN("Diversity", "Essay", [0.35,0.65,0.8],[["Gaussian",0.43,0.15**2,0,1,None],
                                                             ["Gaussian",0.50,0.15**2,0,1,None],
                                                             ["Gaussian",0.57,0.15**2,0,1,None],
                                                             ["Gaussian",0.64,0.15**2,0,1,None]])   


edge_SAT_admission = NtoC("SAT","Admission",[800,1050,1300],[{"Y": 0.4, "N": 0.6}, 
                                                             {"Y": 0.5, "N": 0.5},
                                                             {"Y": 0.6, "N": 0.4},
                                                             {"Y": 0.7, "N": 0.3}])
edge_GPA_admission = NtoC("GPA","Admission",[2.5,3.0,3.5],[{"Y": 0.4, "N": 0.6}, 
                                                           {"Y": 0.5, "N": 0.5},
                                                           {"Y": 0.6, "N": 0.4},
                                                           {"Y": 0.7, "N": 0.3}])
edge_numAPs_admission = NtoC("Number of APs","Admission",[3,4,5],[{"Y": 0.4, "N": 0.6}, 
                                                                  {"Y": 0.5, "N": 0.5},
                                                                  {"Y": 0.6, "N": 0.4},
                                                                  {"Y": 0.7, "N": 0.3}])
edge_meanAPs_admission = NtoC("Mean AP Score","Admission",[2.5,3.0,3.5],[{"Y": 0.4, "N": 0.6}, 
                                                                         {"Y": 0.5, "N": 0.5},
                                                                         {"Y": 0.6, "N": 0.4},
                                                                         {"Y": 0.7, "N": 0.3}]) 
edge_EC_admission = NtoC("Extracurriculars","Admission",[0.35,0.50,0.65],[{"Y": 0.4, "N": 0.6}, 
                                                                          {"Y": 0.5, "N": 0.5},
                                                                          {"Y": 0.6, "N": 0.4},
                                                                          {"Y": 0.7, "N": 0.3}])
edge_letters_admission = NtoC("Letters of Rec","Admission",[0.35,0.50,0.65],[{"Y": 0.4, "N": 0.6}, 
                                                                             {"Y": 0.5, "N": 0.5},
                                                                             {"Y": 0.6, "N": 0.4},
                                                                             {"Y": 0.7, "N": 0.3}])   
edge_essay_admission = NtoC("Essay","Admission",[0.35,0.50,0.65],[{"Y": 0.4, "N": 0.6}, 
                                                                  {"Y": 0.5, "N": 0.5},
                                                                  {"Y": 0.6, "N": 0.4},
                                                                  {"Y": 0.7, "N": 0.3}])
edge_con_admission = NtoC("Diversity","Admission",[0.35,0.50,0.65],[{"Y": 0.4, "N": 0.6}, 
                                                                    {"Y": 0.5, "N": 0.5},
                                                                    {"Y": 0.6, "N": 0.4},
                                                                    {"Y": 0.7, "N": 0.3}])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               