from mirror.nodes import *
from mirror.edges import *
from mirror.generator import Mirror
import pandas as pd

# size of the data
total_n = 300000

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

edge_race_income = CtoN("Race","Income",{"White":["OrdLocal",[0, 25000, 50000, 75000, 100000, 150000, 200000, 300000],[0.18, 0.1, 0.165,0.225,0.15,0.08,0.1],None,None,None],
                                         "Black":["OrdLocal",[0, 25000, 50000, 75000, 100000, 150000, 200000, 300000],[0.18, 0.2, 0.165,0.125,0.15,0.08,0.1],None,None,None],
                                         "American Indian":["OrdLocal",[0, 25000, 50000, 75000, 100000, 150000, 200000, 300000],[0.18, 0.2, 0.165,0.125,0.15,0.08,0.1],None,None,None],
                                         "Asian":["OrdLocal",[0, 25000, 50000, 75000, 100000, 150000, 200000, 300000],[0.18, 0.2, 0.165,0.125,0.15,0.08,0.1],None,None,None],
                                         "Pacific Islander":["OrdLocal",[0, 25000, 50000, 75000, 100000, 150000, 200000, 300000],[0.18, 0.2, 0.165,0.125,0.15,0.08,0.1],None,None,None],
                                         "Other":["OrdLocal",[0, 25000, 50000, 75000, 100000, 150000, 200000, 300000],[0.18, 0.2, 0.165,0.125,0.15,0.08,0.1],None,None,None]})

nodes = [node_race,node_income]
edge_relation = {"Income": edge_race_income}
mirror = Mirror(seed=0)
mirror.generate_csv(nodes, edge_relation)
mirror.save_to_disc("tester.csv")   

data = pd.read_csv("tester.csv")
print(data.groupby("Race").describe())