import os, pathlib, yaml
from bisect import bisect
# comment out for local testing
from mirror.nodes import *
from mirror.edges import *
# comment out for global testing
#from nodes import *
#from edges import *

class Mirror():
    def __init__(self, seed=0):
        self.seed = seed
        np.random.seed(seed)
        self.df = None
        self.cat_cols = []
        self.num_cols = []

    def generate_csv(self, nodes, edges, config_path = None):
        """
        :param nodes: list of Node object. The order represents the order to generate the nodes.
                      E.g. [CategoricalNode("G", [], [], {"M": 0.5, "F": 0.5}, sample_n=100),
                            CategoricalNode("R", [], [], {"W": 0.5, "B": 0.5}, sample_n=100),
                            OrdinalLocalNode("X", [], [], {"bound": [1, 5, 50], "probability": [0.5, 0.5]}, sample_n=100)]
        :param edges: dict, key is the name of the Node object, value is Edge object that represents the incoming edges and its weight for this node.
                      E.g. {"X": ([CtoN("G", "X"), CtoN("R", "X")], [0.5, 0.5])} for NUM and ORD,
                           {"D": [CtoC("G", "D"), NtoC("A", "D")]} for CAT with multiple parents,
                           {"D": CtoC("G", "D")} for CAT with single parent
        :param config_path: string, represents path to config file for parameters
        :return:
        """
        df = pd.DataFrame()
        for node_i in nodes:
            if node_i.type == "NUM":
                self.num_cols.append(node_i.name)
            else:
                self.cat_cols.append(node_i.name)
            if node_i.name in edges.keys(): # have parents
                print(node_i.name, "with parents")
                # iterate the incoming edges from its parents
                if type(edges[node_i.name]) not in [tuple, list]:  # only have one parent node
                    if edges[node_i.name].parent_name not in df.columns:
                        print("The parent does not exist!")
                        raise ValueError
                    df[node_i.name] = edges[node_i.name].instantiate_values(df)
                    print("One parent", edges[node_i.name], list(df.columns))
                else:  # have more than one parent node, update the inputs probability table based on its parents
                    if type(edges[node_i.name]) == tuple:
                        parents_i = [x.parent_name for x in edges[node_i.name][0]]
                    else:
                        parents_i = [x.parent_name for x in edges[node_i.name]]
                    if len(set(parents_i).intersection(df.columns)) != len(parents_i):
                        print("Some parents do not exist!")
                        raise ValueError
                    if node_i.type == "CAT": # current node is CAT
                        df["group"] = "" # get all the possible subgroups from all the parents' categories
                        for incoming_edge_i, weight_i in zip(edges[node_i.name][0], edges[node_i.name][1]):
                            if incoming_edge_i.type[0] == "N":  # get the categories of the numerical node
                                df["C_"+incoming_edge_i.parent_name] = df[incoming_edge_i.parent_name].apply(lambda x: str(bisect(incoming_edge_i.bounds,x)))
                                df["group"] += df["C_"+incoming_edge_i.parent_name]
                                df["group"] += ","
                            else:
                                df["group"] += df[incoming_edge_i.parent_name]
                                df["group"] += ","
                        # compute the new probability table for the child node considering all possible subgroups
                        # this utilizes the definition of conditional probability to compute new table values
                        # for example for P(Y) if there are two incoming edges the below computation is true:
                        # P(Edge_1)P(Y|Edge_1) + P(Edge_2)P(Y|Edge_2) = P(Y,Edge_1) + P(Y,Edge_2) = P(Y)
                        all_cpt = {}
                        for gi in df["group"].unique():
                            gi_probability = {}
                            for node_value_i in node_i.domain:
                                prob_i = 0
                                for incoming_edge_i, weight_i in zip(edges[node_i.name][0], edges[node_i.name][1]):
                                    gi_idx = edges[node_i.name][0].index(incoming_edge_i)
                                    prob_i += weight_i * incoming_edge_i.probability_table[str.split(gi,",")[gi_idx]][node_value_i]
                                gi_probability[node_value_i] = prob_i
                            all_cpt["".join(gi)] = {x: gi_probability[x] for x in gi_probability}
                        # sample the value of the child node using above new cpt table
                        df[node_i.name] = df["group"].apply(lambda x: np.random.choice(list(all_cpt[x].keys()), p=list(all_cpt[x].values())))
                    else: # the child node is NUM or ORD
                        if config_path != None:
                            # Read config file
                            with open(config_path,'r') as file:
                                specifications = yaml.safe_load(file)

                            df[node_i.name] = 0
                            # check to see if config file has specifications for given node
                            if node_i.name in specifications.keys():
                                edge_value_generated  = dict()
                                for incoming_edge_i, weight_i in zip(edges[node_i.name][0], edges[node_i.name][1]):
                                    temp = df[node_i.name].copy()
                                    values_i = incoming_edge_i.instantiate_values(df)
                                    df[node_i.name] = temp
                                    # Get parent name and calculate new values
                                    parent_name = incoming_edge_i.parent_name
                                    edge_value_generated[parent_name] = values_i
                                    if parent_name in specifications[node_i.name].keys():
                                        parent_coef = specifications[node_i.name][parent_name]
                                        df[node_i.name] = df[node_i.name] + parent_coef * values_i
                                # add interaction term
                                interactions = specifications[node_i.name]["interaction"]
                                for interaction in interactions:
                                    firstParent = interaction[0]
                                    secondParent = interaction[1]
                                    interaction_coef = interaction[2]
                                    interaction_values = edge_value_generated[firstParent] * edge_value_generated[secondParent]
                                    df[node_i.name] = df[node_i.name] + interaction_coef * interaction_values
                            else:
                                # if there is no specification we do a weighted average with noise
                                for incoming_edge_i, weight_i in zip(edges[node_i.name][0], edges[node_i.name][1]):
                                    temp = df[node_i.name].copy()
                                    values_i = incoming_edge_i.instantiate_values(df)
                                    df[node_i.name] = temp
                                    # Take weighted average of numbers
                                    df[node_i.name] = df[node_i.name] + weight_i * values_i
                                # add noise to the weighted mean
                                mean = 0
                                sd = np.sqrt(node_i.parameters["var"])
                                noise = np.random.normal(mean,sd)
                                df[node_i.name] = df[node_i.name] + noise
                        else:
                            df[node_i.name] = 0
                            for incoming_edge_i, weight_i in zip(edges[node_i.name][0], edges[node_i.name][1]):
                                temp = df[node_i.name].copy()
                                values_i = incoming_edge_i.instantiate_values(df)
                                df[node_i.name] = temp
                                # Take weighted average of numbers
                                df[node_i.name] = df[node_i.name] + weight_i * values_i
                            # add noise to the weighted mean
                            mean = 0
                            sd = np.sqrt(node_i.parameters["var"])
                            noise = np.random.normal(mean,sd)
                            df[node_i.name] = df[node_i.name] + noise

            else: # no parents
                # instantiate using its parameters
                df[node_i.name] = node_i.instantiate_values()
                print(node_i.name, "independent", list(df.columns))
            print("----"*10+"\n")
        self.df = df

    def save_to_disc(self, file_name_with_path, excluded_cols=[], shorten_num_cols=True):
        if not os.path.exists(file_name_with_path):
            directory = os.path.dirname(file_name_with_path)
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        if shorten_num_cols:
            self.df[self.num_cols] = self.df[self.num_cols].round(3)
        if excluded_cols:
            self.df.drop(columns=excluded_cols).to_csv(file_name_with_path, index=False)
        else:
            self.df.to_csv(file_name_with_path, index=False)
        print('--> Generated data is saved to ', file_name_with_path, '\n')


if __name__ == '__main__':
    # initialize nodes
    total_n = 2
    node_diversity = CategoricalNode("diversity", 
                                    {"White": 0.1, "B": 0.4, "A":0.2, "H":0.1,
                                    "I":0.05, "O":0.15}, 
                                    sample_n=total_n)

    node_test = GaussianNode("test", miu = 90,var = 10**2, sample_n = total_n)

    node_test2 = GaussianNode("test2", miu = 90,var = 10**2, sample_n = total_n)

    node_toefl = GaussianNode("TOEFL", miu=90, var=10**2, sample_n=total_n)

    node_admission = CategoricalNode("admission", {"Y": 0.5, "N": 0.5}, sample_n=total_n)

    edge_diversity_admission = CtoC("diversity", "admission", {"White": {"Y": 0.3, "N": 0.7}, 
                                                            "B": {"Y": 0.7, "N": 0.3},
                                                            "A": {"Y": 0.2, "N": 0.8}, 
                                                            "H": {"Y": 0.5, "N": 0.5},
                                                            "I": {"Y": 0.4, "N": 0.6},
                                                            "O": {"Y": 0.9, "N": 0.1}})
    edge_toefl_admission = NtoC("TOEFL", "admission", [100], [{"Y": 0.6, "N": 0.4}, {"Y": 0.4, "N": 0.6}])

    edge_test_toefl = NtoN("test","TOEFL",[80],[["Gaussian",0,1,None,None,None],["Gaussian",0,1,None,None,None]])
    edge_test2_toefl = NtoN("test2","TOEFL",[80],[["Gaussian",0,1,None,None,None],["Gaussian",0,1,None,None,None]])


    nodes = [node_test, node_test2, node_diversity,node_toefl,node_admission]
    edge_relations = {"admission":([edge_diversity_admission,edge_toefl_admission],[0.5,0.5]),
                      "TOEFL":([edge_test_toefl, edge_test2_toefl],[0.5,0.5])}

    mirror = Mirror(seed=0)
    mirror.generate_csv(nodes, edge_relations, "config.yml")
    print(mirror.df)
    #mirror.save_to_disc("../out/synthetic_data/test/sample2.csv")



    # node_g = CategoricalNode("G", {"M": 0.5, "F": 0.5}, sample_n=total_n)
    # node_r = CategoricalNode("R", {"W", "B"})
    #
    # node_a = OrdinalGlobalNode("A", min=20, max=70)

    # node_x = GaussianNode("X")
    # node_y = GaussianNode("Y")

    # initialize edges
    # edge_g_r = CtoC("G", "R", {"M": {"W": 0.635, "B": 0.365}, "F": {"W": 0.622, "B": 0.378}})
    #
    # edge_r_a = CtoN("R", "A", {"W": ["Gaussian", 30, 10], "B": ["Gaussian", 45, 10]})
    #
    # edge_g_x = CtoN("G", "X", {"M": ["Gaussian", 0, 0.5], "F": ["Gaussian", -1, 1]})
    #
    #
    # edge_r_x = CtoN("R", "X", {"W": ["Gaussian", 0, 0.5], "B": ["Gaussian", -1, 1]})
    #
    # edge_a_x = CtoN("G", "X", {"M": ["Gaussian", 0, 1], "F": ["Gaussian", -1, 1]})
    #
    #
    #
    #
    # edge_g_y = CtoN("G", "Y", {"M": ["Gaussian", 0, 0.5], "F": ["Gaussian", -1, 1]})
    # edge_r_y = CtoN("R", "Y", {"W": ["Gaussian", 0, 0.5], "B": ["Gaussian", -1, 1]})
    #
    # edge_x_y = NtoNLinear("X", "Y")

    # define DAG
    # nodes = [node_g, node_r, node_x, node_y]
    # edge_relations = {"X": ([edge_g_x, edge_r_x], [0.5, 0.5]),
    #                   "Y": ([edge_g_y, edge_r_y, edge_x_y], [0.2, 0.2, 0.6])}

    # mirror = Mirror(seed=0)
    # mirror.generate_csv(nodes, edge_relations)
    # mirror.save_to_disc("../out/synthetic_data/test/R_pareto.csv")




    # total_n = 100000
    #
    # # initialize nodes
    # node_g = CategoricalNode("G", {"M": 0.5, "F": 0.5}, sample_n=total_n)
    # node_r = CategoricalNode("R", {"W": 0.5, "B": 0.5}, sample_n=total_n)
    # node_a = OrdinalGlobalNode("A", sample_n=total_n, min=15, max=80)
    #
    # node_d = CategoricalNode("D", {"Y": 0.5, "N": 0.5})
    #
    # # initialize edges
    # # edge_g_x = CtoN("G", "X", {"M": ["Gaussian", 0, 1], "F": ["Gaussian", -1, 1]})
    # # edge_r_x = CtoN("R", "X", {"W": ["Gaussian", 0, 1], "B": ["Gaussian", -1, 1]})
    #
    # edge_g_d = CtoC("G", "D", {"M": {"Y": 0.7, "N": 0.3}, "F": {"Y": 0.3, "N": 0.7}})
    # edge_r_d = CtoC("R", "D", {"W": {"Y": 0.7, "N": 0.3}, "B": {"Y": 0.3, "N": 0.7}})
    #
    # # edge_a_d = NtoC("A", "D", [25, 45, 65], [{"Y": 0.2, "N": 0.8}, {"Y": 0.7, "N": 0.3}, {"Y": 0.6, "N": 0.4}, {"Y": 0.3, "N": 0.7}])
    # edge_a_d = NtoC("A", "D", [45], [{"Y": 0.7, "N": 0.3}, {"Y": 0.3, "N": 0.7}])
    # # define DAG
    # nodes = [node_g, node_r, node_a, node_d]
    # edge_relations = {"D": [edge_g_d, edge_r_d, edge_a_d]}
    #
    # # nodes = [node_g, node_r, node_d]
    # # edge_relations = {"D": [edge_g_d, edge_r_d]}
    #
    # mirror = Mirror(seed=0)
    # mirror.generate_csv(nodes, edge_relations)
    #
    # # mirror.save_to_disc("../out/test/R1.csv")
    # print(mirror.df.columns)
    # test_df = mirror.df.copy()
    # test_df["GRA"] = test_df["G"] + test_df["R"] + test_df["C_A"]
    # test_df["All"] = test_df["G"] + test_df["R"] + + test_df["C_A"] + test_df["D"]
    # # test_df["All"] = test_df["G"] + test_df["D"]
    #
    # print (test_df["GRA"].value_counts(normalize=True))
    # print (test_df["All"].value_counts(normalize=True))
    # # print(mirror.df[["group", "D", "A"]].groupby(by=["group", "D"]).count())