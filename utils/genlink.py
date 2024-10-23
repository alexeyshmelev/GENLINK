import gc
import os
import json
import time
import torch
import pickle
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import networkx as nx
import torch.nn as nn
from sklearn import metrics
from multiprocessing import Pool
from collections import OrderedDict, Counter
import numba
from numba import njit, prange
# numba.config.THREADING_LAYER = 'workqueue'
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import torch.nn.functional as F
from scipy.spatial.distance import squareform
from torch.utils.data import TensorDataset, DataLoader
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import bernoulli, expon, norm, powerlaw, pearsonr
from sklearn.model_selection import train_test_split
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import one_hot
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.cluster import homogeneity_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from torch.nn import Linear, LayerNorm, BatchNorm1d, Sequential, LeakyReLU, Dropout
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, NNConv, SGConv, ARMAConv, TAGConv, ChebConv, DNAConv, LabelPropagation, \
EdgeConv, FiLMConv, FastRGCNConv, SSGConv, SAGEConv, GATv2Conv, BatchNorm, GraphNorm, MemPooling, SAGPooling, GINConv, CorrectAndSmooth

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def symmetrize(m):
    return m + m.T - np.diag(m.diagonal())


def generate_matrices_fn(population_sizes, offset, edge_probs, mean_weight, rng):
    '''
        main simulation function
    
    Parameters
    ----------
    population_sizes: list 
        list of population sizes
    offset: float
        we assume ibdsum pdf = lam*exp(-lam*(x-offset)) for x>offset and 0 otherwise, lam = 1/mean
    edge_probs: 2d array
        probability of an edge between classes
    mean_weight: 2d array
        mean weight of an existing edge between classes (corrected by offset)
    rng: random number generator
    
    Returns
    -------
    counts: 
        
    sums: 
        
    pop_index: 1d np array
        population index of every node
        
    '''
    p = edge_probs
    teta = mean_weight
    pop_index = []
    n_pops = len(population_sizes)
    for i in range(n_pops):
        pop_index += [i] * population_sizes[i]

    pop_index = np.array(pop_index)
    #print(f"{n_pops=}")
    blocks_sums = [[np.zeros(shape=(population_sizes[i], population_sizes[j])) for j in range(n_pops)] for i in
                   range(n_pops)]
    blocks_counts = [[np.zeros(shape=(population_sizes[i], population_sizes[j])) for j in range(n_pops)] for i
                     in range(n_pops)]

    #print(np.array(blocks_sums).shape)

    for pop_i in range(n_pops):
        for pop_j in range(pop_i + 1):
            if p[pop_i, pop_j] == 0:
                continue
            #print(f"{pop_i=} {pop_j=}")
            pop_cross = population_sizes[pop_i] * population_sizes[pop_j]
            #TODO switch to rng.binomial or something
            bern_samples =  rng.binomial(1, p[pop_i, pop_j], pop_cross) #bernoulli.rvs(p[pop_i, pop_j], size=pop_cross)
            total_segments = np.sum(bern_samples)
            #print(f"{total_segments=}")
            exponential_samples = rng.exponential(teta[pop_i, pop_j], size=total_segments) + offset
            #position = 0
            exponential_totals_samples = np.zeros(pop_cross, dtype=np.float64)
            #mean_totals_samples = np.zeros(pop_cross, dtype=np.float64)
            exponential_totals_samples[bern_samples == 1] = exponential_samples

            bern_samples = np.reshape(bern_samples, newshape=(population_sizes[pop_i], population_sizes[pop_j]))
            exponential_totals_samples = np.reshape(exponential_totals_samples,
                                                    newshape=(population_sizes[pop_i], population_sizes[pop_j]))
            if (pop_i == pop_j):
                bern_samples = np.tril(bern_samples, -1)
                exponential_totals_samples = np.tril(exponential_totals_samples, -1)
            blocks_counts[pop_i][pop_j] = bern_samples
            blocks_sums[pop_i][pop_j] = exponential_totals_samples
    
    
    full_blocks_counts = np.block(blocks_counts)
    full_blocks_sums = np.block(blocks_sums)
    return np.nan_to_num(symmetrize(full_blocks_counts)), np.nan_to_num(symmetrize(full_blocks_sums)), pop_index


def simulate_graph_fn(classes, means, counts, pop_index, path):
    '''
        store simulated dataframe
    
    Parameters
    ----------
    classes: list of str
        names of populations
    means: 2d np array
        0: no link between i-th and j-th individuals
    counts: 2d np array
        ibd sum between i-th and j-th individuals
    pop_index: 1d np array
        population index of every node
    path: string
        csv file to store dataframe
    '''
    indiv = list(range(counts.shape[0]))
    with open(path, 'w', encoding="utf-8") as f:
        f.write('node_id1,node_id2,label_id1,label_id2,ibd_sum,ibd_n\n')
        for i in range(counts.shape[0]):
            for j in range(i):
                if (counts[i][j]):
                    name_i = classes[pop_index[i]] if "," not in classes[pop_index[i]] else '\"' + classes[pop_index[i]] + '\"'
                    name_j = classes[pop_index[j]] if "," not in classes[pop_index[j]] else '\"' + classes[pop_index[j]] + '\"'
                    #f.write(f'node_{i},node_{j},{name_i},{name_j},{means[i][j]},{counts[i][j]}\n')
                    f.write(f'node_{i},node_{j},{name_i},{name_j},{means[i][j]},1\n')



class DataProcessor:
    def __init__(self, path, is_path_object=False, disable_printing=True, dataset_name=None):
        self.dataset_name: str = dataset_name
        self.train_size: float = None
        self.valid_size: float = None
        self.test_size: float = None
        self.mask_size: float = None
        self.sub_train_size: float = None
        self.edge_probs = None
        self.mean_weight = None
        self.offset = 6.0
        self.df = path if is_path_object else pd.read_csv(path)
        self.node_names_to_int_mapping: dict[str, int] = self.get_node_names_to_int_mapping(self.get_unique_nodes(self.df))
        self.classes: list[str] = self.get_classes(self.df)
        self.node_classes_sorted: pd.DataFrame = self.get_node_classes(self.df)
        self.class_to_int_mapping: dict[int, str] = {i:n for i, n in enumerate(self.classes)}
        self.class_colors = self.get_class_colors()
        self.nx_graph = self.make_networkx_graph() # line order matters because self.df is modified in above functions
        self.train_nodes = None
        self.valid_nodes = None
        self.test_nodes = None
        self.mask_nodes = None
        self.array_of_graphs_for_training = []
        self.array_of_graphs_for_validation = []
        self.array_of_graphs_for_testing = []
        self.disable_printing = disable_printing
        # self.rng = np.random.default_rng(42)
        
    def get_class_colors(self):
        colors = ['#00ff72', '#004eff', '#a900ff', '#ff002f', '#ffc800', '#00ffff', '#6f6f6f', '#ff9900']
        assert len(self.classes) <= len(colors)
        return {self.classes[i]:colors[i] for i in range(len(self.classes))}
    
    def make_networkx_graph(self):
        G = nx.from_pandas_edgelist(self.df, source='node_id1', target='node_id2', edge_attr=['ibd_sum', 'ibd_n'])
        node_attr = dict()
        for i in range(self.node_classes_sorted.shape[0]):
            row = self.node_classes_sorted.iloc[i, :]
            node_attr[row[0]] = {'class':row[1]}
        nx.set_node_attributes(G, node_attr)
        return G
        
    def get_classes(self, df):
        classes = pd.concat([df['label_id1'], df['label_id2']], axis=0).unique().tolist()
        if 'masked' in classes:
            classes = classes + ['masked']
            classes.remove('masked')
        return classes

    def get_unique_nodes(self, df):
        return pd.concat([df['node_id1'], df['node_id2']], axis=0).drop_duplicates().to_numpy()

    def get_node_names_to_int_mapping(self, unique_nodes):
        # d = torch.load(r"C:\HSE\genotek-nationality-analysis\data\mapping_indices.pt")
        # d = {'node_'+str(k):v for k, v in d.items()}
        # return d
        return {n:i for i, n in enumerate(unique_nodes)}

    def relabel(self, df):
        '''
        Replace any node names with continious int numbers
        :param df: initial DataFrame
        :return: same DataFrame but with new labels
        '''
        df.iloc[:, 0] = df.iloc[:, 0].apply(lambda n: self.node_names_to_int_mapping[n])
        df.iloc[:, 1] = df.iloc[:, 1].apply(lambda n: self.node_names_to_int_mapping[n])
        return df

    def class_labels_to_int(self, df):
        df.iloc[:, 2] = df.iloc[:, 2].apply(lambda t: self.classes.index(t))
        df.iloc[:, 3] = df.iloc[:, 3].apply(lambda t: self.classes.index(t))
        return df

    def get_node_classes(self, df):
        self.df = self.relabel(self.df)
        self.df = self.class_labels_to_int(self.df)

        n = pd.concat([df['node_id1'], df['node_id2']], axis=0)

        l = pd.concat([df['label_id1'], df['label_id2']], axis=0)

        df_node_classes = pd.concat([n, l], axis=1).drop_duplicates()

        df_node_classes.columns = ['node', 'class_id']

        return df_node_classes.sort_values(by=['node'])

    def node_classes_to_dict(self):
        return {n: c for index, pair in self.node_classes_sorted.iterrows() for n, c in [pair.tolist()]}
        
    def generate_random_train_valid_test_nodes(self, train_size, valid_size, test_size, random_state, save_dir=None, mask_size=None, sub_train_size=None, keep_train_nodes=True, mask_random_state=None):
        if train_size + valid_size + test_size != 1.0:
            raise Exception("All sizes should add up to 1.0!")

        if mask_size is not None and sub_train_size is not None:
            assert mask_size <= 1.
            assert sub_train_size < 1.
        elif mask_size is None and sub_train_size is None:
            pass
        else:
            raise Exception('Impossible parameter configuration!')

        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        if mask_size is not None:
            self.mask_size = mask_size
            self.sub_train_size = sub_train_size
        num_nodes_per_class = self.node_classes_sorted.iloc[:, 1].value_counts()
        node_classes_random = self.node_classes_sorted.sample(frac=1, random_state=mask_random_state if (mask_size is not None) and keep_train_nodes else random_state)
        self.train_nodes, self.valid_nodes, self.test_nodes = [], [], []
        if mask_size is not None:
            self.mask_nodes = []
            train_nodes_for_mask_selection = []
        node_counter = {i: 0 for i in range(num_nodes_per_class.shape[0])}
        # print('NODE COUNTER SHAPE', len(node_counter))
        # if mask_size is not None:
        #     node_counter['masked_nodes'] = 0

        for i in range(node_classes_random.shape[0]):
            node_class = node_classes_random.iloc[i, 1]
            if node_counter[node_class] <= int(self.train_size * num_nodes_per_class.loc[node_class]):
                if self.mask_size is not None:
                    if node_counter[node_class] <= int(self.sub_train_size * self.train_size * num_nodes_per_class.loc[node_class]):
                        self.train_nodes.append(node_classes_random.iloc[i, 0])
                    else:
                        train_nodes_for_mask_selection.append(node_classes_random.iloc[i, 0])
                else:
                    self.train_nodes.append(node_classes_random.iloc[i, 0])
                node_counter[node_class] += 1
            elif int((self.train_size + self.valid_size) * num_nodes_per_class.loc[node_class]) >= node_counter[node_class] > int(self.train_size * num_nodes_per_class.loc[node_class]):
                self.valid_nodes.append(node_classes_random.iloc[i, 0])
                node_counter[node_class] += 1
            else:
                self.test_nodes.append(node_classes_random.iloc[i, 0])

        if mask_size is not None:
            node_classes_sorted_masks = self.node_classes_sorted.iloc[train_nodes_for_mask_selection, :].reset_index(drop=True)
            node_classes_masks_random = node_classes_sorted_masks.sample(frac=1, random_state=random_state) # check that multible random state behevior
        
            num_nodes_per_class_mask = node_classes_sorted_masks.iloc[:, 1].value_counts()
            node_counter_mask = {i: 0 for i in range(num_nodes_per_class_mask.shape[0])}
            for i in range(node_classes_masks_random.shape[0]):
                node_class = node_classes_masks_random.iloc[i, 1]
                if node_counter_mask[node_class] < int(self.mask_size * num_nodes_per_class_mask.loc[node_class]):
                    self.mask_nodes.append(node_classes_masks_random.iloc[i, 0])
                node_counter_mask[node_class] += 1

            if self.mask_size == 0.0:
                assert len(self.mask_nodes) == 0

            if self.mask_size == 1.0:
                assert len(self.train_nodes + self.valid_nodes + self.test_nodes + self.mask_nodes) == self.node_classes_sorted.shape[0]

        if mask_size is not None:
            print(f'{len(set(self.train_nodes + self.valid_nodes + self.test_nodes + self.mask_nodes)) / self.node_classes_sorted.shape[0] * 100}% of all nodes in dataset were used')
        else:
            print(f'{len(set(self.train_nodes + self.valid_nodes + self.test_nodes)) / self.node_classes_sorted.shape[0] * 100}% of all nodes in dataset were used')

        if save_dir is not None:
            with open(save_dir + '/train.pickle', 'wb') as handle:
                pickle.dump(self.train_nodes, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(save_dir + '/valid.pickle', 'wb') as handle:
                pickle.dump(self.valid_nodes, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(save_dir + '/test.pickle', 'wb') as handle:
                pickle.dump(self.test_nodes, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load_train_valid_test_nodes(self, train_path, valid_path, test_path, data_type, mask_path=None):
        if data_type == 'pickle':
            with open(train_path, 'rb') as handle:
                self.train_nodes = pickle.load(handle)
            with open(valid_path, 'rb') as handle:
                self.valid_nodes = pickle.load(handle)
            with open(test_path, 'rb') as handle:
                self.test_nodes = pickle.load(handle)
        if data_type == 'torch':
            self.train_nodes = torch.load(train_path).tolist()
            self.valid_nodes = torch.load(valid_path).tolist()
            self.test_nodes = torch.load(test_path).tolist()
        if data_type == 'numpy':
            self.train_nodes = [self.node_names_to_int_mapping[f'node_{node}'] for node in train_path] # for numpy it is not a path, it is actual array
            self.valid_nodes = [self.node_names_to_int_mapping[f'node_{node}'] for node in valid_path]
            self.test_nodes = [self.node_names_to_int_mapping[f'node_{node}'] for node in test_path]
            if mask_path is not None:
                for mask_node in mask_path:
                    if f'node_{mask_node}' not in self.node_names_to_int_mapping.keys():
                        if not self.disable_printing:
                            print(f'Adding isolated masked node {mask_node}')
                        self.node_names_to_int_mapping[f'node_{mask_node}'] = len(self.node_names_to_int_mapping)
                        self.node_classes_sorted = pd.concat([self.node_classes_sorted, pd.DataFrame([[self.node_names_to_int_mapping[f'node_{mask_node}'], self.classes.index('masked')]], columns=['node', 'class_id'])], axis=0)
                        self.node_classes_sorted = self.node_classes_sorted.sort_values(by=['node'])
                self.mask_nodes = [self.node_names_to_int_mapping[f'node_{node}'] for node in mask_path]
                self.train_nodes = list(filter(lambda node: node not in self.mask_nodes, self.train_nodes))
                self.valid_nodes = list(filter(lambda node: node not in self.mask_nodes, self.valid_nodes))
                self.test_nodes = list(filter(lambda node: node not in self.mask_nodes, self.test_nodes))
                
                # for node in train_path:
                #     if node in self.mask_nodes:
                #         print('Removing mask node from train nodes')
                #         self.mask_nodes.remove(node)
                
                for node in valid_path:
                    if node in self.mask_nodes:
                        if not self.disable_printing:
                            print('Removing mask node from validation nodes')
                        self.mask_nodes.remove(node)
                        
                for node in test_path:
                    if node in self.mask_nodes:
                        if not self.disable_printing:
                            print('Removing mask node from test nodes')
                        self.mask_nodes.remove(node)
            
            if self.mask_nodes is not None:
                assert len(self.mask_nodes) > 0
                assert len(set(self.train_nodes) & set(self.mask_nodes)) == 0
            assert len(self.train_nodes) > 0
            assert len(self.valid_nodes) > 0
            assert len(self.test_nodes) > 0
                
            
        # if data_type == 'object':
        #     self.train_nodes = train_path
        #     self.valid_nodes = valid_path
        #     self.test_nodes = test_path

        if not (type(self.train_nodes) == list and type(self.valid_nodes) == list and type(self.test_nodes) == list):
            raise Exception('Node ids must be stored in Python lists!')
        if len(set(self.train_nodes + self.valid_nodes + self.test_nodes)) < (len(self.train_nodes) + len(self.valid_nodes) + len(self.test_nodes)):
            print('There is intersection between train, valid and test node sets!') 
            
    def number_of_multi_edges(self, G):
        s = []
        for a, b in list(G.edges):
            if a < b:
                ts = f'{a},{b}'
            else:
                ts = f'{b},{a}'
            s.append(ts)
        df = pd.DataFrame(s)
        c = df.pivot_table(index = 0, aggfunc ='size')

        counter = 0
        for i in range(c.shape[0]):
            if c.iloc[i] > 1:
                counter += 1
        return counter
    
    def addlabels(self, ax, x, y, t):
        for i in range(len(x)):
            ax.text(i, y[i] + 2 if t == 0 else y[i] + 20, y[i], ha = 'center', fontsize=8)
            
    def get_graph_features(self, fig_path, fig_size, picture_only=False, dataset_name=None):
        
        graph_node_classes = nx.get_node_attributes(self.nx_graph, 'class')
        class_counts = dict()
        for k, v in graph_node_classes.items():
            if self.class_to_int_mapping[v] not in class_counts.keys():
                class_counts[self.class_to_int_mapping[v]] = 1
            else:
                class_counts[self.class_to_int_mapping[v]] += 1
                
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.bar(range(len(class_counts.keys())), list(class_counts.values()), color='#253957')
        ax.set_xticks(range(len(class_counts.keys())))
        ax.set_xticklabels(class_counts.keys(), rotation = 90, ha='right')
        self.addlabels(ax, list(class_counts.keys()), list(class_counts.values()), 0)
        ax.set_title(f'Class distribution ({dataset_name})')
        ax.set_xlabel(f'Population groups')
        ax.set_ylabel(f'Amount of nodes')
        plt.savefig(f'{fig_path}num_nodes_per_classes_and_dataset.pdf', bbox_inches='tight')
        plt.show()
        
        ##########
        
        def smallest_degree(G):
            return min(G, key=G.degree)
        
        rcm = list(nx.utils.cuthill_mckee_ordering(self.nx_graph, heuristic=None))
        
        B = nx.adjacency_matrix(self.nx_graph, nodelist=rcm)
        img, ax = plt.subplots(1, 1, figsize=(20, 20))
        pic = sns.heatmap(B.todense(), cbar=False, square=True, linewidths=0, annot=False, cmap=mcm.gray, ax=ax)
        
        node_classes = nx.get_node_attributes(self.nx_graph, "class")
        
        ax.set_xticks(range(len(rcm)))
        ax.set_xticklabels([f'{self.class_to_int_mapping[node_classes[node]]} ({node})' for node in rcm], fontsize=2)
        colors = [self.class_colors[self.class_to_int_mapping[node_classes[node]]] for node in rcm]
        for xtick, color in zip(ax.get_xticklabels(), colors):
            xtick.set_color(color)
            
        ax.set_yticks(range(len(rcm)))
        ax.set_yticklabels([f'{self.class_to_int_mapping[node_classes[node]]} ({node})' for node in rcm], fontsize=2)
        colors = [self.class_colors[self.class_to_int_mapping[node_classes[node]]] for node in rcm]
        for ytick, color in zip(ax.get_yticklabels(), colors):
            ytick.set_color(color)
            
        # pic.set(xticklabels=[], yticklabels=[])
        # ax.tick_params(left=False, bottom=False)
        ax.set_title(f'Cuthill-McKee adjacency matrix ({dataset_name})')
        plt.savefig(f'{fig_path}adjacency_matrix.png', bbox_inches='tight', dpi=400) # pdf is too heavy
        plt.show()
        
        ##########
        
        rcm = list(nx.utils.cuthill_mckee_ordering(self.nx_graph, heuristic=smallest_degree))
        
        B = nx.adjacency_matrix(self.nx_graph, nodelist=rcm)
        img, ax = plt.subplots(1, 1, figsize=(20, 20))
        pic = sns.heatmap(B.todense(), cbar=False, square=True, linewidths=0, annot=False, cmap=mcm.gray, ax=ax)
        
        node_classes = nx.get_node_attributes(self.nx_graph, "class")
        
        ax.set_xticks(range(len(rcm)))
        ax.set_xticklabels([f'{self.class_to_int_mapping[node_classes[node]]} ({node})' for node in rcm], fontsize=2)
        colors = [self.class_colors[self.class_to_int_mapping[node_classes[node]]] for node in rcm]
        for xtick, color in zip(ax.get_xticklabels(), colors):
            xtick.set_color(color)
            
        ax.set_yticks(range(len(rcm)))
        ax.set_yticklabels([f'{self.class_to_int_mapping[node_classes[node]]} ({node})' for node in rcm], fontsize=2)
        colors = [self.class_colors[self.class_to_int_mapping[node_classes[node]]] for node in rcm]
        for ytick, color in zip(ax.get_yticklabels(), colors):
            ytick.set_color(color)
  
        ax.set_title(f'Cuthill-McKee adjacency matrix by smallest degree ({dataset_name})')
        plt.savefig(f'{fig_path}adjacency_matrix_smallest_degree.png', bbox_inches='tight', dpi=400) # pdf is too heavy
        plt.show()
        
        ##########
        
        features = dict()
        
        G = self.nx_graph
        
        if not picture_only: # remove that line
        
            features['Number of nodes'] = G.number_of_nodes()
            features['Number of edges'] = G.number_of_edges()
            features['Density'] = nx.density(G)
            features['Self-loop edges'] = list(nx.selfloop_edges(G))
            features['Is connected'] = nx.is_connected(G)
            features['Number of cc'] = nx.number_connected_components(G)
            features['Number of isolated nodes'] = nx.number_of_isolates(G)
            features['Is planar'] = nx.is_planar(G)

            if features['Number of cc'] > 1:
                G = G.subgraph(max(nx.connected_components(G))).copy()
                # mapping = {on:f'{nn}' for nn, on in enumerate(G.nodes())}
                # G = nx.relabel_nodes(G, mapping)
                features['Number of nodes in largest cc'] = G.number_of_nodes()

            features['Diameter'] = nx.diameter(G)
            features['Radius'] = nx.radius(G)
            features['Transitivity'] = nx.transitivity(G)
            features['Number of multi edges'] = self.number_of_multi_edges(G)

            degrees_of_G = [d for node, d in G.degree()]
            features['Max degree'] = np.max(degrees_of_G)
            features['Mean degree'] = np.mean(degrees_of_G)
            features['Min degree'] = np.min(degrees_of_G)

            features['Global efficiency'] = nx.global_efficiency(G)
            features['Local efficiency'] = nx.local_efficiency(G)
            features['Degree assortativity coefficient'] = nx.degree_assortativity_coefficient(G)
            features['Class assortativity coefficient'] = nx.attribute_assortativity_coefficient(G, "class")
            features['Average clustering'] = nx.average_clustering(G)
            features['Center'] = list(nx.center(G))
            features['Periphery'] = nx.periphery(G)
            features['Is Eulerian'] = nx.is_eulerian(G)
            features['Is semi-Eulerian'] = nx.is_semieulerian(G)
            features['Is regular'] = nx.is_regular(G)
            features['Average shortest path length'] = nx.average_shortest_path_length(G)
            features['Weighted average shortest path length'] = nx.average_shortest_path_length(G, weight='ibd_sum')
            features['Is tree'] = nx.is_tree(G)
            features['Is forest'] = nx.is_forest(G)

            A = nx.to_numpy_array(G)
            degrees = np.sum(A, axis=1)
            numerator = 0
            n = len(G.nodes)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        numerator += A[i, j] * A[j, k] * A[k, i]

            denominator = 0.5 * np.sum(degrees * (degrees - 1))

            if denominator > 0:
                features['global_clustering_coefficient'] = numerator / denominator
            else:
                features['global_clustering_coefficient'] = 0
            
            cd = nx.degree_centrality(G)
            cda = []
            for node in G.nodes:
                cda.append(cd[node])

            cda = np.array(cda)
            features['Max degree centrality'] = np.max(cda)
            features['Mean degree centrality'] = np.mean(cda)
            features['Min degree centrality'] = np.min(cda)
            
            ce = nx.eigenvector_centrality(G)
            cea = []
            for node in G.nodes:
                cea.append(ce[node])

            cea = np.array(cea)
            features['Max eigenvector centrality'] = np.max(cea)
            features['Mean eigenvector centrality'] = np.mean(cea)
            features['Min eigenvector centrality'] = np.min(cea)
            
            ccl = nx.closeness_centrality(G)
            ccla = []
            for node in G.nodes:
                ccla.append(ccl[node])

            ccla = np.array(ccla)
            features['Max closeness centrality'] = np.max(ccla)
            features['Mean closeness centrality'] = np.mean(ccla)
            features['Min closeness centrality'] = np.min(ccla)
            
            cb = nx.betweenness_centrality(G)
            cba = []
            for node in G.nodes:
                cba.append(cb[node])

            cba = np.array(cba)
            features['Max betweenness centrality'] = np.max(cba)
            features['Mean betweenness centrality'] = np.mean(cba)
            features['Min betweenness centrality'] = np.min(cba)
            
            ck = nx.katz_centrality_numpy(G)
            cka = []
            for node in G.nodes:
                cka.append(ck[node])

            cka = np.array(cka)
            features['Max katz centrality'] = np.max(cka)
            features['Mean katz centrality'] = np.mean(cka)
            features['Min katz centrality'] = np.min(cka)
            
            communities_per_class_dict = dict()
            nx_graph_node_classes = nx.get_node_attributes(G, 'class')
            for node, node_class in nx_graph_node_classes.items():
                if node_class not in communities_per_class_dict.keys():
                    communities_per_class_dict[node_class] = set([node])
                else:
                    communities_per_class_dict[node_class].add(node)
                    
            features['class_partition_modularity'] = nx.community.modularity(G, list(communities_per_class_dict.values()))
            features['Max largest clique'] = max(nx.find_cliques(G), key=len)
            
            features['Center_node_description'] = dict()
            features['Periphery_node_description'] = dict()
            features['Max_degree_centrality_node'] = dict()
            features['Max_eigenvector_centrality_node'] = dict()
            features['Max_closeness_centrality_node'] = dict()
            features['Max_betweenness_centrality_node'] = dict()
            features['Max_katz_centrality_node'] = dict()

            node_classes = nx.get_node_attributes(G, "class")
            for node in features['Center']:
                features['Center_node_description'][str(node)] = [self.class_to_int_mapping[node_classes[node]], f'degree: {G.degree[node]}', f'degree centrality: {cd[node]}', f'eigenvector centrality: {ce[node]}', f'closeness centrality: {ccl[node]}', f'betweenness centrality: {cb[node]}', f'katz centrality: {ck[node]}']
                
            for node in features['Periphery']:
                features['Periphery_node_description'][str(node)] = [self.class_to_int_mapping[node_classes[node]], f'degree: {G.degree[node]}', f'degree centrality: {cd[node]}', f'eigenvector centrality: {ce[node]}', f'closeness centrality: {ccl[node]}', f'betweenness centrality: {cb[node]}', f'katz centrality: {ck[node]}']
            
            max_centrality = -np.inf
            selected_node = None
            for node, centrality in cd.items():
                if centrality > max_centrality:
                    max_centrality = centrality
                    selected_node = node
            features['Max_degree_centrality_node'][str(selected_node)] = [self.class_to_int_mapping[node_classes[selected_node]], f'degree: {G.degree[selected_node]}', f'degree centrality: {cd[selected_node]}', f'eigenvector centrality: {ce[selected_node]}', f'closeness centrality: {ccl[selected_node]}', f'betweenness centrality: {cb[selected_node]}', f'katz centrality: {ck[selected_node]}']
            
            max_centrality = -np.inf
            selected_node = None
            for node, centrality in ce.items():
                if centrality > max_centrality:
                    max_centrality = centrality
                    selected_node = node
            features['Max_eigenvector_centrality_node'][str(selected_node)] = [self.class_to_int_mapping[node_classes[selected_node]], f'degree: {G.degree[selected_node]}', f'degree centrality: {cd[selected_node]}', f'eigenvector centrality: {ce[selected_node]}', f'closeness centrality: {ccl[selected_node]}', f'betweenness centrality: {cb[selected_node]}', f'katz centrality: {ck[selected_node]}']
            
            max_centrality = -np.inf
            selected_node = None
            for node, centrality in ccl.items():
                if centrality > max_centrality:
                    max_centrality = centrality
                    selected_node = node
            features['Max_closeness_centrality_node'][str(selected_node)] = [self.class_to_int_mapping[node_classes[selected_node]], f'degree: {G.degree[selected_node]}', f'degree centrality: {cd[selected_node]}', f'eigenvector centrality: {ce[selected_node]}', f'closeness centrality: {ccl[selected_node]}', f'betweenness centrality: {cb[selected_node]}', f'katz centrality: {ck[selected_node]}']
            
            max_centrality = -np.inf
            selected_node = None
            for node, centrality in cb.items():
                if centrality > max_centrality:
                    max_centrality = centrality
                    selected_node = node
            features['Max_betweenness_centrality_node'][str(selected_node)] = [self.class_to_int_mapping[node_classes[selected_node]], f'degree: {G.degree[selected_node]}', f'degree centrality: {cd[selected_node]}', f'eigenvector centrality: {ce[selected_node]}', f'closeness centrality: {ccl[selected_node]}', f'betweenness centrality: {cb[selected_node]}', f'katz centrality: {ck[selected_node]}']
            
            max_centrality = -np.inf
            selected_node = None
            for node, centrality in ck.items():
                if centrality > max_centrality:
                    max_centrality = centrality
                    selected_node = node
            features['Max_katz_centrality_node'][str(selected_node)] = [self.class_to_int_mapping[node_classes[selected_node]], f'degree: {G.degree[selected_node]}', f'degree centrality: {cd[selected_node]}', f'eigenvector centrality: {ce[selected_node]}', f'closeness centrality: {ccl[selected_node]}', f'betweenness centrality: {cb[selected_node]}', f'katz centrality: {ck[selected_node]}']

            # features['PageRank'] = nx.pagerank(G, alpha=0.8)
        
        cc = []
        for node in G.nodes:
            cc.append(nx.clustering(G, node))
        
        ##########
        G_max_clique = G.subgraph(features['Max largest clique'])
        pos = nx.spring_layout(G_max_clique, iterations=10)
        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.axis('off')
        plt.title(f'Max largest clique ({dataset_name})')

        node_classes = nx.get_node_attributes(G_max_clique, "class")
        unique_node_classes = np.unique(list(node_classes.values())).astype(int)
        unique_node_classes = np.array([self.class_to_int_mapping[c] for c in unique_node_classes])
        current_node_colors = []
        for node in G_max_clique.nodes:
            current_node_colors.append(self.class_colors[self.class_to_int_mapping[node_classes[node]]])
        nx.draw_networkx_nodes(G_max_clique, pos=pos, node_color=current_node_colors, node_size=304, ax=ax)
        nx.draw_networkx_labels(G_max_clique, pos=pos, labels={node:node for node in G_max_clique.nodes}, font_size=8, font_color='w')

        nx.draw_networkx_edges(G_max_clique, pos=pos, alpha=0.15, width=1, edge_cmap=plt.cm.Greys, edge_color=list(nx.get_edge_attributes(G_max_clique, 'ibd_sum').values()), ax=ax)

        for k, v in self.class_colors.items():
            if k in unique_node_classes:
                plt.scatter([],[], c=v, label=k)

        plt.legend()

        plt.savefig(f'{fig_path}largest_clique.pdf')
        plt.show()
        
        ##########
        
        spl = []
        splw = []
        spl_dict = dict(nx.shortest_path_length(G))
        splw_dict = dict(nx.shortest_path_length(G, weight='ibd_sum'))
        for source in G.nodes:
            for target in G.nodes:
                spl_curr = spl_dict[source][target]
                splw_curr = splw_dict[source][target]
                if spl_curr:
                    spl.append(spl_curr)
                if splw_curr:
                    splw.append(splw_curr)
        
        plt.clf()
        img, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.set_title(f'Distribution of shortest paths ({dataset_name})')
        n, bins, patches = ax.hist(spl, bins=100, color='#253957', edgecolor='white', linewidth=1.2, density=False)

        ax.set_xlabel('Path length')
        ax.set_ylabel('Amount of paths')
        plt.savefig(f'{fig_path}shortest_path_dist.pdf', bbox_inches="tight")
        plt.show()
        
        plt.clf()
        img, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.set_title(f'Distribution of weighted shortest paths ({dataset_name})')
        n, bins, patches = ax.hist(splw, bins=100, color='#253957', edgecolor='white', linewidth=1.2, density=False)

        ax.set_xlabel('Path length')
        ax.set_ylabel('Amount of weighted paths')
        plt.savefig(f'{fig_path}weighted_shortest_path_dist.pdf', bbox_inches="tight")
        plt.show()
        
        ##########
        
        all_centralities = [cda, cea, ccla, cba, cka]

        centrality_correlation = np.zeros((len(all_centralities), len(all_centralities)))
        for i in range(len(all_centralities)):
            for j in range(len(all_centralities)):
                centrality_correlation[i, j] = pearsonr(all_centralities[i], all_centralities[j])[0]

        plt.clf()
        plt.title(f'Centrality correlation ({dataset_name})')
        centralities = ['degree', 'eigenvector', 'closeness', 'betweenness', 'katz']
        sns.heatmap(centrality_correlation, xticklabels=centralities, yticklabels=centralities, annot=True, fmt=".2f", linewidths=.5, cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True))
        plt.xticks(rotation=90)
        plt.savefig(f'{fig_path}centralities_correlation.pdf', bbox_inches="tight")
        plt.show()
            
        ##########
            
        plt.clf()
        img, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.set_title(f'Distribution of clustering coefficient ({dataset_name})')
        n, bins, patches = ax.hist(cc, bins=100, color='#253957', edgecolor='white', linewidth=1.2)
        ax.set_xlabel('Clustering coefficient')
        ax.set_ylabel('Number of nodes')
        plt.savefig(f'{fig_path}clustering_dist.pdf', bbox_inches="tight")
        plt.show()
        
        ##########
        
        plt.clf()
        img, ax = plt.subplots(1, 1, figsize=fig_size)
        # best fit of data
        (mu, sigma) = norm.fit(degrees_of_G)

        # the histogram of the data
        n, bins, patches = ax.hist(degrees_of_G, bins=100, density=True, label='observed', color='#253957', edgecolor='white', linewidth=1.2)

        # add a 'best fit' line
        y = norm.pdf(bins, mu, sigma)
        l = ax.plot(bins, y, 'r--', linewidth=2, label='normal approximation')
        
        scipy_hat_alpha, scipy_loc, scipy_scale = powerlaw.fit(degrees_of_G)
        y = powerlaw.pdf(bins, scipy_hat_alpha, round(scipy_loc), scipy_scale)
        l = ax.plot(bins[1:], y[1:], '--', linewidth=2, label='powerlaw approximation', color='lime')

        #plot
        ax.set_xlabel('Degree of node')
        ax.set_ylabel('Probability of degree')
        plt.title(f'Histogram of degree dist ({dataset_name}): ' + r'$\mu=%.1f,\ \sigma=%.1f,\ \alpha=%.1f,\ loc=%.1f,\ scale=%.1f$' %(mu, sigma, scipy_hat_alpha, scipy_loc, scipy_scale))
        # plt.grid(True)
        ax.legend(fontsize="10")
        plt.savefig(f'{fig_path}deg_dist_approx.pdf', bbox_inches="tight")

        plt.show()
        
        ##########
        
        nn = self.nx_graph.number_of_nodes()
        i = 2
        while 1:
            kG=nx.k_core(G,k=i)
            if kG.number_of_nodes() == 0:
                break
            i += 1
        i -= 1
        kG=nx.k_core(G,k=i)
        pos = nx.spring_layout(kG,iterations=10)
        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.axis('off')
        plt.title(f'k-core decomposition, k: {i}, number of nodes: {kG.number_of_nodes()} ({dataset_name})')

        node_classes = nx.get_node_attributes(kG, "class")
        unique_node_classes = np.unique(list(node_classes.values())).astype(int)
        unique_node_classes = np.array([self.class_to_int_mapping[c] for c in unique_node_classes])
        current_node_colors = []
        for node in kG.nodes:
            current_node_colors.append(self.class_colors[self.class_to_int_mapping[node_classes[node]]])
        nx.draw_networkx_nodes(kG, pos=pos, node_color=current_node_colors, node_size=104, ax=ax)
        nx.draw_networkx_labels(kG, pos=pos, labels={node:node for node in kG.nodes}, font_size=4, font_color='w')

        nx.draw_networkx_edges(kG, pos=pos, alpha=0.35, width=1, edge_cmap=plt.cm.Greys, edge_color=list(nx.get_edge_attributes(kG, 'ibd_sum').values()), ax=ax)

        for k, v in self.class_colors.items():
            if k in unique_node_classes:
                plt.scatter([],[], c=v, label=k)

        plt.legend()

        plt.savefig(f'{fig_path}k_core_{i}.pdf', bbox_inches="tight")
        plt.show()
        
        ##########
        
        simrank_similarity = nx.simrank_similarity(self.nx_graph)
        simrank_similarity_matrix = np.empty((len(self.nx_graph), len(self.nx_graph)))
        for source in self.nx_graph.nodes:
            for target in self.nx_graph.nodes:
                simrank_similarity_matrix[source, target] = simrank_similarity[source][target]

        plt.clf()
        img, ax = plt.subplots(1, 1, figsize=(20, 20))
        plt.title(f'Simrank similarity ({dataset_name})')
        pic = sns.heatmap(simrank_similarity_matrix, square=True, linewidths=0, annot=False, cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True), ax=ax)
        
        node_classes = nx.get_node_attributes(self.nx_graph, "class")
        
        ax.set_xticks(range(len(self.nx_graph)))
        ax.set_xticklabels([f'{self.class_to_int_mapping[node_classes[node]]} ({node})' for node in self.nx_graph.nodes], fontsize=2)
        colors = [self.class_colors[self.class_to_int_mapping[node_classes[node]]] for node in self.nx_graph.nodes]
        for xtick, color in zip(ax.get_xticklabels(), colors):
            xtick.set_color(color)
            
        ax.set_yticks(range(len(self.nx_graph)))
        ax.set_yticklabels([f'{self.class_to_int_mapping[node_classes[node]]} ({node})' for node in self.nx_graph.nodes], fontsize=2)
        colors = [self.class_colors[self.class_to_int_mapping[node_classes[node]]] for node in self.nx_graph.nodes]
        for ytick, color in zip(ax.get_yticklabels(), colors):
            ytick.set_color(color)
        
        plt.xticks(rotation=90)
        plt.savefig(f'{fig_path}simrank_similarity.png', bbox_inches="tight", dpi=400)
        plt.show()
        
        ##########
        
        simrank_similarity = nx.simrank_similarity(self.nx_graph)
        simrank_similarity_matrix = np.empty((len(self.nx_graph), len(self.nx_graph)))
        for source in self.nx_graph.nodes:
            for target in self.nx_graph.nodes:
                simrank_similarity_matrix[source, target] = simrank_similarity[source][target]

        rcm = np.array(list(nx.utils.cuthill_mckee_ordering(self.nx_graph, heuristic=None)))
        plt.clf()
        img, ax = plt.subplots(1, 1, figsize=(20, 20))
        plt.title(f'Simrank similarity reordered by Cuthill-McKee algorithm ({dataset_name})')
        simrank_similarity_matrix = simrank_similarity_matrix[rcm, :]
        simrank_similarity_matrix = simrank_similarity_matrix[:, rcm]
        pic = sns.heatmap(simrank_similarity_matrix, square=True, linewidths=0, annot=False, cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True), ax=ax)
        
        node_classes = nx.get_node_attributes(self.nx_graph, "class")
        
        ax.set_xticks(range(len(rcm)))
        ax.set_xticklabels([f'{self.class_to_int_mapping[node_classes[node]]} ({node})' for node in rcm], fontsize=2)
        colors = [self.class_colors[self.class_to_int_mapping[node_classes[node]]] for node in rcm]
        for xtick, color in zip(ax.get_xticklabels(), colors):
            xtick.set_color(color)
            
        ax.set_yticks(range(len(rcm)))
        ax.set_yticklabels([f'{self.class_to_int_mapping[node_classes[node]]} ({node})' for node in rcm], fontsize=2)
        colors = [self.class_colors[self.class_to_int_mapping[node_classes[node]]] for node in rcm]
        for ytick, color in zip(ax.get_yticklabels(), colors):
            ytick.set_color(color)
        
        plt.xticks(rotation=90)
        plt.savefig(f'{fig_path}cuthill_mckee_simrank_similarity.png', bbox_inches="tight", dpi=400)
        plt.show()
        
        with open(f'{fig_path}{dataset_name}_result.json', 'w') as f:
            json.dump(features, f, cls=NpEncoder)
        
        return features
            #####################################################################################################################################################################################################

    def place_specific_node_to_the_end(self, node_list, node_id):
        nl = node_list.copy()
        curr_node = nl[node_id]
        new_node_list = nl + [curr_node]
        new_node_list.remove(curr_node)  # remove node from the beginning and leave at the end

        return new_node_list, curr_node

    def make_hashmap(self, nodes):
        hashmap = np.array([int(len(nodes)+1) for i in range(self.node_classes_sorted.shape[0])]).astype(int)
        for i, node in enumerate(nodes):
            hashmap[node] = i

        return hashmap

    def make_one_hot_encoded_features(self, all_nodes, specific_nodes, hashmap, dict_node_classes, no_mask_class_in_df=None, mask_nodes=None):
        # order of features is the same as order nodes in self.nodes
        if mask_nodes is not None:
            if no_mask_class_in_df is None:
                raise Exception('Impossible value for parameter!')
            elif no_mask_class_in_df:
                num_classes = len(self.classes)
            else:
                num_classes = len(self.classes) - 1
        else:
            num_classes = len(self.classes)
        features = np.zeros((len(all_nodes), num_classes))
        for n in all_nodes:
            if mask_nodes is not None:
                if n in mask_nodes:
                    features[hashmap[int(n)], :] = [1 / num_classes] * num_classes
                elif n in specific_nodes:
                    features[hashmap[int(n)], :] = [1 / num_classes] * num_classes
                else:
                    features[hashmap[int(n)], :] = [1 if i == dict_node_classes[n] else 0 for i in range(num_classes)]     
            else:
                if n in specific_nodes:
                    features[hashmap[int(n)], :] = [1 / num_classes] * num_classes
                else:
                    features[hashmap[int(n)], :] = [1 if i == dict_node_classes[n] else 0 for i in range(num_classes)]

        return features
    
    @staticmethod
    @njit(cache=True, parallel=True)
    def make_graph_based_features(df, hashmap, specific_nodes, num_classes, num_nodes, log_edge_weights):
        
        features_num_edges = np.zeros((num_nodes, num_classes))
        features_ibd_tmp = np.zeros((num_nodes, num_nodes, num_classes))
        features_ibd = np.zeros((num_nodes, num_classes*2))
        
        for i in range(df.shape[0]):
            row = df[i]
            if int(row[0]) in specific_nodes and int(row[1]) not in specific_nodes: # check dulicated rows in initial df
                features_num_edges[hashmap[int(row[0])], int(row[3])] += 1
                features_ibd_tmp[hashmap[int(row[0])], hashmap[int(row[1])], int(row[3])] += -np.log2(row[4] / 6600) if log_edge_weights else row[4]
            elif int(row[1]) in specific_nodes and int(row[0]) not in specific_nodes:
                features_num_edges[hashmap[int(row[1])], int(row[2])] += 1
                features_ibd_tmp[hashmap[int(row[1])], hashmap[int(row[0])], int(row[2])] += -np.log2(row[4] / 6600) if log_edge_weights else row[4]
            elif int(row[1]) in specific_nodes and int(row[0]) in specific_nodes:
                continue
            else:
                features_num_edges[hashmap[int(row[0])], int(row[3])] += 1
                features_num_edges[hashmap[int(row[1])], int(row[2])] += 1
                features_ibd_tmp[hashmap[int(row[0])], hashmap[int(row[1])], int(row[3])] += -np.log2(row[4] / 6600) if log_edge_weights else row[4]
                features_ibd_tmp[hashmap[int(row[1])], hashmap[int(row[0])], int(row[2])] += -np.log2(row[4] / 6600) if log_edge_weights else row[4]
                
        for i in prange(num_nodes): # enhance speed in future by using part of training features for test and validation features
            for j in range(num_classes):
                current_ibd_features = features_ibd_tmp[i, :, j]
                current_ibd_features = current_ibd_features[current_ibd_features != 0]
                if len(current_ibd_features) != 0:
                    features_ibd[i, j] = np.mean(current_ibd_features)
                    features_ibd[i, num_classes+j] = np.std(current_ibd_features)
                else:
                    features_ibd[i, j] = 0
                    features_ibd[i, num_classes+j] = 0
                
        return np.concatenate((features_num_edges, features_ibd), axis=1)

    def construct_node_classes(self, nodes, dict_node_classes, mask=None):
        targets = []
        for node in nodes:
            if mask is None:
                targets.append(dict_node_classes[node])
            else:
                if node in mask:
                    targets.append(-1)
                else:
                    targets.append(dict_node_classes[node])

        return targets

    @staticmethod
    @njit(cache=True)
    def drop_rows_for_training_dataset(df, keep_nodes):
        drop_rows = []
        for i in range(df.shape[0]):  # speed it up in future
            row = df[i, :]
            if int(row[0]) in keep_nodes and int(row[1]) in keep_nodes:
                continue
            else:
                drop_rows.append(i)

        return drop_rows

    @staticmethod
    @njit(cache=True)
    def construct_edges(df, hashmap):

        weighted_edges = []

        for i in range(df.shape[0]):
            row = df[i]
            weighted_edges.append([hashmap[int(row[0])], hashmap[int(row[1])], row[4]])
            weighted_edges.append([hashmap[int(row[1])], hashmap[int(row[0])], row[4]])

        return np.array(weighted_edges)

    @staticmethod
    @njit(cache=True)
    def find_connections_to_nodes(df, train_nodes, non_train_nodes):

        rows_for_adding_per_node = []

        for i in range(len(non_train_nodes)):
            tmp = []
            for j in range(df.shape[0]):
                row = df[j]
                if int(row[0]) == non_train_nodes[i] and int(row[1]) in train_nodes or int(row[1]) == non_train_nodes[i] and int(
                        row[0]) in train_nodes:
                    tmp.append(j)

            rows_for_adding_per_node.append(tmp)

        return rows_for_adding_per_node
    
    def get_mask(self, nodes, mask_nodes):
        mask = []
        for node in nodes:
            if mask_nodes is not None:
                if node in mask_nodes:
                    mask.append(False)
                else:
                    mask.append(True)
            else:
                mask.append(True)

        return np.array(mask)

    def generate_graph(self, curr_nodes, specific_node, dict_node_classes, df, log_edge_weights, feature_type, masking, no_mask_class_in_df):
        # numba.np.ufunc.parallel._is_initialized = False
        hashmap = self.make_hashmap(curr_nodes)
        if feature_type == 'one_hot':
            if masking:
                features = self.make_one_hot_encoded_features(curr_nodes, [specific_node], hashmap,
                                                              dict_node_classes, mask_nodes=self.mask_nodes, no_mask_class_in_df=no_mask_class_in_df)
            else:
                features = self.make_one_hot_encoded_features(curr_nodes, [specific_node], hashmap,
                                                              dict_node_classes)
            assert np.sum(np.array(features).sum(axis=1) == 0) == 0
        elif feature_type == 'graph_based':
            if masking:
                features = self.make_graph_based_features(df.to_numpy(), hashmap, numba.typed.List([specific_node] + self.mask_nodes), len(self.classes) if no_mask_class_in_df else len(self.classes)-1, len(curr_nodes), log_edge_weights)
                # node_mask = self.get_mask(curr_nodes, self.mask_nodes)
            else:
                features = self.make_graph_based_features(df.to_numpy(), hashmap, numba.typed.List([specific_node]), len(self.classes), len(curr_nodes), log_edge_weights)
        else:
            raise Exception('Such feature type is not known!')
        targets = self.construct_node_classes(curr_nodes, dict_node_classes, self.mask_nodes)
        weighted_edges = self.construct_edges(df.to_numpy(), hashmap)

        # sort edges
        node_mask = self.get_mask(curr_nodes, self.mask_nodes)
        sort_idx = np.lexsort((weighted_edges[:, 1], weighted_edges[:, 0]))
        weighted_edges = weighted_edges[sort_idx]

        # checking
        if feature_type == 'one_hot':
            if not np.all(features[~node_mask] == 1 / len(self.classes)):
                raise Exception('Not uniform distributions encountered for masked nodes!')
            if not np.all(features[:-1][node_mask[:-1]] != 1 / len(self.classes)):
                raise Exception('Uniform distributions encountered not for masked nodes!')
            assert np.all(features[-1] == (1 / len(self.classes)))
            
        graph = Data.from_dict(
            {'y': torch.tensor(targets, dtype=torch.long), 'x': torch.tensor(features),
             'weight': -torch.log2(torch.tensor(weighted_edges[:, 2]) / 6600) if log_edge_weights else torch.tensor(weighted_edges[:, 2]), # try 1) log(IBD/8 * e) 2) 1 / T
             'edge_index': torch.tensor(weighted_edges[:, :2].T, dtype=torch.long),
             'mask': torch.tensor(node_mask)})

        if not masking and feature_type == 'one_hot':
            # mask for correcting GNN predictions with label propagation
            correct_and_smooth_mask = torch.tensor([True] * (len(targets)-1) + [False]).bool()
            graph.correct_and_smooth_mask = correct_and_smooth_mask
        graph.num_classes = len(self.classes) - 1 if (masking and not no_mask_class_in_df) else len(self.classes)

        return graph

    def make_train_valid_test_datasets_with_numba(self, feature_type, model_type, train_dataset_type, test_dataset_type, log_edge_weights=False, skip_train_val=False, masking=False, no_mask_class_in_df=True):

        self.array_of_graphs_for_training = []
        self.array_of_graphs_for_testing = []
        self.array_of_graphs_for_validation = []
        
        assert list(self.df.columns)[:5] == ['node_id1', 'node_id2', 'label_id1', 'label_id2', 'ibd_sum']

        if feature_type == 'one_hot' and model_type == 'homogeneous':
            if train_dataset_type == 'multiple' and test_dataset_type == 'multiple':
                dict_node_classes = self.node_classes_to_dict()
                df_for_training = self.df.copy()
                if masking:
                    drop_rows = self.drop_rows_for_training_dataset(self.df.to_numpy(), numba.typed.List(self.train_nodes + self.mask_nodes))
                else:
                    drop_rows = self.drop_rows_for_training_dataset(self.df.to_numpy(), numba.typed.List(self.train_nodes))
                df_for_training = df_for_training.drop(drop_rows)

                # make training samples
                if not skip_train_val:
                    for k in tqdm(range(len(self.train_nodes)), desc='Make train samples', disable=self.disable_printing):
                        if masking:
                            curr_train_nodes, specific_node = self.place_specific_node_to_the_end(self.train_nodes + self.mask_nodes, k)
                        else:
                            curr_train_nodes, specific_node = self.place_specific_node_to_the_end(self.train_nodes, k)

                        graph = self.generate_graph(curr_train_nodes, specific_node, dict_node_classes, df_for_training, log_edge_weights, feature_type, masking=masking, no_mask_class_in_df=no_mask_class_in_df)
                        
                        assert graph.x.shape[0] == len(curr_train_nodes)

                        self.array_of_graphs_for_training.append(graph)

                # make validation samples
                if not skip_train_val:
                    if masking:
                        rows_for_adding_per_node = self.find_connections_to_nodes(self.df.to_numpy(),
                                                                                       np.array(self.train_nodes + self.mask_nodes),
                                                                                       np.array(self.valid_nodes))
                    else:
                        rows_for_adding_per_node = self.find_connections_to_nodes(self.df.to_numpy(),
                                                                                       np.array(self.train_nodes),
                                                                                       np.array(self.valid_nodes))
                    for k in tqdm(range(len(self.valid_nodes)), desc='Make valid samples', disable=self.disable_printing):
                        rows_for_adding = rows_for_adding_per_node[k]
                        df_for_validation = pd.concat([df_for_training, self.df.iloc[rows_for_adding]], axis=0)

                        if df_for_validation.shape[0] == df_for_training.shape[0]:
                            if not self.disable_printing:
                                print('Isolated val node found! Restart with different seed or this node will be ignored.')
                            continue

                        specific_node = self.valid_nodes[k]
                        if masking:
                            current_valid_nodes = self.train_nodes + self.mask_nodes + [specific_node]
                        else:
                            current_valid_nodes = self.train_nodes + [specific_node]

                        graph = self.generate_graph(current_valid_nodes, specific_node, dict_node_classes, df_for_validation, log_edge_weights, feature_type, masking=masking, no_mask_class_in_df=no_mask_class_in_df)
                        
                        assert graph.x.shape[0] == len(current_valid_nodes)

                        self.array_of_graphs_for_validation.append(graph)

                # make testing samples
                if masking:
                    rows_for_adding_per_node = self.find_connections_to_nodes(self.df.to_numpy(),
                                                                               np.array(self.train_nodes + self.mask_nodes),
                                                                               np.array(self.test_nodes))
                else:
                    rows_for_adding_per_node = self.find_connections_to_nodes(self.df.to_numpy(),
                                                                                   np.array(self.train_nodes),
                                                                                   np.array(self.test_nodes))
                for k in tqdm(range(len(self.test_nodes)), desc='Make test samples', disable=self.disable_printing):
                    rows_for_adding = rows_for_adding_per_node[k]
                    df_for_testing = pd.concat([df_for_training, self.df.iloc[rows_for_adding]], axis=0)

                    if df_for_testing.shape[0] == df_for_training.shape[0]:
                        if not self.disable_printing:
                            print('Isolated test node found! Restart with different seed or this node will be ignored.')
                        continue

                    specific_node = self.test_nodes[k]
                    if masking:
                        current_test_nodes = self.train_nodes + self.mask_nodes + [specific_node]
                    else:
                        current_test_nodes = self.train_nodes + [specific_node]

                    graph = self.generate_graph(current_test_nodes, specific_node, dict_node_classes, df_for_testing, log_edge_weights, feature_type, masking=masking, no_mask_class_in_df=no_mask_class_in_df)
                    
                    assert graph.x.shape[0] == len(current_test_nodes)

                    self.array_of_graphs_for_testing.append(graph)

        elif feature_type == 'graph_based' and model_type == 'homogeneous':
            if train_dataset_type == 'one' and test_dataset_type == 'multiple':
                dict_node_classes = self.node_classes_to_dict()
                df_for_training = self.df.copy()
                if masking:
                    drop_rows = self.drop_rows_for_training_dataset(self.df.to_numpy(), numba.typed.List(self.train_nodes + self.mask_nodes))
                else:
                    drop_rows = self.drop_rows_for_training_dataset(self.df.to_numpy(), numba.typed.List(self.train_nodes))
                df_for_training = df_for_training.drop(drop_rows)

                # make training samples
                if not skip_train_val:
                    for k in tqdm(range(1), desc='Make train samples', disable=self.disable_printing):

                        if masking:
                            current_train_nodes = self.train_nodes + self.mask_nodes
                        else:
                            current_train_nodes = self.train_nodes

                        graph = self.generate_graph(current_train_nodes, -1, dict_node_classes, df_for_training, log_edge_weights, feature_type, masking=masking, no_mask_class_in_df=no_mask_class_in_df)
                        
                        assert graph.x.shape[0] == len(current_train_nodes)

                        self.array_of_graphs_for_training.append(graph)

                # make validation samples
                if not skip_train_val:
                    # print('VALID NODES', sum(self.valid_nodes), len(self.valid_nodes))
                    if masking:
                        rows_for_adding_per_node = self.find_connections_to_nodes(self.df.to_numpy(),
                                                                                           np.array(self.train_nodes + self.mask_nodes),
                                                                                           np.array(self.valid_nodes))
                    else:
                        rows_for_adding_per_node = self.find_connections_to_nodes(self.df.to_numpy(),
                                                                                           np.array(self.train_nodes),
                                                                                           np.array(self.valid_nodes))
                    for k in tqdm(range(len(self.valid_nodes)), desc='Make valid samples', disable=self.disable_printing):
                        rows_for_adding = rows_for_adding_per_node[k]
                        df_for_validation = pd.concat([df_for_training, self.df.iloc[rows_for_adding]], axis=0)

                        if df_for_validation.shape[0] == df_for_training.shape[0]:
                            if not self.disable_printing:
                                print('Isolated val node found! Restart with different seed or this node will be ignored.')
                            continue

                        specific_node = self.valid_nodes[k]
                        if masking:
                            current_valid_nodes = self.train_nodes + self.mask_nodes + [specific_node] # important to place specific_node in the end
                        else:
                            current_valid_nodes = self.train_nodes + [specific_node] # important to place specific_node in the end

                        graph = self.generate_graph(current_valid_nodes, specific_node, dict_node_classes, df_for_validation, log_edge_weights, feature_type, masking=masking, no_mask_class_in_df=no_mask_class_in_df)
                        
                        assert graph.x.shape[0] == len(current_valid_nodes)
                        assert torch.all(self.array_of_graphs_for_training[0].x == graph.x[:-1, :])
                        assert torch.all(self.array_of_graphs_for_training[0].y == graph.y[:-1])

                        self.array_of_graphs_for_validation.append(graph)

                # make testing samples
                if masking:
                    rows_for_adding_per_node = self.find_connections_to_nodes(self.df.to_numpy(),
                                                                                       np.array(self.train_nodes + self.mask_nodes),
                                                                                       np.array(self.test_nodes))
                else:
                    rows_for_adding_per_node = self.find_connections_to_nodes(self.df.to_numpy(),
                                                                                       np.array(self.train_nodes),
                                                                                       np.array(self.test_nodes))
                for k in tqdm(range(len(self.test_nodes)), desc='Make test samples', disable=self.disable_printing):
                    rows_for_adding = rows_for_adding_per_node[k]
                    df_for_testing = pd.concat([df_for_training, self.df.iloc[rows_for_adding]], axis=0)

                    if df_for_testing.shape[0] == df_for_training.shape[0]:
                        if not self.disable_printing:
                            print('Isolated test node found! Restart with different seed or this node will be ignored.')
                        continue

                    specific_node = self.test_nodes[k]
                    if masking:
                        current_test_nodes = self.train_nodes + self.mask_nodes + [specific_node] # important to place specific_node in the end
                    else:
                        current_test_nodes = self.train_nodes + [specific_node] # important to place specific_node in the end

                    graph = self.generate_graph(current_test_nodes, specific_node, dict_node_classes, df_for_testing, log_edge_weights, feature_type, masking=masking, no_mask_class_in_df=no_mask_class_in_df)
                    
                    assert graph.x.shape[0] == len(current_test_nodes)
                    assert torch.all(self.array_of_graphs_for_training[0].x == graph.x[:-1, :])
                    assert torch.all(self.array_of_graphs_for_training[0].y == graph.y[:-1])

                    self.array_of_graphs_for_testing.append(graph)

        else:
            raise Exception('No such method for graph generation')

    def compute_simulation_params(self):
        self.edge_probs = np.zeros((len(self.classes), len(self.classes)))
        self.mean_weight = np.zeros((len(self.classes), len(self.classes)))
        for i in range(len(self.classes)):
            for j in range(len(self.classes)):
                if i == j:
                    real_connections_df = self.df[(self.df.label_id1 == i) & (self.df.label_id2 == j)]
                else:
                    real_connections_df = self.df[
                        ((self.df.label_id1 == i) & (self.df.label_id2 == j)) | (
                                    (self.df.label_id1 == j) & (self.df.label_id2 == i))]
                real_connections = real_connections_df.shape[0]
                num_nodes = len(
                    pd.concat([real_connections_df['node_id1'], real_connections_df['node_id2']], axis=0).unique())

                self.mean_weight[i, j] = real_connections_df['ibd_sum'].to_numpy().mean() - self.offset
                if np.isnan(self.mean_weight[i, j]):
                    self.mean_weight[i, j] = -self.offset #################### can be improved

                if i == j:
                    all_possible_connections = num_nodes * (num_nodes - 1) / 2
                else:
                    n = pd.concat([self.df['node_id1'], self.df['node_id2']], axis=0)
                    l = pd.concat([self.df['label_id1'], self.df['label_id2']], axis=0)
                    df_new = pd.concat([n, l], axis=1)
                    df_new = df_new.drop_duplicates()
                    num_nodes_class_1 = len(df_new.iloc[:, 1][df_new.iloc[:, 1] == i].to_numpy())
                    num_nodes_class_2 = len(df_new.iloc[:, 1][df_new.iloc[:, 1] == j].to_numpy())
                    all_possible_connections = num_nodes_class_1 * num_nodes_class_2
                if real_connections == 0:
                    self.edge_probs[i, j] = np.nan
                else:
                    self.edge_probs[i, j] = real_connections / all_possible_connections
                
                
    def plot_simulated_probs(self, save_path=None, dataset_name=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(self.edge_probs, xticklabels=self.classes, yticklabels=self.classes, annot=True, fmt='.4f', cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True), ax=ax)
        ax.set_title(f'Edge probabilities ({dataset_name})', loc='center') # fontweight='bold'
        for i, tick_label in enumerate(ax.axes.get_yticklabels()):
            # tick_label.set_color("#008668")
            tick_label.set_fontsize("10")
        for i, tick_label in enumerate(ax.axes.get_xticklabels()):
            # tick_label.set_color("#008668")
            tick_label.set_fontsize("10")
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


    def plot_simulated_weights(self, save_path=None, dataset_name=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(self.mean_weight, xticklabels=self.classes, yticklabels=self.classes, annot=True, fmt='.4f', cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True), ax=ax)
        ax.set_title(f'Mean edge weights ({dataset_name})', loc='center') # fontweight='bold'
        for i, tick_label in enumerate(ax.axes.get_yticklabels()):
            # tick_label.set_color("#008668")
            tick_label.set_fontsize("10")
        for i, tick_label in enumerate(ax.axes.get_xticklabels()):
            # tick_label.set_color("#008668")
            tick_label.set_fontsize("10")
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    

    def generate_matrices(self, population_sizes):
        return generate_matrices_fn(population_sizes, self.offset, self.edge_probs, self.mean_weight, self.rng)
        
    def simulate_graph(self, means, counts, pop_index, path):
        simulate_graph_fn(self.classes, means, counts, pop_index, path)        
        # remove isolated nodes
        # G.remove_nodes_from(list(nx.isolates(G)))
        
    def get_stats(self):
        pass
    
    def plot_edge_weight_distribution(self, fig_size, save_path=None, custom_class_names=None, fontsize=8):
        if custom_class_names is not None:
            classes = custom_class_names
        else:
            classes = self.classes
        img, axes = plt.subplots(len(classes), len(classes), figsize=fig_size)
        for i in range(len(classes)):
            for j in range(len(classes)):
                weights = self.df.ibd_sum[((self.df.label_id1 == i) & (self.df.label_id2 == j)) | ((self.df.label_id1 == j) & (self.df.label_id2 == i))].to_numpy()
                if len(weights) == 0:
                    axes[i][j].set_title(f'{classes[i]} x {classes[j]}', fontsize=fontsize)
                    continue
                else:
                    num_bins = int(2 * len(weights) ** (1/3))
                    final_num_bins = num_bins if num_bins > 10 else 10
                    counts, bins, bars = axes[i][j].hist(weights, bins=final_num_bins, color='#69b3a2', edgecolor='white', linewidth=1.2, density=True)
                    axes[i][j].set_xlabel('edge weight')
                    axes[i][j].set_ylabel('probability')
                    axes[i][j].set_title(f'{classes[i]} x {classes[j]}', fontsize=fontsize)
                    
                    points = np.linspace(np.min(weights), np.max(weights), final_num_bins)
                    str_lables_start = r'$\frac{1}{\lambda}$'
                    axes[i][j].plot(bins[:-1] + (bins[1] - bins[0]) / 2, expon.pdf(points, loc=8.0, scale=np.mean(weights)), label=f'simulation, {str_lables_start}={np.round(np.mean(weights), 1)}', linestyle='--', marker='o', color='b')
                    axes[i][j].legend()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        


class Heuristics:
    def __init__(self, data: DataProcessor):
        self.data = data

    def compute_metrics(self, y_true, y_pred):

        f1_macro_score = f1_score(y_true, y_pred, average='macro')
        f1_weighted_score = f1_score(y_true, y_pred, average='weighted')
        recall_macro_score = recall_score(y_true, y_pred, average='macro')
        recall_weighted_score = recall_score(y_true, y_pred, average='weighted')
        precision_macro_score = precision_score(y_true, y_pred, average='macro')
        precision_weighted_score = precision_score(y_true, y_pred, average='weighted')
        acc = accuracy_score(y_true, y_pred)

        f1_macro_score_per_class = dict()
        for i in range(len(self.data.classes)):
            if self.data.classes[i] != 'masked':
                score_per_class = f1_score(y_true, y_pred, average='macro', labels=[i])
                f1_macro_score_per_class[self.data.classes[i]] = score_per_class

        return f1_macro_score, f1_weighted_score, recall_macro_score, recall_weighted_score, precision_macro_score, precision_weighted_score, acc, f1_macro_score_per_class

    def max_number_of_edges_per_class(self):
        y_true, y_preds = [], []
        isolated_test_nodes = 0
        for test_node in self.data.test_nodes:
            edges_per_class = {i:0 for i in range(len(self.data.classes))}
            # ibd_sum_per_class = {i:0 for i in range(len(self.data.classes))}
            G = self.data.nx_graph.subgraph(self.data.train_nodes + [test_node])
            node_classes = nx.get_node_attributes(G, "class")
            # edge_ibd_sum = nx.get_edge_attributes(G, "ibd_sum")
            test_node_neighbors = [node for node in G.neighbors(test_node)]
            if len(test_node_neighbors):
                for node in test_node_neighbors:
                    edges_per_class[node_classes[node]] += 1

                y_preds.append(max(edges_per_class, key=edges_per_class.get))
                y_true.append(node_classes[test_node])
            else:
                isolated_test_nodes += 1

        f1_macro_score, f1_weighted_score, recall_macro_score, recall_weighted_score, precision_macro_score, precision_weighted_score, acc, f1_macro_score_per_class = self.compute_metrics(y_true, y_preds)

        return {'f1_macro': f1_macro_score, 'f1_weighted': f1_weighted_score, 'precision_macro': precision_macro_score, 'precision_weighted': precision_weighted_score, 'recall_macro': recall_macro_score, 'recall_weighted': recall_weighted_score, 'accuracy':acc, 'class_scores': f1_macro_score_per_class, 'skipped_nodes': isolated_test_nodes}


    def max_number_of_edges_per_class_per_population(self):
        y_true, y_preds = [], []
        isolated_test_nodes = 0
        all_node_classes = nx.get_node_attributes(self.data.nx_graph, "class")
        class_balance = {i:0 for i in range(len(self.data.classes))}
        for node, cls in all_node_classes.items():
            class_balance[cls] += 1
        for test_node in self.data.test_nodes:
            edges_per_class = {i:0 for i in range(len(self.data.classes))}
            G = nx.subgraph(self.data.nx_graph, self.data.train_nodes + [test_node])
            node_classes = nx.get_node_attributes(G, "class")
            test_node_neighbors = [node for node in G.neighbors(test_node)]
            if len(test_node_neighbors):
                for node in test_node_neighbors:
                    edges_per_class[node_classes[node]] += 1

                for cls, count in class_balance.items():
                    edges_per_class[cls] /= count

                y_preds.append(max(edges_per_class, key=edges_per_class.get))
                y_true.append(node_classes[test_node])
            else:
                isolated_test_nodes += 1

        f1_macro_score, f1_weighted_score, recall_macro_score, recall_weighted_score, precision_macro_score, precision_weighted_score, acc, f1_macro_score_per_class = self.compute_metrics(y_true, y_preds)

        return {'f1_macro': f1_macro_score, 'f1_weighted': f1_weighted_score, 'precision_macro': precision_macro_score, 'precision_weighted': precision_weighted_score, 'recall_macro': recall_macro_score, 'recall_weighted': recall_weighted_score, 'accuracy':acc, 'class_scores': f1_macro_score_per_class, 'skipped_nodes': isolated_test_nodes}
            
    def max_number_of_segments_per_class(self):
        y_true, y_preds = [], []
        isolated_test_nodes = 0
        for test_node in self.data.test_nodes:
            segments_per_class = {i:0 for i in range(len(self.data.classes))}
            # ibd_sum_per_class = {i:0 for i in range(len(self.data.classes))}
            G = self.data.nx_graph.subgraph(self.data.train_nodes + [test_node])
            node_classes = nx.get_node_attributes(G, "class")
            num_segments = nx.get_edge_attributes(G, "ibd_n")
            # edge_ibd_sum = nx.get_edge_attributes(G, "ibd_sum")
            test_node_neighbors = [node for node in G.neighbors(test_node)]
            if len(test_node_neighbors):
                for node in test_node_neighbors:
                    if (node, test_node) in num_segments.keys():
                        segments_per_class[node_classes[node]] += num_segments[(node, test_node)]
                    elif (test_node, node) in num_segments.keys():
                        segments_per_class[node_classes[node]] += num_segments[(test_node, node)]
                    else:
                        raise Exception('No edge in subgraph!')
                y_preds.append(max(segments_per_class, key=segments_per_class.get))
                y_true.append(node_classes[test_node])
            else:
                isolated_test_nodes += 1

        f1_macro_score, f1_weighted_score, recall_macro_score, recall_weighted_score, precision_macro_score, precision_weighted_score, acc, f1_macro_score_per_class = self.compute_metrics(y_true, y_preds)

        return {'f1_macro': f1_macro_score, 'f1_weighted': f1_weighted_score, 'precision_macro': precision_macro_score, 'precision_weighted': precision_weighted_score, 'recall_macro': recall_macro_score, 'recall_weighted': recall_weighted_score, 'accuracy':acc, 'class_scores': f1_macro_score_per_class, 'skipped_nodes': isolated_test_nodes}


    def longest_ibd(self):
        y_true, y_preds = [], []
        isolated_test_nodes = 0
        for test_node in self.data.test_nodes:
            ibd_max_per_class = {i:0 for i in range(len(self.data.classes))}
            G = self.data.nx_graph.subgraph(self.data.train_nodes + [test_node])
            node_classes = nx.get_node_attributes(G, "class")
            edge_ibd_sum = nx.get_edge_attributes(G, "ibd_sum")
            test_node_neighbors = [node for node in G.neighbors(test_node)]
            if len(test_node_neighbors):
                for node in test_node_neighbors:
                    if (node, test_node) in edge_ibd_sum.keys():
                        if edge_ibd_sum[(node, test_node)] > ibd_max_per_class[node_classes[node]]:
                            ibd_max_per_class[node_classes[node]] = edge_ibd_sum[(node, test_node)]
                    elif (test_node, node) in edge_ibd_sum.keys():
                        if edge_ibd_sum[(test_node, node)] > ibd_max_per_class[node_classes[node]]:
                            ibd_max_per_class[node_classes[node]] = edge_ibd_sum[(test_node, node)]
                    else:
                        raise Exception('No edge in subgraph!')

                y_preds.append(max(ibd_max_per_class, key=ibd_max_per_class.get))
                y_true.append(node_classes[test_node])
            else:
                isolated_test_nodes += 1

        f1_macro_score, f1_weighted_score, recall_macro_score, recall_weighted_score, precision_macro_score, precision_weighted_score, acc, f1_macro_score_per_class = self.compute_metrics(y_true, y_preds)

        return {'f1_macro': f1_macro_score, 'f1_weighted': f1_weighted_score, 'precision_macro': precision_macro_score, 'precision_weighted': precision_weighted_score, 'recall_macro': recall_macro_score, 'recall_weighted': recall_weighted_score, 'accuracy':acc, 'class_scores': f1_macro_score_per_class, 'skipped_nodes': isolated_test_nodes}


    def max_ibd_sum_per_class(self):
        y_true, y_preds = [], []
        isolated_test_nodes = 0
        for test_node in self.data.test_nodes:
            # edges_per_class = {i:0 for i in range(len(self.data.classes))}
            ibd_sum_per_class = {i:0 for i in range(len(self.data.classes))}
            G = self.data.nx_graph.subgraph(self.data.train_nodes + [test_node])
            node_classes = nx.get_node_attributes(G, "class")
            edge_ibd_sum = nx.get_edge_attributes(G, "ibd_sum")
            test_node_neighbors = [node for node in G.neighbors(test_node)]
            if len(test_node_neighbors):
                for node in test_node_neighbors:
                    if (node, test_node) in edge_ibd_sum.keys():
                        ibd_sum_per_class[node_classes[node]] += edge_ibd_sum[(node, test_node)]
                    elif (test_node, node) in edge_ibd_sum.keys():
                        ibd_sum_per_class[node_classes[node]] += edge_ibd_sum[(test_node, node)]
                    else:
                        raise Exception('No edge in subgraph!')

                y_preds.append(max(ibd_sum_per_class, key=ibd_sum_per_class.get))
                y_true.append(node_classes[test_node])
            else:
                isolated_test_nodes += 1

        f1_macro_score, f1_weighted_score, recall_macro_score, recall_weighted_score, precision_macro_score, precision_weighted_score, acc, f1_macro_score_per_class = self.compute_metrics(y_true, y_preds)

        return {'f1_macro': f1_macro_score, 'f1_weighted': f1_weighted_score, 'precision_macro': precision_macro_score, 'precision_weighted': precision_weighted_score, 'recall_macro': recall_macro_score, 'recall_weighted': recall_weighted_score, 'accuracy':acc, 'class_scores': f1_macro_score_per_class, 'skipped_nodes': isolated_test_nodes}


    def max_ibd_sum_per_class_per_population(self):
        y_true, y_preds = [], []
        isolated_test_nodes = 0
        all_node_classes = nx.get_node_attributes(self.data.nx_graph, "class")
        class_balance = {i:0 for i in range(len(self.data.classes))}
        for node, cls in all_node_classes.items():
            class_balance[cls] += 1
        for test_node in self.data.test_nodes:
            ibd_sum_per_class = {i:0 for i in range(len(self.data.classes))}
            G = self.data.nx_graph.subgraph(self.data.train_nodes + [test_node])
            node_classes = nx.get_node_attributes(G, "class")
            edge_ibd_sum = nx.get_edge_attributes(G, "ibd_sum")
            test_node_neighbors = [node for node in G.neighbors(test_node)]
            if len(test_node_neighbors):
                for node in test_node_neighbors:
                    if (node, test_node) in edge_ibd_sum.keys():
                        ibd_sum_per_class[node_classes[node]] += edge_ibd_sum[(node, test_node)]
                    elif (test_node, node) in edge_ibd_sum.keys():
                        ibd_sum_per_class[node_classes[node]] += edge_ibd_sum[(test_node, node)]
                    else:
                        raise Exception('No edge in subgraph!')
                    
                for cls, count in class_balance.items():
                    ibd_sum_per_class[cls] /= count

                y_preds.append(max(ibd_sum_per_class, key=ibd_sum_per_class.get))
                y_true.append(node_classes[test_node])
            else:
                isolated_test_nodes += 1

        f1_macro_score, f1_weighted_score, recall_macro_score, recall_weighted_score, precision_macro_score, precision_weighted_score, acc, f1_macro_score_per_class = self.compute_metrics(y_true, y_preds)

        return {'f1_macro': f1_macro_score, 'f1_weighted': f1_weighted_score, 'precision_macro': precision_macro_score, 'precision_weighted': precision_weighted_score, 'recall_macro': recall_macro_score, 'recall_weighted': recall_weighted_score, 'accuracy':acc, 'class_scores': f1_macro_score_per_class, 'skipped_nodes': isolated_test_nodes}

                

    def run_heuristic(self, heuristic_name):
        if heuristic_name == 'MaxEdgeCount':
            return self.max_number_of_edges_per_class()
        elif heuristic_name == 'MaxEdgeCountPerClassSize':
            return self.max_number_of_edges_per_class_per_population()
        elif heuristic_name == 'MaxSegmentCount':
            return self.max_number_of_segments_per_class()
        elif heuristic_name == 'LongestIbd':
            return self.longest_ibd()
        elif heuristic_name == 'MaxIbdSum':
            return self.max_ibd_sum_per_class()
        elif heuristic_name == 'MaxIbdSumPerClassSize':
            return self.max_ibd_sum_per_class_per_population()


class NullSimulator:
    def __init__(self, num_classes, edge_probs, mean_weight):
        self.num_classes = num_classes
        self.classes = [f'P{i}' for i in range(self.num_classes)]
        self.edge_probs = edge_probs
        self.mean_weight = mean_weight
        self.offset = 6.0

    def symmetrize(self, m):
        return m + m.T - np.diag(m.diagonal())


    def generate_matrices(self, population_sizes, rng):
        '''
            main simulation function
        
        Parameters
        ----------
        population_sizes: list 
            list of population sizes
        offset: float
            we assume ibdsum pdf = lam*exp(-lam*(x-offset)) for x>offset and 0 otherwise, lam = 1/mean
        edge_probs: 2d array
            probability of an edge between classes
        mean_weight: 2d array
            mean weight of an existing edge between classes (corrected by offset)
        rng: random number generator
        
        Returns
        -------
        counts: 
            
        sums: 
            
        pop_index: 1d np array
            population index of every node
            
        '''
        p = self.edge_probs
        teta = self.mean_weight
        # print(teta)
        pop_index = []
        n_pops = len(population_sizes)
        for i in range(n_pops):
            pop_index += [i] * population_sizes[i]

        pop_index = np.array(pop_index)
        #print(f"{n_pops=}")
        blocks_sums = [[np.zeros(shape=(population_sizes[i], population_sizes[j])) for j in range(n_pops)] for i in
                    range(n_pops)]
        blocks_counts = [[np.zeros(shape=(population_sizes[i], population_sizes[j])) for j in range(n_pops)] for i
                        in range(n_pops)]

        #print(np.array(blocks_sums).shape)

        for pop_i in range(n_pops):
            for pop_j in range(pop_i + 1):
                if p[pop_i, pop_j] == 0:
                    continue
                #print(f"{pop_i=} {pop_j=}")
                pop_cross = population_sizes[pop_i] * population_sizes[pop_j]
                #TODO switch to rng.binomial or something
                bern_samples =  rng.binomial(1, p[pop_i, pop_j], pop_cross) #bernoulli.rvs(p[pop_i, pop_j], size=pop_cross)
                total_segments = np.sum(bern_samples)
                #print(f"{total_segments=}")
                exponential_samples = rng.exponential(teta[pop_i, pop_j], size=total_segments) + self.offset
                #position = 0
                exponential_totals_samples = np.zeros(pop_cross, dtype=np.float64)
                #mean_totals_samples = np.zeros(pop_cross, dtype=np.float64)
                exponential_totals_samples[bern_samples == 1] = exponential_samples

                bern_samples = np.reshape(bern_samples, newshape=(population_sizes[pop_i], population_sizes[pop_j]))
                exponential_totals_samples = np.reshape(exponential_totals_samples,
                                                        newshape=(population_sizes[pop_i], population_sizes[pop_j]))
                if (pop_i == pop_j):
                    bern_samples = np.tril(bern_samples, -1)
                    exponential_totals_samples = np.tril(exponential_totals_samples, -1)
                blocks_counts[pop_i][pop_j] = bern_samples
                blocks_sums[pop_i][pop_j] = exponential_totals_samples
        
        
        full_blocks_counts = np.block(blocks_counts)
        full_blocks_sums = np.block(blocks_sums)
        # print(np.unique(full_blocks_sums))
        # print(np.unique(np.nan_to_num(symmetrize(full_blocks_sums))))
        return np.nan_to_num(symmetrize(full_blocks_counts)), np.nan_to_num(symmetrize(full_blocks_sums)), pop_index


    def simulate_graph(self, means, counts, pop_index, path):
        '''
            store simulated dataframe
        
        Parameters
        ----------
        classes: list of str
            names of populations
        means: 2d np array
            0: no link between i-th and j-th individuals
        counts: 2d np array
            ibd sum between i-th and j-th individuals
        pop_index: 1d np array
            population index of every node
        path: string
            csv file to store dataframe
        '''
        indiv = list(range(counts.shape[0]))
        with open(path, 'w', encoding="utf-8") as f:
            f.write('node_id1,node_id2,label_id1,label_id2,ibd_sum,ibd_n\n')
            for i in range(counts.shape[0]):
                for j in range(i):
                    if (counts[i][j]):
                        name_i = self.classes[pop_index[i]] if "," not in self.classes[pop_index[i]] else '\"' + self.classes[pop_index[i]] + '\"'
                        name_j = self.classes[pop_index[j]] if "," not in self.classes[pop_index[j]] else '\"' + self.classes[pop_index[j]] + '\"'
                        #f.write(f'node_{i},node_{j},{name_i},{name_j},{means[i][j]},{counts[i][j]}\n')
                        f.write(f'node_{i},node_{j},{name_i},{name_j},{means[i][j]},1\n')

    # def symmetrize(self, m):
    #     return m + m.T - np.diag(m.diagonal())

    # def generate_matrices(self, population_sizes):
    #     p = self.edge_probs
    #     teta = self.mean_weight
    #     pop_index = []
    #     n_pops = len(population_sizes)
    #     for i in range(n_pops):
    #         pop_index += [i] * population_sizes[i]

    #     pop_index = np.array(pop_index)
    #     print(f"{n_pops=}")
    #     blocks_sums = [[np.zeros(shape=(population_sizes[i], population_sizes[j])) for i in range(n_pops)] for j in
    #                    range(n_pops)]
    #     blocks_counts = [[np.zeros(shape=(population_sizes[i], population_sizes[j])) for i in range(n_pops)] for j
    #                      in range(n_pops)]

    #     print(np.array(blocks_sums).shape)

    #     for pop_i in range(n_pops):
    #         for pop_j in range(pop_i + 1):
    #             if p[pop_i, pop_j] == 0:
    #                 continue
    #             # print(f"{pop_i=} {pop_j=}")
    #             pop_cross = population_sizes[pop_i] * population_sizes[pop_j]
    #             bern_samples = bernoulli.rvs(p[pop_i, pop_j], size=pop_cross)
    #             total_segments = np.sum(bern_samples)
    #             # print(f"{total_segments=}")
    #             exponential_samples = np.random.exponential(teta[pop_i, pop_j], size=total_segments) + self.offset
    #             position = 0
    #             exponential_totals_samples = np.zeros(pop_cross, dtype=np.float64)
    #             mean_totals_samples = np.zeros(pop_cross, dtype=np.float64)
    #             exponential_totals_samples[bern_samples == 1] = exponential_samples

    #             bern_samples = np.reshape(bern_samples, newshape=(population_sizes[pop_i], population_sizes[pop_j]))
    #             exponential_totals_samples = np.reshape(exponential_totals_samples,
    #                                                     newshape=(population_sizes[pop_i], population_sizes[pop_j]))
    #             if (pop_i == pop_j):
    #                 bern_samples = np.tril(bern_samples, -1)
    #                 exponential_totals_samples = np.tril(exponential_totals_samples, -1)
    #             blocks_counts[pop_i][pop_j] = bern_samples
    #             blocks_sums[pop_i][pop_j] = exponential_totals_samples
    #     return np.nan_to_num(self.symmetrize(np.block(blocks_counts))), np.nan_to_num(self.symmetrize(np.block(blocks_sums))), pop_index

    # def simulate_graph(self, means, counts, pop_index, path):
    #     indiv = list(range(counts.shape[0]))
    #     with open(path, 'w', encoding="utf-8") as f:
    #         f.write('node_id1,node_id2,label_id1,label_id2,ibd_sum\n')
    #         for i in range(counts.shape[0]):
    #             for j in range(i):
    #                 if (means[i][j]):
    #                     name_i = self.classes[pop_index[i]] if "," not in self.classes[pop_index[i]] else '\"' + self.classes[pop_index[i]] + '\"'
    #                     name_j = self.classes[pop_index[j]] if "," not in self.classes[pop_index[j]] else '\"' + self.classes[pop_index[j]] + '\"'
    #                     f.write(f'node_{i},node_{j},{name_i},{name_j},{counts[i][j]}\n')



class Trainer:
    def __init__(self, data: DataProcessor, model_cls, lr, wd, loss_fn, batch_size, log_dir, patience, num_epochs, feature_type, train_iterations_per_sample, evaluation_steps, weight=None, cuda_device_specified: int = None, masking=False, disable_printing=True, seed=42, save_model_in_ram=False, correct_and_smooth=False, no_mask_class_in_df=True, remove_saved_model_after_testing=False, plot_cm=False, use_class_balance_weight=False):
        self.data = data
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if cuda_device_specified is None else torch.device(f'cuda:{cuda_device_specified}' if torch.cuda.is_available() else 'cpu')
        self.gpuidx = cuda_device_specified
        self.model_cls = model_cls
        self.learning_rate = lr
        self.weight_decay = wd
        self.loss_fn = loss_fn
        if masking and not no_mask_class_in_df:
            self.weight = torch.tensor([1. for i in range(len(self.data.classes)-1)]).to(self.device) if weight is None else weight
        else:
            self.weight = torch.tensor([1. for i in range(len(self.data.classes))]).to(self.device) if weight is None else weight
        if use_class_balance_weight and no_mask_class_in_df:
            print('Using loss weights according to class balance')
            all_classes = self.data.node_classes_sorted['class_id'].to_numpy()
            count_dict = Counter(all_classes)
            count_dict = dict(sorted(count_dict.items()))
            self.weight = torch.tensor(list(count_dict.values())).to(self.device)
            self.weight = torch.max(self.weight) / self.weight
        self.batch_size = batch_size # not used by far
        self.log_dir = log_dir
        self.patience = patience
        self.num_epochs = num_epochs
        self.max_f1_score_macro = 0
        self.patience_counter = 0
        self.feature_type = feature_type
        self.train_iterations_per_sample = train_iterations_per_sample
        self.evaluation_steps = evaluation_steps
        self.masking = masking
        self.disable_printing = disable_printing
        self.seed = seed
        self.save_model_in_ram = save_model_in_ram
        self.correct_and_smooth = correct_and_smooth
        self.remove_saved_model_after_testing = remove_saved_model_after_testing
        self.plot_cm = plot_cm
            
        self.post = CorrectAndSmooth(num_correction_layers=2, correction_alpha=0.9,
                        num_smoothing_layers=1, smoothing_alpha=0.0001,
                        autoscale=True) if self.correct_and_smooth else None

    def compute_metrics_cross_entropy(self, graphs, mask=False, phase=None):
        y_true = []
        y_pred = []

        if self.feature_type == 'one_hot':
            for i in tqdm(range(len(graphs)), desc='Compute metrics', disable=self.disable_printing):
                if self.correct_and_smooth:
                    y_soft = F.softmax(self.model(graphs[i].to(self.device)), dim=-1).detach()
                    y_soft = self.post.correct(y_soft, graphs[i].y[graphs[i].correct_and_smooth_mask], graphs[i].correct_and_smooth_mask, graphs[i].edge_index, graphs[i].weight.float())
                    y_soft = self.post.smooth(y_soft, graphs[i].y[graphs[i].correct_and_smooth_mask], graphs[i].correct_and_smooth_mask, graphs[i].edge_index, graphs[i].weight.float())
                    p = y_soft[-1].cpu().detach().numpy()
                    y_pred.append(np.argmax(p))
                    y_true.append(int(graphs[i].y[-1].cpu().detach().numpy()))
                    graphs[i].to('cpu')
                else:
                    p = F.softmax(self.model(graphs[i].to(self.device))[-1], dim=0).cpu().detach().numpy()
                    y_pred.append(np.argmax(p))
                    y_true.append(int(graphs[i].y[-1].cpu().detach().numpy()))
                    graphs[i].to('cpu')
        elif self.feature_type == 'graph_based':
            if not mask:
                for i in tqdm(range(len(graphs)), desc='Compute metrics', disable=self.disable_printing):
                    if phase=='training':
                        p = F.softmax(self.model(graphs[i].to(self.device)),
                                      dim=0).cpu().detach().numpy()
                        y_pred = np.argmax(p, axis=1)
                        y_true = graphs[i].y.cpu().detach()
                    elif phase=='scoring':
                        p = F.softmax(self.model(graphs[i].to(self.device))[-1],
                                      dim=0).cpu().detach().numpy()
                        y_pred.append(np.argmax(p))
                        y_true.append(graphs[i].y[-1].cpu().detach())
                    else:
                        raise Exception('No such phase!')
                    graphs[i].to('cpu')
            else:
                for i in tqdm(range(len(graphs)), desc='Compute metrics', disable=self.disable_printing):
                    if phase=='training':
                        p = F.softmax(self.model(graphs[i].to(self.device)),
                                      dim=0)
                        p = p[graphs[i].mask].cpu().detach().numpy()
                        y_pred = np.argmax(p, axis=1)
                        y_true = graphs[i].y[graphs[i].mask].cpu().detach()
                    elif phase=='scoring':
                        p = F.softmax(self.model(graphs[i].to(self.device))[-1],
                                      dim=0).cpu().detach().numpy()
                        y_pred.append(np.argmax(p))
                        y_true.append(graphs[i].y[-1].cpu().detach().numpy().item())
                    else:
                        raise Exception('No such phase!')
                    graphs[i].to('cpu')
        else:
            raise Exception('Trainer is not implemented for such feature type name while calculating training scores!')

        # if phase=='scoring':
        #     print(len(graphs), len(y_true), len(y_pred), sum(y_true), sum(y_pred))
        #     print(y_true)
        #     print(y_pred)
        #     assert False
        return y_true, y_pred

    def evaluation(self, i, mask=False):
        self.model.eval()

        y_true, y_pred = self.compute_metrics_cross_entropy(self.data.array_of_graphs_for_validation, mask=mask, phase='scoring')

        if not self.disable_printing:
            print('Evaluation report')
            print(classification_report(y_true, y_pred))
        for i in range(len(self.data.classes)):
            if self.data.classes[i] != 'masked':
                score_per_class = f1_score(y_true, y_pred, average='macro', labels=[i])
                if not self.disable_printing:
                    print(f"f1 macro score on valid dataset for class {i} which is {self.data.classes[i]}: {score_per_class}")

        current_f1_score_macro = f1_score(y_true, y_pred, average='macro')
        if current_f1_score_macro >= self.max_f1_score_macro:
            self.patience_counter = 0
            self.max_f1_score_macro = current_f1_score_macro
            if not self.disable_printing:
                print(f'f1 macro improvement to {self.max_f1_score_macro}')
            torch.save(self.model.state_dict(), self.log_dir + '/model_best.bin')
        else:
            self.patience_counter += 1
            if not self.disable_printing:
                print(f'Metric was not improved for the {self.patience_counter}th time')

    def test(self, mask=False):
        self.model = self.model_cls(self.data.array_of_graphs_for_training[0]).to(self.device)
        self.model.load_state_dict(torch.load(self.log_dir + '/model_best.bin'))
        self.model.eval()
        y_true, y_pred = self.compute_metrics_cross_entropy(self.data.array_of_graphs_for_testing, mask=mask, phase='scoring')
        if not self.disable_printing:
            print('Test report')
            print(classification_report(y_true, y_pred))
        
        f1_macro_score = f1_score(y_true, y_pred, average='macro')
        if not self.disable_printing:
            print(f"f1 macro score on test dataset: {f1_macro_score}")
        
        f1_weighted_score = f1_score(y_true, y_pred, average='weighted')
        if not self.disable_printing:
            print(f"f1 weighted score on test dataset: {f1_weighted_score}")

        recall_macro_score = recall_score(y_true, y_pred, average='macro')
        if not self.disable_printing:
            print(f"recall macro score on test dataset: {recall_macro_score}")

        recall_weighted_score = recall_score(y_true, y_pred, average='weighted')
        if not self.disable_printing:
            print(f"recall weighted score on test dataset: {recall_weighted_score}")

        precision_macro_score = precision_score(y_true, y_pred, average='macro')
        if not self.disable_printing:
            print(f"recall macro score on test dataset: {precision_macro_score}")

        precision_weighted_score = precision_score(y_true, y_pred, average='weighted')
        if not self.disable_printing:
            print(f"recall weighted score on test dataset: {precision_weighted_score}")
        
        acc = accuracy_score(y_true, y_pred)
        if not self.disable_printing:
            print(f"accuracy score on test dataset: {acc}")
        
        f1_macro_score_per_class = dict()
        
        for i in range(len(self.data.classes)):
            if self.data.classes[i] != 'masked':
                score_per_class = f1_score(y_true, y_pred, average='macro', labels=[i])
                if not self.disable_printing:
                    print(f"f1 macro score on test dataset for class {i} which is {self.data.classes[i]}: {score_per_class}")
                f1_macro_score_per_class[self.data.classes[i]] = score_per_class

        cm = confusion_matrix(y_true, y_pred, normalize='true')

        if self.plot_cm:
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(cm, xticklabels=self.data.classes, yticklabels=self.data.classes, annot=True, fmt='.2f', cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True), ax=ax)
            ax.set_title(f'Confusion matrix for {self.data.dataset_name}', loc='center')
            for i, tick_label in enumerate(ax.axes.get_yticklabels()):
                # tick_label.set_color("#008668")
                tick_label.set_fontsize("10")
            for i, tick_label in enumerate(ax.axes.get_xticklabels()):
                # tick_label.set_color("#008668")
                tick_label.set_fontsize("10")
            plt.tight_layout()
            plt.savefig(self.log_dir + '/cm_on_test_data.png')
            # plt.clf()
            # fig, ax = plt.subplots(1, 1)
            # sns.heatmap(cm, annot=True, fmt=".2f", ax=ax)
            # plt.show()

        if self.device != 'cpu':
            memory_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            memory_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        else:
            memory_allocated = None
            memory_reserved = None

        if self.remove_saved_model_after_testing:
            os.remove(self.log_dir + '/model_best.bin')
            
        if not self.save_model_in_ram:
            self.model = None
            gc.collect() # Python thing
            torch.cuda.empty_cache() # PyTorch thing
        else:
            self.model = self.model.eval().cpu()

        return {'f1_macro': f1_macro_score, 'f1_weighted': f1_weighted_score, 'precision_macro': precision_macro_score, 'precision_weighted': precision_weighted_score, 'recall_macro': recall_macro_score, 'recall_weighted': recall_weighted_score, 'accuracy':acc, 'class_scores': f1_macro_score_per_class, 'skipped_nodes': len(self.data.test_nodes) - len(self.data.array_of_graphs_for_testing), 'memory_allocated_MB': memory_allocated, 'memory_reserved_MB': memory_reserved}
        

    def run(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)  # if you are using multi-GPU.
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.model = self.model_cls(self.data.array_of_graphs_for_training[0]).to(self.device) # just initialize the parameters of the model
        criterion = self.loss_fn(weight=self.weight)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.95)
        print(f'Training for data: {self.data.dataset_name}')
        self.max_f1_score_macro = 0
        self.patience_counter = 0

        if self.loss_fn == torch.nn.CrossEntropyLoss:
            if self.feature_type == 'one_hot':
                for i in tqdm(range(self.num_epochs), desc='Training epochs', disable=self.disable_printing):
                    if self.patience_counter == self.patience:
                        break
                    self.evaluation(i)

                    self.model.train()

                    selector = np.array([i for i in range(len(self.data.array_of_graphs_for_training))])
                    np.random.shuffle(selector)

                    mean_epoch_loss = []

                    pbar = tqdm(range(len(selector)), desc='Training samples', disable=self.disable_printing)
                    pbar.set_postfix({'val_best_score': self.max_f1_score_macro})
                    for j, data_curr in enumerate(pbar):
                        n = selector[j]
                        data_curr = self.data.array_of_graphs_for_training[n].to(self.device)
                        optimizer.zero_grad()
                        out = self.model(data_curr)
                        # print(data_curr.x.shape, out[-1], data_curr.y[-1])
                        loss = criterion(out[-1], data_curr.y[-1])
                        loss.backward()
                        mean_epoch_loss.append(loss.detach().cpu().numpy())
                        optimizer.step()
                        scheduler.step()
                        self.data.array_of_graphs_for_training[n].to('cpu')
                        
                    if not self.disable_printing:
                        print(f'Mean loss: {np.mean(mean_epoch_loss)}')

                    y_true, y_pred = self.compute_metrics_cross_entropy(self.data.array_of_graphs_for_training)

                    if not self.disable_printing:
                        print('Training report')
                        print(classification_report(y_true, y_pred))
                    
            elif self.feature_type == 'graph_based':
                if self.masking:
                    data_curr = self.data.array_of_graphs_for_training[0].to(self.device)
                    self.model.train()
                    for i in tqdm(range(self.train_iterations_per_sample), desc='Training iterations', disable=self.disable_printing):
                        if self.patience_counter == self.patience:
                            break
                        if i % self.evaluation_steps == 0:
                            y_true, y_pred = self.compute_metrics_cross_entropy(self.data.array_of_graphs_for_training, mask=True, phase='training')

                            if not self.disable_printing:
                                print('Training report')
                                print(classification_report(y_true, y_pred))

                            self.data.array_of_graphs_for_training[0].to('cpu')
                            self.evaluation(i, mask=True)
                            self.model.train()
                            self.data.array_of_graphs_for_training[0].to(self.device)

                        optimizer.zero_grad()
                        out = self.model(data_curr)
                        # print(self.model.fc1.weight)
                        # assert False
                        # print(data_curr.x[data_curr.mask].detach().cpu().numpy().sum())
                        # print(out[data_curr.mask].shape, len(self.data.train_nodes))
                        # assert False
                        loss = criterion(out[data_curr.mask], data_curr.y[data_curr.mask])
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                else:
                    data_curr = self.data.array_of_graphs_for_training[0].to(self.device)
                    self.model.train()
                    for i in tqdm(range(self.train_iterations_per_sample), desc='Training iterations', disable=self.disable_printing):
                        if self.patience_counter == self.patience:
                            break
                        if i % self.evaluation_steps == 0:
                            y_true, y_pred = self.compute_metrics_cross_entropy(self.data.array_of_graphs_for_training, phase='training')

                            if not self.disable_printing:
                                print('Training report')
                                print(classification_report(y_true, y_pred))

                            self.data.array_of_graphs_for_training[0].to('cpu')
                            self.evaluation(i)
                            self.model.train()
                            self.data.array_of_graphs_for_training[0].to(self.device)

                        optimizer.zero_grad()
                        out = self.model(data_curr)
                        loss = criterion(out, data_curr.y)
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

            else:
                raise Exception('Trainer is not implemented for such feature type name!')

            return self.test(mask=self.masking)
        


def independent_test(model_path, model_cls, df, vertex_id, gpu_id, test_type, mask_nodes=None):

    dp = DataProcessor(df.copy(), is_path_object=True)
    dp.classes.remove('unknown')
    unique_nodes = list(pd.concat([df['node_id1'], df['node_id2']], axis=0).unique())
    if f'node_{vertex_id}' in unique_nodes:
        unique_nodes.remove(f'node_{vertex_id}')
    else:
        raise Exception('Test node not in DataFrame!')
    train_split = np.array(list(map(lambda x: int(x[5:]), unique_nodes)))
    valid_split = np.array([vertex_id])
    test_split = np.array([vertex_id])
    if mask_nodes is not None:
        mask_data = np.array(mask_nodes)
        train_split = np.array(list(filter(lambda node: node not in mask_data, train_split)))
        valid_split = np.array(list(filter(lambda node: node not in mask_data, valid_split)))
        test_split = np.array(list(filter(lambda node: node not in mask_data, test_split)))
    
    dp.load_train_valid_test_nodes(train_split, valid_split, test_split, 'numpy', mask_path=mask_nodes)
    
    # what if some vertex_id node can't be attached to train graph? Add handling of this behavior
    
    if test_type == 'one_hot':        
        dp.make_train_valid_test_datasets_with_numba('one_hot', 'homogeneous', 'multiple', 'multiple', 'debug_debug', skip_train_val=True)
    elif test_type == 'graph_based':
        dp.make_train_valid_test_datasets_with_numba('graph_based', 'homogeneous', 'one', 'multiple', 'debug_debug', skip_train_val=True)
    elif test_type == 'graph_based_masked' and mask_nodes is not None:
        dp.make_train_valid_test_datasets_with_numba('graph_based', 'homogeneous', 'one', 'multiple', 'debug_debug', skip_train_val=True, masking=True)
    elif test_type == 'one_hot_masked' and mask_nodes is not None:
        dp.make_train_valid_test_datasets_with_numba('one_hot', 'homogeneous', 'multiple', 'multiple', 'debug_debug', skip_train_val=True, masking=True)  
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else 'cpu')
    model = model_cls(dp.array_of_graphs_for_testing[0]).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    p = F.softmax(model(dp.array_of_graphs_for_testing[0].to(device))[-1], dim=0).cpu().detach().numpy()
    dp.array_of_graphs_for_testing[0].to('cpu')
    return dp.classes[np.argmax(p)]
    
    
   

        
class CommunityDetection:
    def __init__(self, data: DataProcessor):
        self.data = data
        
    def torch_geometric_label_propagation(self, num_layers, alpha, use_weight=True, use_masking_from_data=True):
        model = LabelPropagation(num_layers=num_layers, alpha=alpha)
        
        y_pred = []
        y_true = []
        for i in tqdm(range(len(self.data.array_of_graphs_for_testing)), desc='Label propagation'):
            graph = self.data.array_of_graphs_for_testing[i]
            # print(graph.x[-1])
            y_true.append(graph.y[-1])
            if use_masking_from_data:
                node_mask = list(graph.mask.to('cpu').detach().numpy())
                assert np.all(np.array(node_mask) == True) == True
                assert graph.y[-1] != -1
                node_mask = node_mask[:-1] + [False]
            else:
                node_mask = [True] * (len(graph.y)-1) + [False]
                
            y_pred.append(model(y=graph.y, mask = node_mask,  edge_index=graph.edge_index, edge_weight=graph.weight if use_weight==True else None).argmax(dim=-1)[-1]) # -1 is always test vertex
            
        f1_macro_score = f1_score(y_true, y_pred, average='macro')
        # print(f"f1 macro score on test dataset: {f1_macro_score}")
        
        f1_weighted_score = f1_score(y_true, y_pred, average='weighted')
        # print(f"f1 weighted score on test dataset: {f1_weighted_score}")
        
        acc = accuracy_score(y_true, y_pred)
        # print(f"accuracy score on test dataset: {acc}")

        recall_macro_score = recall_score(y_true, y_pred, average='macro')
        recall_weighted_score = recall_score(y_true, y_pred, average='weighted')
        precision_macro_score = precision_score(y_true, y_pred, average='macro')
        precision_weighted_score = precision_score(y_true, y_pred, average='weighted')
        
        f1_macro_score_per_class = dict()
        
        for i in range(len(self.data.classes)):
            score_per_class = f1_score(y_true, y_pred, average='macro', labels=[i])
            # print(f"f1 macro score on test dataset for class {i} which is {self.data.classes[i]}: {score_per_class}")
            f1_macro_score_per_class[self.data.classes[i]] = score_per_class

        return {'f1_macro': f1_macro_score, 'f1_weighted': f1_weighted_score, 'precision_macro': precision_macro_score, 'precision_weighted': precision_weighted_score, 'recall_macro': recall_macro_score, 'recall_weighted': recall_weighted_score, 'accuracy':acc, 'class_scores': f1_macro_score_per_class, 'skipped_nodes': len(self.data.test_nodes) - len(self.data.array_of_graphs_for_testing)}
    
    def map_cluster_labels_with_target_classes(self, cluster_labels, target_labels, test_node_list, test_node):
        # vouter algorithm
        uniq_targets = np.unique(target_labels)
        uniq_clusters = np.unique(cluster_labels)
        vouter = {cluster_class:{target_class:0 for target_class in uniq_targets} for cluster_class in uniq_clusters}
        
        checker = 0
        for i, n in enumerate(list(zip(test_node_list, cluster_labels))):
            node, _ = n
            if node != test_node:
                c_l = cluster_labels[i]
                t_l = target_labels[i]
                vouter[c_l][t_l] += 1
            else:
                checker = 1
            
        assert checker == 1
            
        mapping = dict()
        for k, v in vouter.items():
            mapping[k] = uniq_targets[np.argmax(list(v.values()))]
            
        return mapping
    
    def spectral_clustering_thread(self, test_node_idx):

        current_nodes = self.data.train_nodes + [self.data.test_nodes[test_node_idx]]
        G_test_init = self.data.nx_graph.subgraph(current_nodes).copy()
        # print(nx.number_connected_components(G_test)) ########################## check it for all datasets
        for c in nx.connected_components(G_test_init):
            if self.data.test_nodes[test_node_idx] in c:
                G_test = G_test_init.subgraph(c).copy()
        if len(G_test.nodes) == 1:
            # print('Isolated test node found, skipping!')
            return -1, -1, -1, 1
        elif len(G_test) <= len(self.data.classes):
            # print('Too few nodes!!! Skipping!!!')
            return -1, -1, -1, 1
        else:
            L = nx.to_numpy_array(G_test, weight='ibd_sum')
            assert np.allclose(L, L.T, rtol=1e-05, atol=1e-08) # symmetry check
            L = csr_matrix(L)
            # L = nx.normalized_laplacian_matrix(G_test, weight='ibd_sum' if use_weight else None) # node order like in G.nodes
            clustering = SpectralClustering(n_clusters=int(len(self.data.classes)), assign_labels='discretize', random_state=42, affinity='precomputed', eigen_solver="lobpcg").fit(L)
            preds = clustering.labels_

            ground_truth = []
            nodes_classes = nx.get_node_attributes(G_test, name='class')
            # print(len(G_test.nodes))
            # print(nodes_ibd_sum)
            for node in G_test.nodes:
                ground_truth.append(nodes_classes[node])

            graph_test_node_list = list(G_test.nodes)
            
            y_pred_cluster = preds[graph_test_node_list.index(self.data.test_nodes[test_node_idx])]
            y_true = ground_truth[graph_test_node_list.index(self.data.test_nodes[test_node_idx])]

            cluster2target_mapping = self.map_cluster_labels_with_target_classes(preds, ground_truth, graph_test_node_list, self.data.test_nodes[test_node_idx])
            y_pred_classes = cluster2target_mapping[preds[graph_test_node_list.index(self.data.test_nodes[test_node_idx])]]
            
            return y_pred_classes, y_pred_cluster, y_true, 0
        
    
    def spectral_clustering(self, use_weight=False, random_state=42):
        y_pred_classes = []
        y_pred_cluster = []
        y_true = []
        skipped_nodes = []
        
        with Pool(os.cpu_count()) as p: # os.cpu_count()
            res = list(tqdm(p.imap(self.spectral_clustering_thread, range(len(self.data.test_nodes))), total=len(self.data.test_nodes), desc='Spectral clustering'))
        
        for item in res:
            y_pred_classes.append(item[0])
            y_pred_cluster.append(item[1])
            y_true.append(item[2])
            skipped_nodes.append(item[3])
            
        y_pred_classes = np.array(y_pred_classes)
        y_pred_cluster = np.array(y_pred_cluster)
        y_true = np.array(y_true)
        
        y_pred_classes = y_pred_classes[y_pred_classes != -1]
        y_pred_cluster = y_pred_cluster[y_pred_cluster != -1]
        y_true = y_true[y_true != -1]
                
        # print(f'Homogenity score: {homogeneity_score(y_true, y_pred_cluster)}')
        
        f1_macro_score = f1_score(y_true, y_pred_classes, average='macro')
        # print(f"f1 macro score on test dataset: {f1_macro_score}")
        
        f1_weighted_score = f1_score(y_true, y_pred_classes, average='weighted')
        # print(f"f1 weighted score on test dataset: {f1_weighted_score}")
        
        acc = accuracy_score(y_true, y_pred_classes)
        # print(f"accuracy score on test dataset: {acc}")

        recall_macro_score = recall_score(y_true, y_pred_classes, average='macro')
        recall_weighted_score = recall_score(y_true, y_pred_classes, average='weighted')
        precision_macro_score = precision_score(y_true, y_pred_classes, average='macro')
        precision_weighted_score = precision_score(y_true, y_pred_classes, average='weighted')
        
        f1_macro_score_per_class = dict()
        
        for i in range(len(self.data.classes)):
            score_per_class = f1_score(y_true, y_pred_classes, average='macro', labels=[i])
            # print(f"f1 macro score on test dataset for class {i} which is {self.data.classes[i]}: {score_per_class}")
            f1_macro_score_per_class[self.data.classes[i]] = score_per_class

        return {'f1_macro': f1_macro_score, 'f1_weighted': f1_weighted_score, 'precision_macro': precision_macro_score, 'precision_weighted': precision_weighted_score, 'recall_macro': recall_macro_score, 'recall_weighted': recall_weighted_score, 'accuracy':acc, 'class_scores': f1_macro_score_per_class, 'skipped_nodes': sum(skipped_nodes)}
    
    
    def simrank_distance(self, G):
        simrank = nx.simrank_similarity(G)
        simrank_matrix = []
        for k in simrank.keys():
            simrank_matrix.append(list(simrank[k].values()))

        return np.round(1 - np.array(simrank_matrix), 6) # check order of nodes
    
    def plot_dendogram(self, test_node, fig_size, leaf_font_size, save_path=None):

        current_nodes = self.data.train_nodes + [test_node]
        G_test_init = self.data.nx_graph.subgraph(current_nodes).copy()
        
        distance = self.simrank_distance(G_test_init)
        
        plt.figure(figsize=fig_size)
        linked = linkage(squareform(distance), 'complete')
        dendrogram(linked, labels=list(G_test_init.nodes),
                   leaf_font_size=leaf_font_size)
        # plt.plot([0, len(G_test_init.nodes)+1], [0.89, 0.89], linestyle='--', c='tab:red')
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
    
    def agglomerative_clustering(self):
        y_pred_classes = []
        y_pred_cluster = []
        y_true = []
        skipped_nodes = 0
        
        for i in tqdm(range(len(self.data.test_nodes)), desc='Agglomerative clustering'):

            current_nodes = self.data.train_nodes + [self.data.test_nodes[i]]
            G_test_init = self.data.nx_graph.subgraph(current_nodes).copy()
            for c in nx.connected_components(G_test_init):
                if self.data.test_nodes[i] in c:
                    G_test = G_test_init.subgraph(c).copy()
            if len(G_test.nodes) == 1:
                # print('Isolated test node found, skipping!')
                skipped_nodes += 1
                continue
            elif len(G_test) <= len(self.data.classes):
                # print('Too few nodes!!! Skipping!!!')
                skipped_nodes += 1
                continue
            else:
                # print(len(G_test))
                distance = self.simrank_distance(G_test)
                preds = AgglomerativeClustering(n_clusters=int(len(self.data.classes)), linkage='complete', compute_full_tree=True, metric='precomputed').fit_predict(distance)

                ground_truth = []
                nodes_classes = nx.get_node_attributes(G_test, name='class')
                # print(len(G_test.nodes))
                # print(nodes_ibd_sum)
                for node in G_test.nodes:
                    ground_truth.append(nodes_classes[node])

                graph_test_node_list = list(G_test.nodes)
                y_pred_cluster.append(preds[graph_test_node_list.index(self.data.test_nodes[i])])
                y_true.append(ground_truth[graph_test_node_list.index(self.data.test_nodes[i])])

                cluster2target_mapping = self.map_cluster_labels_with_target_classes(preds, ground_truth, graph_test_node_list, self.data.test_nodes[i])
                y_pred_classes.append(cluster2target_mapping[preds[graph_test_node_list.index(self.data.test_nodes[i])]])
                
        # print(f'Homogenity score: {homogeneity_score(y_true, y_pred_cluster)}')
        
        f1_macro_score = f1_score(y_true, y_pred_classes, average='macro')
        # print(f"f1 macro score on test dataset: {f1_macro_score}")
        
        f1_weighted_score = f1_score(y_true, y_pred_classes, average='weighted')
        # print(f"f1 weighted score on test dataset: {f1_weighted_score}")
        
        acc = accuracy_score(y_true, y_pred_classes)
        # print(f"accuracy score on test dataset: {acc}")

        recall_macro_score = recall_score(y_true, y_pred_classes, average='macro')
        recall_weighted_score = recall_score(y_true, y_pred_classes, average='weighted')
        precision_macro_score = precision_score(y_true, y_pred_classes, average='macro')
        precision_weighted_score = precision_score(y_true, y_pred_classes, average='weighted')
        
        f1_macro_score_per_class = dict()
        
        for i in range(len(self.data.classes)):
            score_per_class = f1_score(y_true, y_pred_classes, average='macro', labels=[i])
            # print(f"f1 macro score on test dataset for class {i} which is {self.data.classes[i]}: {score_per_class}")
            f1_macro_score_per_class[self.data.classes[i]] = score_per_class

        return {'f1_macro': f1_macro_score, 'f1_weighted': f1_weighted_score, 'precision_macro': precision_macro_score, 'precision_weighted': precision_weighted_score, 'recall_macro': recall_macro_score, 'recall_weighted': recall_weighted_score, 'accuracy':acc, 'class_scores': f1_macro_score_per_class, 'skipped_nodes': skipped_nodes}
    
    def girvan_newman_thread(self, test_node_idx):

        current_nodes = self.data.train_nodes + [self.data.test_nodes[test_node_idx]]
        G_test_init = self.data.nx_graph.subgraph(current_nodes).copy()
        for c in nx.connected_components(G_test_init):
            if self.data.test_nodes[test_node_idx] in c:
                G_test = G_test_init.subgraph(c).copy()
        if len(G_test.nodes) == 1:
            # print('Isolated test node found, skipping!')
            return -1, -1, -1, 1
        else:
            comp = nx.community.girvan_newman(G_test.copy())
            for communities in itertools.islice(comp, int(len(self.data.classes))):
                preds_nodes_per_cluster = communities

            preds_nodes_per_cluster = [list(c) for c in preds_nodes_per_cluster]

            preds = []
            for j in range(len(preds_nodes_per_cluster)):
                curr_cluster = preds_nodes_per_cluster[j]
                for cl in range(len(curr_cluster)):
                    preds.append(j)

            graph_test_node_list = np.array([x for xx in preds_nodes_per_cluster for x in xx])
            sorting_arguments = np.argsort(graph_test_node_list)
            graph_test_node_list = list(graph_test_node_list[sorting_arguments])
            preds = np.array(preds)[sorting_arguments]

            # print(graph_test_node_list)
            # print(preds)

            ground_truth = []
            nodes_classes = nx.get_node_attributes(G_test, name='class')

            for node in graph_test_node_list:
                ground_truth.append(nodes_classes[node])


            y_pred_cluster = preds[graph_test_node_list.index(self.data.test_nodes[test_node_idx])]
            y_true = ground_truth[graph_test_node_list.index(self.data.test_nodes[test_node_idx])]

            cluster2target_mapping = self.map_cluster_labels_with_target_classes(preds, ground_truth)
            y_pred_classes = cluster2target_mapping[preds[graph_test_node_list.index(self.data.test_nodes[test_node_idx])]]
            
            return y_pred_classes, y_pred_cluster, y_true, 0

    
    def girvan_newman(self):
        y_pred_classes = []
        y_pred_cluster = []
        y_true = []
        skipped_nodes = []
        
        with Pool(os.cpu_count()) as p: # os.cpu_count()
            res = list(tqdm(p.imap(self.girvan_newman_thread, range(len(self.data.test_nodes))), total=len(self.data.test_nodes), desc='Girvan-Newman'))
        
        for item in res:
            y_pred_classes.append(item[0])
            y_pred_cluster.append(item[1])
            y_true.append(item[2])
            skipped_nodes.append(item[3])
            
        y_pred_classes = np.array(y_pred_classes)
        y_pred_cluster = np.array(y_pred_cluster)
        y_true = np.array(y_true)
        
        y_pred_classes = y_pred_classes[y_pred_classes != -1]
        y_pred_cluster = y_pred_cluster[y_pred_cluster != -1]
        y_true = y_true[y_true != -1]
                
        # print(f'Homogenity score: {homogeneity_score(y_true, y_pred_cluster)}')
        
        f1_macro_score = f1_score(y_true, y_pred_classes, average='macro')
        # print(f"f1 macro score on test dataset: {f1_macro_score}")
        
        f1_weighted_score = f1_score(y_true, y_pred_classes, average='weighted')
        # print(f"f1 weighted score on test dataset: {f1_weighted_score}")
        
        acc = accuracy_score(y_true, y_pred_classes)
        # print(f"accuracy score on test dataset: {acc}")

        recall_macro_score = recall_score(y_true, y_pred_classes, average='macro')
        recall_weighted_score = recall_score(y_true, y_pred_classes, average='weighted')
        precision_macro_score = precision_score(y_true, y_pred_classes, average='macro')
        precision_weighted_score = precision_score(y_true, y_pred_classes, average='weighted')
        
        f1_macro_score_per_class = dict()
        
        for i in range(len(self.data.classes)):
            score_per_class = f1_score(y_true, y_pred_classes, average='macro', labels=[i])
            # print(f"f1 macro score on test dataset for class {i} which is {self.data.classes[i]}: {score_per_class}")
            f1_macro_score_per_class[self.data.classes[i]] = score_per_class

        return {'f1_macro': f1_macro_score, 'f1_weighted': f1_weighted_score, 'precision_macro': precision_macro_score, 'precision_weighted': precision_weighted_score, 'recall_macro': recall_macro_score, 'recall_weighted': recall_weighted_score, 'accuracy':acc, 'class_scores': f1_macro_score_per_class, 'skipped_nodes': sum(skipped_nodes)}
        
        
    def initial_conditional(self, G, y_labeled, x_labeled):
        probs = np.ones((len(G.nodes), len(self.data.classes)))
        
        graph_nodes = list(G.nodes)
        
        # one for labeled nodes
        for i in range(len(x_labeled)):
            probs[graph_nodes.index(x_labeled[i])] = 0
            probs[graph_nodes.index(x_labeled[i]), y_labeled[i]] = 1
        # probs[x_labeled] = 0
        # probs[x_labeled, y_labeled] = 1
        
        assert np.sum(probs.sum(axis=1) > 1) == 1

        probs = probs / probs.sum(1, keepdims=1)

        return probs
    
    def update_conditional(self, A, cond, x_labeled, graph_nodes):

        new_cond = A @ cond
    
        for i in range(len(x_labeled)):
            new_cond[graph_nodes.index(x_labeled[i])] = cond[[graph_nodes.index(x_labeled[i])]]
        # new_cond[x_labeled] = cond[x_labeled]
        new_cond = new_cond / new_cond.sum(1, keepdims=1)
        return new_cond
        
    def relational_neighbor_classifier_core(self, G, threshold, x_labeled, x_unlabeled, y_labeled):
        cond = self.initial_conditional(G, y_labeled, x_labeled)
        A = nx.to_numpy_array(G)
        diffs = []
        diff = np.inf
        graph_nodes = list(G.nodes)
        while diff > threshold:
            # print(diff)
            next_cond = self.update_conditional(A, cond, x_labeled, graph_nodes)
            # print(np.all(next_cond == cond))
            diff = np.linalg.norm(cond[graph_nodes.index(x_unlabeled[0])] - next_cond[graph_nodes.index(x_unlabeled[0])])
            diffs.append(diff)
            cond = next_cond
        return np.argmax(cond, axis=1)
    
        
    def relational_neighbor_classifier(self, threshold):
        y_pred_classes = []
        y_pred_cluster = []
        y_true = []
        skipped_nodes = 0
        
        for i in tqdm(range(len(self.data.test_nodes)), desc='Relational classifier'):
            current_nodes = self.data.train_nodes + [self.data.test_nodes[i]]
            G_test_init = self.data.nx_graph.subgraph(current_nodes).copy()
            for c in nx.connected_components(G_test_init):
                if self.data.test_nodes[i] in c:
                    G_test = G_test_init.subgraph(c).copy()
            if len(G_test.nodes) == 1:
                # print('Isolated test node found, skipping!')
                skipped_nodes += 1
                continue
            elif len(G_test) <= len(self.data.classes):
                # print('Too few nodes!!! Skipping!!!')
                skipped_nodes += 1
                continue
            else:
                
                ground_truth_all = []
                ground_truth_train_nodes_only = []
                nodes_classes = nx.get_node_attributes(G_test, name='class')
                for node in G_test.nodes:
                    ground_truth_all.append(nodes_classes[node])
                    if node != self.data.test_nodes[i]:
                        ground_truth_train_nodes_only.append(nodes_classes[node])
                cc_train_nodes = np.array(list(G_test.nodes))
                cc_train_nodes = cc_train_nodes[cc_train_nodes != self.data.test_nodes[i]]
                assert len(ground_truth_train_nodes_only) == len(cc_train_nodes)
                preds = self.relational_neighbor_classifier_core(G_test, threshold, cc_train_nodes, np.array([self.data.test_nodes[i]]), np.array(ground_truth_train_nodes_only)) # ground_truth contains classes for ALL nodes, includind test node

                graph_test_node_list = list(G_test.nodes)
                y_true.append(ground_truth_all[graph_test_node_list.index(self.data.test_nodes[i])])

                y_pred_classes.append(preds[graph_test_node_list.index(self.data.test_nodes[i])])
        
        f1_macro_score = f1_score(y_true, y_pred_classes, average='macro')
        # print(f"f1 macro score on test dataset: {f1_macro_score}")
        
        f1_weighted_score = f1_score(y_true, y_pred_classes, average='weighted')
        # print(f"f1 weighted score on test dataset: {f1_weighted_score}")
        
        acc = accuracy_score(y_true, y_pred_classes)
        # print(f"accuracy score on test dataset: {acc}")

        recall_macro_score = recall_score(y_true, y_pred_classes, average='macro')
        recall_weighted_score = recall_score(y_true, y_pred_classes, average='weighted')
        precision_macro_score = precision_score(y_true, y_pred_classes, average='macro')
        precision_weighted_score = precision_score(y_true, y_pred_classes, average='weighted')
        
        f1_macro_score_per_class = dict()
        
        for i in range(len(self.data.classes)):
            score_per_class = f1_score(y_true, y_pred_classes, average='macro', labels=[i])
            # print(f"f1 macro score on test dataset for class {i} which is {self.data.classes[i]}: {score_per_class}")
            f1_macro_score_per_class[self.data.classes[i]] = score_per_class

        return {'f1_macro': f1_macro_score, 'f1_weighted': f1_weighted_score, 'precision_macro': precision_macro_score, 'precision_weighted': precision_weighted_score, 'recall_macro': recall_macro_score, 'recall_weighted': recall_weighted_score, 'accuracy':acc, 'class_scores': f1_macro_score_per_class, 'skipped_nodes': skipped_nodes}
    
    
    def multi_rank_walk_core(self, G, x_labeled, x_unlabeled, y_labeled, alpha):
        n_classes = len(self.data.classes)
        y_pred = np.zeros((len(G), n_classes))
        for c in range(n_classes):
            y_pred[:, c] = self.personalized_pr(G, y_labeled, x_labeled, c, alpha)
        return y_pred.argmax(axis=1)#[x_unlabeled]
    
    
    def personalized_pr(self, G, y_labeled, x_labeled, c, alpha):
        important_nodes = dict()
        for i in range(len(y_labeled)):
            if y_labeled[i] == c:
                important_nodes[x_labeled[i]] = 1 / np.sum(y_labeled == c)
        # print(len(G), important_nodes)
        return np.array(list(nx.pagerank(G, personalization=important_nodes if len(important_nodes) > 0 else None, alpha=alpha).values()))
    
    
    def multi_rank_walk(self, alpha):
        y_pred_classes = []
        y_pred_cluster = []
        y_true = []
        skipped_nodes = 0
        
        for i in tqdm(range(len(self.data.test_nodes)), desc='Multi rank walk'):
            current_nodes = self.data.train_nodes + [self.data.test_nodes[i]]
            G_test_init = self.data.nx_graph.subgraph(current_nodes).copy()
            for c in nx.connected_components(G_test_init):
                if self.data.test_nodes[i] in c:
                    G_test = G_test_init.subgraph(c).copy()
            if len(G_test.nodes) == 1:
                # print('Isolated test node found, skipping!')
                skipped_nodes += 1
                continue
            elif len(G_test) <= len(self.data.classes):
                # print('Too few nodes!!! Skipping!!!')
                skipped_nodes += 1
                continue
            else:
                
                ground_truth_all = []
                ground_truth_train_nodes_only = []
                nodes_classes = nx.get_node_attributes(G_test, name='class')
                for node in G_test.nodes:
                    ground_truth_all.append(nodes_classes[node])
                    if node != self.data.test_nodes[i]:
                        ground_truth_train_nodes_only.append(nodes_classes[node])
                cc_train_nodes = np.array(list(G_test.nodes))
                cc_train_nodes = cc_train_nodes[cc_train_nodes != self.data.test_nodes[i]]
                preds = self.multi_rank_walk_core(G_test, cc_train_nodes, np.array([self.data.test_nodes[i]]), np.array(ground_truth_train_nodes_only), alpha)

                graph_test_node_list = list(G_test.nodes)
                y_true.append(ground_truth_all[graph_test_node_list.index(self.data.test_nodes[i])])

                y_pred_classes.append(preds[graph_test_node_list.index(self.data.test_nodes[i])])
        
        f1_macro_score = f1_score(y_true, y_pred_classes, average='macro')
        # print(f"f1 macro score on test dataset: {f1_macro_score}")
        
        f1_weighted_score = f1_score(y_true, y_pred_classes, average='weighted')
        # print(f"f1 weighted score on test dataset: {f1_weighted_score}")
        
        acc = accuracy_score(y_true, y_pred_classes)
        # print(f"accuracy score on test dataset: {acc}")

        recall_macro_score = recall_score(y_true, y_pred_classes, average='macro')
        recall_weighted_score = recall_score(y_true, y_pred_classes, average='weighted')
        precision_macro_score = precision_score(y_true, y_pred_classes, average='macro')
        precision_weighted_score = precision_score(y_true, y_pred_classes, average='weighted')
        
        f1_macro_score_per_class = dict()
        
        for i in range(len(self.data.classes)):
            score_per_class = f1_score(y_true, y_pred_classes, average='macro', labels=[i])
            # print(f"f1 macro score on test dataset for class {i} which is {self.data.classes[i]}: {score_per_class}")
            f1_macro_score_per_class[self.data.classes[i]] = float(score_per_class)

        return {'f1_macro': f1_macro_score, 'f1_weighted': f1_weighted_score, 'precision_macro': precision_macro_score, 'precision_weighted': precision_weighted_score, 'recall_macro': recall_macro_score, 'recall_weighted': recall_weighted_score, 'accuracy':acc, 'class_scores': f1_macro_score_per_class, 'skipped_nodes': skipped_nodes}
        
        
    def tikhonov_regularization(self, G, gamma, x_labeled, y_labeled, p):

        from numpy.linalg import inv
        
        graph_nodes = list(G.nodes)

        num_nodes = G.number_of_nodes()

        A = nx.adjacency_matrix(G)
        D = np.diag(A.sum(axis=1))
        L = D - A

        L = np.linalg.matrix_power(L, p)
        S = L

        I = np.diag([1 if i in x_labeled else 0 for i in range(num_nodes)])

        y = np.zeros(num_nodes)
        y_mean = np.mean(y_labeled)
        for i in range(len(x_labeled)):
            y[graph_nodes.index(x_labeled[i])] = y_labeled[i] - y_mean

        A = len(y_labeled) * gamma * S + I
        A_inv = np.linalg.inv(A)

        f_t = A_inv @ y

        return f_t + y_mean
        
        
    def ridge_regression(self, gamma, p):
        y_pred_classes = []
        y_pred_cluster = []
        y_true = []
        skipped_nodes = 0
        
        for i in tqdm(range(len(self.data.test_nodes)), desc='Ridge regression'):
            current_nodes = self.data.train_nodes + [self.data.test_nodes[i]]
            G_test_init = self.data.nx_graph.subgraph(current_nodes).copy()
            for c in nx.connected_components(G_test_init):
                if self.data.test_nodes[i] in c:
                    G_test = G_test_init.subgraph(c).copy()
            if len(G_test.nodes) == 1:
                # print('Isolated test node found, skipping!')
                skipped_nodes += 1
                continue
            elif len(G_test) <= len(self.data.classes):
                # print('Too few nodes!!! Skipping!!!')
                skipped_nodes += 1
                continue
            else:
                
                ground_truth_all = []
                ground_truth_train_nodes_only = []
                nodes_classes = nx.get_node_attributes(G_test, name='class')
                for node in G_test.nodes:
                    ground_truth_all.append(nodes_classes[node])
                    if node != self.data.test_nodes[i]:
                        ground_truth_train_nodes_only.append(nodes_classes[node])
                cc_train_nodes = np.array(list(G_test.nodes))
                cc_train_nodes = cc_train_nodes[cc_train_nodes != self.data.test_nodes[i]]
                preds = np.round(self.tikhonov_regularization(G_test, gamma, cc_train_nodes, np.array(ground_truth_train_nodes_only), p)).astype(int)

                graph_test_node_list = list(G_test.nodes)
                y_true.append(ground_truth_all[graph_test_node_list.index(self.data.test_nodes[i])])

                y_pred_classes.append(preds[graph_test_node_list.index(self.data.test_nodes[i])])
        
        f1_macro_score = f1_score(y_true, y_pred_classes, average='macro')
        # print(f"f1 macro score on test dataset: {f1_macro_score}")
        
        f1_weighted_score = f1_score(y_true, y_pred_classes, average='weighted')
        # print(f"f1 weighted score on test dataset: {f1_weighted_score}")
        
        acc = accuracy_score(y_true, y_pred_classes)
        # print(f"accuracy score on test dataset: {acc}")

        recall_macro_score = recall_score(y_true, y_pred_classes, average='macro')
        recall_weighted_score = recall_score(y_true, y_pred_classes, average='weighted')
        precision_macro_score = precision_score(y_true, y_pred_classes, average='macro')
        precision_weighted_score = precision_score(y_true, y_pred_classes, average='weighted')
        
        f1_macro_score_per_class = dict()
        
        for i in range(len(self.data.classes)):
            score_per_class = f1_score(y_true, y_pred_classes, average='macro', labels=[i])
            # print(f"f1 macro score on test dataset for class {i} which is {self.data.classes[i]}: {score_per_class}")
            f1_macro_score_per_class[self.data.classes[i]] = score_per_class

        return {'f1_macro': f1_macro_score, 'f1_weighted': f1_weighted_score, 'precision_macro': precision_macro_score, 'precision_weighted': precision_weighted_score, 'recall_macro': recall_macro_score, 'recall_weighted': recall_weighted_score, 'accuracy':acc, 'class_scores': f1_macro_score_per_class, 'skipped_nodes': skipped_nodes}
    
        
    def run_community_detection(self, heuristic_name):
        if heuristic_name == 'LabelPropagation':
            self.data.make_train_valid_test_datasets_with_numba(feature_type='one_hot', 
                                                                model_type='homogeneous', 
                                                                train_dataset_type='multiple', 
                                                                test_dataset_type='multiple',
                                                                masking=False,
                                                                no_mask_class_in_df=True)
            return self.torch_geometric_label_propagation(1, 0.0001)
        elif heuristic_name == 'GirvanNewmann':
            return self.girvan_newman()
        elif heuristic_name == 'AgglomerativeClustering':
            return self.agglomerative_clustering()
        elif heuristic_name == 'SpectralClustering':
            return self.spectral_clustering()
        elif heuristic_name == 'RelationalNeighborClassifier':
            return self.relational_neighbor_classifier(0.001)
            
