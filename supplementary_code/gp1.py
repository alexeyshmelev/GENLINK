import sys
import os
import torch
import pickle
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(''), os.path.pardir)))
from utils.genlink import DataProcessor

dp = DataProcessor('/mnt/10tb/home/shmelev/genlink_real_data_alike_simulated/Western-Europe_non_diagonal_edge_prob_add_0.0.csv')
dp.get_graph_features(fig_path='/mnt/10tb/home/shmelev/GENLINK/pictures/correct_simulation_WE_sim_1/', fig_size=(10, 6), dataset_name='WE_sim_1')