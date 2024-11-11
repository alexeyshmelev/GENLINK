import sys
import os
import torch
import pickle
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(''), os.path.pardir)))
from utils.genlink import DataProcessor

dp = DataProcessor('/mnt/10tb/home/shmelev/genlink_real_data_alike_simulated_1/NC_graph_rel_eng_non_diagonal_edge_prob_add_0.0.csv')
dp.get_graph_features(fig_path='/mnt/10tb/home/shmelev/GENLINK/pictures/correct_simulation_NC_sim_2/', fig_size=(10, 6), dataset_name='NC_sim_2')

dp = DataProcessor('/mnt/10tb/home/shmelev/genlink_real_data_alike_simulated_1/Scandinavia_non_diagonal_edge_prob_add_0.0.csv')
dp.get_graph_features(fig_path='/mnt/10tb/home/shmelev/GENLINK/pictures/correct_simulation_SC_sim_2/', fig_size=(10, 6), dataset_name='SC_sim_2')

dp = DataProcessor('/mnt/10tb/home/shmelev/genlink_real_data_alike_simulated_1/Volga_non_diagonal_edge_prob_add_0.0.csv')
dp.get_graph_features(fig_path='/mnt/10tb/home/shmelev/GENLINK/pictures/correct_simulation_VU_sim_2/', fig_size=(10, 6), dataset_name='VU_sim_2')

dp = DataProcessor('/mnt/10tb/home/shmelev/genlink_real_data_alike_simulated_1/Western-Europe_non_diagonal_edge_prob_add_0.0.csv')
dp.get_graph_features(fig_path='/mnt/10tb/home/shmelev/GENLINK/pictures/correct_simulation_WE_sim_2/', fig_size=(10, 6), dataset_name='WE_sim_2')