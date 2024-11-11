import sys
import os
import torch
import pickle
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(''), os.path.pardir)))
from utils.genlink import DataProcessor

dp = DataProcessor('/mnt/10tb/home/shmelev/genlink_real_data/NC_graph_rel_eng.csv')
dp.get_graph_features(fig_path='/mnt/10tb/home/shmelev/GENLINK/pictures/mane_NC/', fig_size=(10, 6), dataset_name='NC')

dp = DataProcessor('/mnt/10tb/home/shmelev/genlink_real_data/Scandinavia.csv')
dp.get_graph_features(fig_path='/mnt/10tb/home/shmelev/GENLINK/pictures/mane_SC/', fig_size=(10, 6), dataset_name='SC')

dp = DataProcessor('/mnt/10tb/home/shmelev/genlink_real_data/Volga.csv')
dp.get_graph_features(fig_path='/mnt/10tb/home/shmelev/GENLINK/pictures/mane_VU/', fig_size=(10, 6), dataset_name='VU')

dp = DataProcessor('/mnt/10tb/home/shmelev/genlink_real_data/Western-Europe.csv')
dp.get_graph_features(fig_path='/mnt/10tb/home/shmelev/GENLINK/pictures/mane_WE/', fig_size=(10, 6), dataset_name='WE')