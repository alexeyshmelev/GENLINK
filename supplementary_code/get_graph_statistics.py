import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(''), os.path.pardir)))
from utils.genlink import DataProcessor

dp = DataProcessor('/disk/10tb/home/shmelev/CR_real_masks_more_labeled_veritices/CR_real_masks_more_labeled_veritices.csv')
features = dp.get_graph_features(fig_path='/disk/10tb/home/shmelev/GENLINK/pictures/CR_real_masks/', 
                                 fig_size=(10, 6), 
                                 dataset_name='CR_real_masks_more_labeled_veritices',
                                 simple_run=True)