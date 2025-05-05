import argparse
from os import listdir
from os.path import isfile, join
import json
from genlink import DataProcessor, NullSimulator
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GENLINK_simulator')

    parser.add_argument('--source_folder', type=str, help='Folder that contain initial graphs')
    parser.add_argument('--destination_folder', type=str, help='Folder that contain simulated graphs')
    parser.add_argument('--simulation_params', type=str, help='Path to .json file that contain parameters for simulation')
    parser.add_argument('--random_state', type=int, help='Random state')

    args = parser.parse_args()

    data_files = [join(args.source_folder, f) for f in listdir(args.source_folder) if isfile(join(args.source_folder, f))]
    for f in data_files:
        print(f'Observed data file: {f}')

    with open(args.simulation_params, 'r') as f:
        simulation_params = json.load(f)

    if simulation_params['type'] == 'non_diagonal_edge_prob_change' or simulation_params['type'] == 'equal_probs' or simulation_params['type'] == 'prob_decreaser' or simulation_params['type'] == 'maxed_equal_class_balance' or simulation_params['type'] == 'class_balance_interpolation' or simulation_params['type'] == 'diagonal_prob_changer':
        for f in data_files:
            if 'prob_boost' in simulation_params.keys() and simulation_params['prob_boost'] is not None:
                global_iterator = simulation_params['prob_boost']
            elif 'edge_prob_drop' in simulation_params.keys() and simulation_params['edge_prob_drop'] is not None:
                global_iterator = simulation_params['edge_prob_drop']
            elif 'num_interpolation_steps' in simulation_params.keys() and simulation_params['num_interpolation_steps'] is not None:
                global_iterator = range(simulation_params['num_interpolation_steps']+1)
            elif 'diagonal_prob_multipliers' in simulation_params.keys() and simulation_params['diagonal_prob_multipliers'] is not None:
                global_iterator = simulation_params['diagonal_prob_multipliers']
            else:
                global_iterator = range(1)

            for iterator_item in global_iterator:

                dp = DataProcessor(f)
                dp.compute_simulation_params()

                pop_sizes = []
                real_pop_sizes = dp.node_classes_sorted.iloc[:, 1].value_counts()
                for i in range(len(dp.classes)):
                    pop_sizes.append(real_pop_sizes.loc[i])

                if simulation_params['type'] == 'non_diagonal_edge_prob_change':
                    new_edge_prob = dp.edge_probs.copy()
                    new_edge_prob_diag = new_edge_prob.diagonal()
                    new_edge_prob = new_edge_prob + iterator_item
                    np.fill_diagonal(new_edge_prob, new_edge_prob_diag)
                elif simulation_params['type'] == 'diagonal_prob_changer':
                    new_edge_prob = dp.edge_probs.copy()
                    new_edge_prob_diag = new_edge_prob.diagonal() * iterator_item
                    np.fill_diagonal(new_edge_prob, new_edge_prob_diag)
                elif simulation_params['type'] == 'equal_probs':
                    new_edge_prob = np.full(dp.edge_probs.shape, simulation_params['equal_edge_prob_value'])
                elif simulation_params['type'] == 'prob_decreaser':
                    new_edge_prob = dp.edge_probs.copy()
                    new_edge_prob = new_edge_prob * iterator_item
                elif simulation_params['type'] == 'maxed_equal_class_balance':
                    new_edge_prob = dp.edge_probs.copy()
                    pop_sizes = [np.max(pop_sizes)] * len(pop_sizes)
                elif simulation_params['type'] == 'class_balance_interpolation':
                    new_edge_prob = dp.edge_probs.copy()
                    back_sort = np.argsort(np.argsort(pop_sizes))
                    old_pop_sizes = np.array(pop_sizes).copy()
                    pop_sizes = np.linspace(np.min(pop_sizes) + (np.max(pop_sizes) - np.min(pop_sizes)) / (len(global_iterator)-1) * iterator_item, np.max(pop_sizes), len(pop_sizes)).astype(int)[back_sort]
                    pop_sizes = (pop_sizes / pop_sizes.sum() * old_pop_sizes.sum()).astype(int)
                    print(f'Population size balance: {pop_sizes}, total size is {pop_sizes.sum()}, real size is {old_pop_sizes.sum()}')


                ns = NullSimulator(len(pop_sizes), np.nan_to_num(new_edge_prob), dp.mean_weight)
                counts, means, pop_index = ns.generate_matrices(np.array(pop_sizes), np.random.default_rng(args.random_state))
                # print(means)

                if simulation_params['type'] == 'non_diagonal_edge_prob_change':
                    new_file_name = f.split('/')[-1].split('.')[0] + f'_non_diagonal_edge_prob_add_{iterator_item}.csv'
                elif simulation_params['type'] == 'diagonal_prob_changer':
                    new_file_name = f.split('/')[-1].split('.')[0] + f'_diagonal_probs_multiplied_by_0_{int(iterator_item*100)}.csv'
                elif simulation_params['type'] == 'equal_probs':
                    new_file_name = f.split('/')[-1].split('.')[0] + f'_all_probs_{simulation_params["equal_edge_prob_value"]}.csv'
                elif simulation_params['type'] == 'prob_decreaser':
                    new_file_name = f.split('/')[-1].split('.')[0] + f'_decreased_by_0_{int(iterator_item*100)}.csv'
                elif simulation_params['type'] == 'maxed_equal_class_balance':
                    new_file_name = f.split('/')[-1].split('.')[0] + f'_maxed_equal_class_balance.csv'
                elif simulation_params['type'] == 'class_balance_interpolation':
                    new_file_name = f.split('/')[-1].split('.')[0] + f'_class_balance_interpolation_step_{iterator_item}.csv'
                

                ns.simulate_graph(means, counts, pop_index, join(args.destination_folder, new_file_name))
