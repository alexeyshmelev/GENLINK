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

    if simulation_params['type'] == 'non_diagonal_edge_prob_change' or simulation_params['type'] == 'equal_probs' or simulation_params['type'] == 'prob_decreaser':
        for f in data_files:
            if simulation_params['prob_boost'] is not None:
                global_iterator = simulation_params['prob_boost']
            elif simulation_params['edge_prob_drop'] is not None:
                global_iterator = simulation_params['edge_prob_drop']
            else:
                global_iterator = range(1)

            for iterator_item in global_iterator:

                dp = DataProcessor(f)
                dp.compute_simulation_params()

                pop_sizes = []
                for i in range(len(dp.classes)):
                    pop_sizes.append(dp.node_classes_sorted.iloc[:, 1].value_counts().loc[i])

                if simulation_params['type'] == 'non_diagonal_edge_prob_change':
                    new_edge_prob = dp.edge_probs.copy()
                    new_edge_prob_diag = new_edge_prob.diagonal()
                    new_edge_prob = new_edge_prob + iterator_item
                    np.fill_diagonal(new_edge_prob, new_edge_prob_diag)
                elif simulation_params['type'] == 'equal_probs':
                    new_edge_prob = np.full(dp.edge_probs.shape, simulation_params['equal_edge_prob_value'])
                elif simulation_params['type'] == 'prob_decreaser':
                    new_edge_prob = dp.edge_probs.copy()
                    new_edge_prob = new_edge_prob * iterator_item

                ns = NullSimulator(len(pop_sizes), np.nan_to_num(new_edge_prob), dp.mean_weight)
                counts, means, pop_index = ns.generate_matrices(np.array(pop_sizes), np.random.default_rng(args.random_state))
                # print(means)

                if simulation_params['type'] == 'non_diagonal_edge_prob_change':
                    new_file_name = f.split('/')[-1].split('.')[0] + f'_non_diagonal_edge_prob_add_{iterator_item}.csv'
                elif simulation_params['type'] == 'equal_probs':
                    new_file_name = f.split('/')[-1].split('.')[0] + f'_all_probs_{simulation_params["equal_edge_prob_value"]}.csv'
                elif simulation_params['type'] == 'prob_decreaser':
                    new_file_name = f.split('/')[-1].split('.')[0] + f'_decreased_by_0_{int(iterator_item*100)}.csv'
                

                ns.simulate_graph(means, counts, pop_index, join(args.destination_folder, new_file_name))
