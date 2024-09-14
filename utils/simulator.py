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

    if simulation_params['type'] == 'non_diagonal_edge_prob_change' or simulation_params['type'] == 'equal_probs':
        for f in data_files:
            for prob_boost in simulation_params['prob_boost']:

                dp = DataProcessor(f)
                dp.compute_simulation_params()

                pop_sizes = []
                for i in range(len(dp.classes)):
                    pop_sizes.append(dp.node_classes_sorted.iloc[:, 1].value_counts().loc[i])

                if simulation_params['equal_edge_prob_value'] is None:
                    new_edge_prob = dp.edge_probs.copy()
                    new_edge_prob_diag = new_edge_prob.diagonal()
                    new_edge_prob = new_edge_prob + prob_boost
                    np.fill_diagonal(new_edge_prob, new_edge_prob_diag)
                else:
                    new_edge_prob = np.full(dp.edge_probs.shape, simulation_params['equal_edge_prob_value'])

                ns = NullSimulator(len(pop_sizes), np.nan_to_num(new_edge_prob), dp.mean_weight)
                means, counts, pop_index = ns.generate_matrices(np.array(pop_sizes), np.random.default_rng(args.random_state))

                if simulation_params['equal_edge_prob_value'] is None:
                    new_file_name = f.split('/')[-1].split('.')[0] + f'_non_diagonal_edge_prob_add_{prob_boost}.csv'
                else:
                    new_file_name = f.split('/')[-1].split('.')[0] + f'_all_probs_{simulation_params["equal_edge_prob_value"]}.csv'

                ns.simulate_graph(means, counts, pop_index, join(args.destination_folder, new_file_name))
