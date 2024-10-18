from genlink import DataProcessor, NullSimulator, Trainer, CommunityDetection, Heuristics
from multiprocessing import Manager, Array, current_process, get_context, Process, Lock
from tqdm import tqdm
import torch
import networkx as nx
from sklearn.model_selection import ParameterGrid
import numpy as np
import models
import os
import time
import json
import copy
import gc

class Runner:
    def __init__(self, data_files, feature_types, running_params, gnn_models, heuristic_models, community_detection_models, device, gpu_map, models_per_gpu):
        self.data_files = data_files
        self.feature_types = feature_types
        self.running_params = running_params
        self.gnn_models = gnn_models
        self.heuristic_models = heuristic_models
        self.community_detection_models = community_detection_models
        self.device = device
        self.gpu_map = gpu_map
        self.models_per_gpu = models_per_gpu
        self.datasets = dict()

    def gpu_runner_core(self, explist, lock):
        while True:

            with lock:
                if 0 not in shared_explist:
                    break
                else:
                    for i in range(len(shared_explist)):
                        if shared_explist[i] == 0:
                            shared_explist[i] = 1
                            curr_exp = explist[i]
                            break

            # print('LENLE', len(shared_explist))
            # print(shared_explist[:])
            # assert 0 == 1
            # raise Exception('hello')

            p = current_process()
            cp = p._identity[0] - 1
            print('process counter:', p._identity[0], 'pid:', os.getpid())
            gpu_idx = self.gpu_map[cp % len(self.device)]

            assert len(curr_exp) == 4

            feature_type = curr_exp[0]
            model = curr_exp[1]
            dataset_name = curr_exp[2]
            split_id = curr_exp[3]
            # print(feature_type, model, path, split_id)

            dataset = copy.deepcopy(self.datasets[dataset_name][split_id])
            # dataset = DataProcessor(path)
            # dataset.generate_random_train_valid_test_nodes(train_size=self.running_params['train_size'], 
            #                                                 valid_size=self.running_params['valid_size'], 
            #                                                 test_size=self.running_params['test_size'], 
            #                                                 random_state=self.running_params['seed'] + split_id,
            #                                                 save_dir=self.running_params['splits_save_dir'], 
            #                                                 mask_size=self.running_params['mask_size'],
            #                                                 sub_train_size=self.running_params['sub_train_size'], 
            #                                                 keep_train_nodes=self.running_params['keep_train_nodes'], 
            #                                                 mask_random_state=self.running_params['mask_random_state'])
            
            if self.running_params['mask_size'] is not None:
                masking=True
            else:
                masking=False

            if feature_type=='one_hot':
                dataset.make_train_valid_test_datasets_with_numba(feature_type=feature_type, 
                                                                model_type='homogeneous', 
                                                                train_dataset_type='multiple', 
                                                                test_dataset_type='multiple',
                                                                masking=masking,
                                                                no_mask_class_in_df=True,
                                                                log_edge_weights=self.running_params['log_ibd'])
            elif feature_type=='graph_based':
                dataset.make_train_valid_test_datasets_with_numba(feature_type=feature_type, 
                                                                model_type='homogeneous', 
                                                                train_dataset_type='one', 
                                                                test_dataset_type='multiple',
                                                                masking=masking,
                                                                no_mask_class_in_df=True,
                                                                log_edge_weights=self.running_params['log_ibd'])
            # select parameters for grid search
            curr_params = dict()
            curr_params['lr'] = self.running_params['lr']
            curr_params['wd'] = self.running_params['wd']
            curr_params['loss_weights'] = self.running_params['loss_weights'][dataset] if self.running_params['loss_weights'] != [None] else [None]
            curr_params['loss'] = self.running_params['loss']
            curr_params['patience'] = self.running_params['patience']
            if feature_type == 'graph_based':
                curr_params['train_iterations_per_sample'] = self.running_params['train_iterations_per_sample']
                curr_params['evaluation_steps'] = self.running_params['evaluation_steps']
            elif feature_type == 'one_hot':
                curr_params['num_epochs'] = self.running_params['num_epochs']

            curr_params_grid = list(ParameterGrid(curr_params))
            max_f1_macro_score = 0
            print(f'Here will be {len(curr_params_grid)} runs for model {model}')

            for curr_param in curr_params_grid:
                log_dir = self.running_params['log_dir'] + '/' + dataset_name + ('_log' if self.running_params['log_ibd'] else '') + '/' + model + '_' + feature_type + '_' + f'split_{split_id}'
                trainer = Trainer(data=dataset,
                                model_cls=getattr(models, model), 
                                lr=curr_param['lr'], 
                                wd=curr_param['wd'], 
                                loss_fn=getattr(torch.nn, curr_param['loss']), 
                                batch_size=self.running_params['batch_size'], 
                                log_dir=log_dir, 
                                patience=curr_param['patience'], 
                                num_epochs=curr_param['num_epochs'] if feature_type == 'one_hot' else None, 
                                feature_type=feature_type, 
                                train_iterations_per_sample=curr_param['train_iterations_per_sample'] if feature_type == 'graph_based' else None, 
                                evaluation_steps=curr_param['evaluation_steps'] if feature_type == 'graph_based' else None, 
                                weight=curr_param['loss_weights'],
                                masking=self.running_params['masking'], 
                                cuda_device_specified=gpu_idx,
                                disable_printing=self.running_params['disable_printing'],
                                seed=self.running_params['seed'], 
                                save_model_in_ram=False, 
                                correct_and_smooth=self.running_params['correct_and_smooth'], 
                                no_mask_class_in_df=True,
                                remove_saved_model_after_testing=True,
                                plot_cm=self.running_params['plot_cm'],
                                use_class_balance_weight=self.running_params['use_class_balance_weight'])
                results = trainer.run()

                if results['f1_macro'] >= max_f1_macro_score:
                    max_f1_macro_score = results['f1_macro']
                    with open(log_dir + f'/results.json', 'w') as f:
                        results['model_name'] = model
                        json.dump(results, f)
                    with open(log_dir + f'/curr_param.json', 'w') as f:
                        json.dump(curr_param, f)

            del results
            del trainer
            del dataset
            torch.cuda.empty_cache()
            gc.collect()
        # print('EXITING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


    def run(self):

        # generating splits for all models to avoid problems with seed
        for path in self.data_files:
            if self.running_params['sub_train_size'] is None:
                stss = [None]
            else:
                stss = self.running_params['sub_train_size']
            for sts in stss:
                dataset_name = path.split('/')[-1].split('.')[0]
                if len(stss) > 1:
                    dataset_name += f'_sts_{sts}'
                self.datasets[dataset_name] = dict()
                for s in range(self.running_params['num_splits']):
                    dataset = DataProcessor(path, dataset_name=dataset_name)
                    dataset.generate_random_train_valid_test_nodes(train_size=self.running_params['train_size'], 
                                                                valid_size=self.running_params['valid_size'], 
                                                                test_size=self.running_params['test_size'], 
                                                                random_state=self.running_params['seed'] + s,
                                                                save_dir=self.running_params['splits_save_dir'], 
                                                                mask_size=self.running_params['mask_size'],
                                                                sub_train_size=sts, 
                                                                keep_train_nodes=self.running_params['keep_train_nodes'], 
                                                                mask_random_state=self.running_params['mask_random_state'])
                    self.datasets[dataset_name][s] = dataset
                    if self.running_params['save_dataset_stats']:
                        dataset_stats_log_dir = self.running_params['log_dir'] + '/' + 'dataset_stats' + '/' + dataset_name + '/' + f'split_{s}'
                        if not os.path.exists(dataset_stats_log_dir):
                            os.makedirs(dataset_stats_log_dir)
                        dataset_copy = copy.deepcopy(dataset)
                        G = dataset_copy.nx_graph.subgraph(dataset_copy.train_nodes)

                        dataset_statistic = dict()
                        dataset_statistic['number_connected_components'] = nx.number_connected_components(G)
                        with open(dataset_stats_log_dir + f'/stats.json', 'w') as f:
                            json.dump(dataset_statistic, f)

        if len(self.gnn_models):
            explist = []
            for tuple_hash, feature_type in self.gnn_models.items():
                gnn_model_name, _ = tuple_hash
                for dataset_name in self.datasets.keys():
                    for s in range(self.running_params['num_splits']):
                        explist.append([feature_type, gnn_model_name, dataset_name, s])

            # if len(self.gnn_models):
                # with get_context("spawn").Pool(len(self.device) * self.models_per_gpu) as p:
                #     _ = list(tqdm(p.imap(self.gpu_runner_core, explist), total=len(explist), desc='Running training pipeline'))
            list_of_processes = []
            global shared_explist
            shared_explist = Array('i', [0] * len(explist), lock=True)
            lock = Lock()
            for i in range(len(self.device) * self.models_per_gpu):
                list_of_processes.append(Process(target=self.gpu_runner_core, args=(explist, lock)))
                list_of_processes[-1].start()
            
            for process in list_of_processes:
                process.join()

            # self.gpu_runner_core(explist, lock)

            

        
        if len(self.heuristic_models):
            for dataset_name in self.datasets.keys():
                for heuristic in tqdm(self.heuristic_models, desc=f"Running heuristics for {dataset_name}"): # make tqdm to show progress bar for everything at once
                    for s in range(self.running_params['num_splits']):
                        log_dir = self.running_params['log_dir'] + '/' + dataset_name + ('_log' if self.running_params['log_ibd'] else '') + '/' + heuristic + '_' + f'split_{s}'
                        if self.running_params['log_ibd']:
                            dataset = copy.deepcopy(self.datasets[dataset_name][s])
                            # edge_weights = nx.get_edge_attributes(dataset.nx_graph, 'ibd_sum')
                            for edge in dataset.nx_graph.edges:
                                dataset.nx_graph[edge[0]][edge[1]]['ibd_sum'] = -np.log2(dataset.nx_graph[edge[0]][edge[1]]['ibd_sum'] / 6600)
                        else:
                            dataset = copy.deepcopy(self.datasets[dataset_name][s])
                        h = Heuristics(dataset)
                        results = h.run_heuristic(heuristic)
                        if not os.path.isdir(log_dir):
                            os.mkdir(log_dir)
                        with open(log_dir + f'/results.json', 'w') as f:
                            results['model_name'] = heuristic
                            json.dump(results, f)




        if len(self.community_detection_models):
            for dataset_name in self.datasets.keys():
                for cd_model in tqdm(self.community_detection_models, desc=f"Running community detection for {dataset_name}"):
                    for s in range(self.running_params['num_splits']):
                        log_dir = self.running_params['log_dir'] + '/' + dataset_name + ('_log' if self.running_params['log_ibd'] else '') + '/' + cd_model + '_' + f'split_{s}'
                        if self.running_params['log_ibd']:
                            dataset = copy.deepcopy(self.datasets[dataset_name][s])
                            # edge_weights = nx.get_edge_attributes(dataset.nx_graph, 'ibd_sum')
                            for edge in dataset.nx_graph.edges:
                                dataset.nx_graph[edge[0]][edge[1]]['ibd_sum'] = -np.log2(dataset.nx_graph[edge[0]][edge[1]]['ibd_sum'] / 6600)
                        else:
                            dataset = copy.deepcopy(self.datasets[dataset_name][s])
                        cd = CommunityDetection(dataset)
                        if self.running_params['mask_size'] is not None:
                            masking=True
                        else:
                            masking=False
                        results = cd.run_community_detection(cd_model, masking)
                        if not os.path.isdir(log_dir):
                            os.mkdir(log_dir)
                        with open(log_dir + f'/results.json', 'w') as f:
                            results['model_name'] = heuristic
                            json.dump(results, f)


