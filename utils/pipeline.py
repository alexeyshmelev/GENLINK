import argparse
from os import listdir
from os.path import isfile, join
import sys
import models
import inspect
import numpy as np
import json
from runner import Runner
from multiprocessing import freeze_support



if __name__ == '__main__':
    # freeze_support()

    parser = argparse.ArgumentParser(description='GENLINK')

    parser.add_argument('--data_folder', type=str, help='Folder that contain initial graphs')
    parser.add_argument('--save_checkpoint_folder', type=str, help='Folder to store resulted checkpoints')
    parser.add_argument('--summary_folder', type=str, help='Folder to store summary of training')
    parser.add_argument('--model_list', type=str, help='Path to .json file that contain list of models to train')
    parser.add_argument('--hardware', type=str, help='Specifies the hardware to run training')
    parser.add_argument('--models_per_gpu', type=int, help='Specifies the numbers of models that will train in parallel on each gpu')
    parser.add_argument('--running_params', type=str, help='Path to .json file that contain parameters for training')
    parser.add_argument('--fp16', action='store_true', help='Whether to use low precision or not')

    args = parser.parse_args()

    data_files = [join(args.data_folder, f) for f in listdir(args.data_folder) if isfile(join(args.data_folder, f))]
    for f in data_files:
        print(f'Observed data file: {f}')

    if args.hardware == 'cpu':
        device = [args.hardware]
        gpu_map = None
    else:
        device = list(map(lambda x: int(x), args.hardware.split('|')))
        gpu_map = {idx:gpu for idx, gpu in enumerate(device)}

    NON_AI_MODELS = ['MaxEdgeCount', 'MaxEdgeCountPerClassSize', 'MaxIbdSum', 'MaxIbdSumPerClassSize', 'LongestIbd', 'MaxSegmentCount', 'AgglomerativeClustering', 'GirvanNewmann', 'LabelPropagation', 'MultiRankWalk', 'RelationalNeighborClassifier', 'RidgeRegression', 'SpectralClustering']
    all_models = list(filter(lambda x: x[:2] == 'GL', [cls_name for cls_name, cls_obj in inspect.getmembers(sys.modules['models']) if inspect.isclass(cls_obj)])) + NON_AI_MODELS

    # load model names
    with open(args.model_list, 'r') as f:
        model_dict = json.load(f)
        feature_types = set()

        feature_typed_gnn = dict()

        for k in model_dict.keys():
            if k != 'gnn':
                for m in model_dict[k]:
                    if m not in all_models:
                        raise Exception('There is no such model defined in GENLINK.')
            else:
                for m in model_dict[k].keys():
                    for feature_type in model_dict[k][m]:
                        feature_types.add(feature_type)
                        if m not in all_models:
                            raise Exception('There is no such NN defined in GENLINK.')
                        feature_typed_gnn[tuple([m, feature_type])] = feature_type

        gnn_models = feature_typed_gnn
        heuristic_models = model_dict['heuristic']
        community_detection_models = model_dict['community_detection']

    # load training params
    with open(args.running_params, 'r') as f:
        running_params = json.load(f)

    runner = Runner(data_files, feature_types, running_params, gnn_models, heuristic_models, community_detection_models, device, gpu_map, args.models_per_gpu)
    runner.run()