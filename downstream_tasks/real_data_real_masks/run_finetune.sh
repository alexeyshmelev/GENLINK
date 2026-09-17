#!/usr/bin/env bash
cd ../..

JSON_PATH=./downstream_tasks/real_data_real_masks
python utils/pipeline.py --data_folder /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/final_datasets_2/2nd_degree/all_unlabeled_with_pca/ --hardware "4|5" --model_list ${JSON_PATH}/model_list.json --running_params ${JSON_PATH}/running_params.json --models_per_gpu 1