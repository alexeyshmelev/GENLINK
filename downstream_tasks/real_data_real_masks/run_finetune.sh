#!/usr/bin/env bash
cd ../..

JSON_PATH=./downstream_tasks/real_data_real_masks
python utils/pipeline.py --data_folder /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/final_datasets_2_G4_FA_unlabeled_20261012/FA/without_pca/ --hardware "1|2|3|4|5|6" --model_list ${JSON_PATH}/model_list.json --running_params ${JSON_PATH}/running_params.json --models_per_gpu 1