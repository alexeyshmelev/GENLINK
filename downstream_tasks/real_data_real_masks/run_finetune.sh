#!/usr/bin/env bash
PROCESS_NAME="torchrun --master_addr 127.0.0.1 --nproc_per_node 5 segmentation/train.py --config segmentation/configs/moderngena_base_gpt.json"

cd ../..
JSON_PATH=./downstream_tasks/real_data_real_masks
python utils/pipeline.py --data_folder /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/final_datasets/2nd_degree/unlabeled_pca_needed --hardware "0|1|2|3|4|5|6" --model_list ${JSON_PATH}/model_list.json --running_params ${JSON_PATH}/running_params.json \
--models_per_gpu 2
