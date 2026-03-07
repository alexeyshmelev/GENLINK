#bash
cd ../..
JSON_PATH=./downstream_tasks/real_data_no_masks
python utils/pipeline.py --data_folder /disk/10tb/home/shmelev/New_CR_2025/CE_no_masks_final --hardware "0|1" --model_list ${JSON_PATH}/model_list.json --running_params ${JSON_PATH}/running_params.json \
--models_per_gpu 3
