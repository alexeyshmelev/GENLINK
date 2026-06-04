#bash
cd ../..
JSON_PATH=./downstream_tasks/real_data_mask_50
python utils/pipeline.py --data_folder /home/jovyan/genlink_real_data --hardware "0|1|2|3|4|5|6" --model_list ${JSON_PATH}/model_list.json --running_params ${JSON_PATH}/running_params.json \
--models_per_gpu 1
