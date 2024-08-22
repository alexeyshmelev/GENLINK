#bash
cd ../..
JSON_PATH=./downstream_tasks/real_data_mask_75_test
python utils/pipeline.py --data_folder ~/vu_only --hardware "0" --model_list ${JSON_PATH}/model_list.json --running_params ${JSON_PATH}/running_params.json \
--models_per_gpu 1