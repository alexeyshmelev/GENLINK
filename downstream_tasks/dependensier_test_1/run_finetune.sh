#bash
cd ../..
JSON_PATH=./downstream_tasks/dependensier_test_1
python utils/pipeline.py --data_folder /disk/10tb/home/shmelev/vu_only --hardware "0|1" --model_list ${JSON_PATH}/model_list.json --running_params ${JSON_PATH}/running_params.json \
--models_per_gpu 2