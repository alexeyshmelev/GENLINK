#bash
cd ../..
JSON_PATH=./downstream_tasks/real_data_simulation_diagonal_changer
<<<<<<< HEAD
python utils/pipeline.py --data_folder ~/real_data_simulation_diagonal_changer --hardware "0|1" --model_list ${JSON_PATH}/model_list.json --running_params ${JSON_PATH}/running_params.json \
--models_per_gpu 2
=======
python utils/pipeline.py --data_folder ~/real_data_simulation_diagonal_changer --hardware "1" --model_list ${JSON_PATH}/model_list.json --running_params ${JSON_PATH}/running_params.json \
--models_per_gpu 3
>>>>>>> 9803d4957c156512176c103f42f8090b72f18a80
