#bash
cd ../..
JSON_PATH=./downstream_tasks/real_data_simulation_equal_probs
python utils/pipeline.py --data_folder ~/genlink_real_data_alike_simulated_equal_probs --hardware "0|1" --model_list ${JSON_PATH}/model_list.json --running_params ${JSON_PATH}/running_params.json \
--models_per_gpu 2