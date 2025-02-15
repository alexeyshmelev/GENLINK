#bash
cd ../..
JSON_PATH=./downstream_tasks/real_data_simulation_class_balance_interpolation
python utils/simulator.py --source_folder ~/vu_and_cr --destination_folder ~/genlink_real_data_simulation_class_balance_interpolation --simulation_params ${JSON_PATH}/simulation_params.json \
--random_state 42