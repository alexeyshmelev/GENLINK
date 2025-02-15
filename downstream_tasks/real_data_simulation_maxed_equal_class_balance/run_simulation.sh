#bash
cd ../..
JSON_PATH=./downstream_tasks/real_data_simulation_maxed_equal_class_balance
python utils/simulator.py --source_folder ~/vu_and_cr --destination_folder ~/genlink_simulated_real_data_maxed_equal_class_balance --simulation_params ${JSON_PATH}/simulation_params.json \
--random_state 42