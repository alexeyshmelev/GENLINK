#bash
cd ../..
JSON_PATH=./downstream_tasks/real_data_simulation
python utils/simulator.py --source_folder ~/genlink_real_data --destination_folder ~/genlink_real_data_alike_simulated --simulation_params ${JSON_PATH}/simulation_params.json \
--random_state 42