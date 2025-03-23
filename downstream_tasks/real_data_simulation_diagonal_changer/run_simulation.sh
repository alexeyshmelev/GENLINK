#bash
cd ../..
JSON_PATH=./downstream_tasks/real_data_simulation_diagonal_changer
python utils/simulator.py --source_folder ~/vu_only --destination_folder ~/real_data_simulation_diagonal_changer --simulation_params ${JSON_PATH}/simulation_params.json \
--random_state 42