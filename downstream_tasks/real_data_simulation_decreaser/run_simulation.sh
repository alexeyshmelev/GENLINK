#bash
cd ../..
JSON_PATH=./downstream_tasks/real_data_simulation_decreaser
python utils/simulator.py --source_folder ~/vu_only --destination_folder ~/real_data_simulation_decreaser --simulation_params ${JSON_PATH}/simulation_params.json \
--random_state 42