#bash
cd ../..
JSON_PATH=./downstream_tasks/real_data_no_masks
python utils/pipeline.py --data_folder /disk/10tb/home/shmelev/New_CR_2025/june_es_2026_benchmark_with_pca_fixed --hardware "1" --model_list ${JSON_PATH}/model_list.json --running_params ${JSON_PATH}/running_params.json \
--models_per_gpu 2
