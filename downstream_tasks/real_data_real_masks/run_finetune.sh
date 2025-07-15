#bash
# export NETWORKX_BACKEND_PRIORITY_ALGOS=cugraph
# export NETWORKX_BACKEND_PRIORITY_GENERATORS=cugraph
# export NETWORKX_FALLBACK_TO_NX=True
# export NX_CUGRAPH_AUTOCONFIG=True
cd ../..
JSON_PATH=./downstream_tasks/real_data_real_masks
python utils/pipeline.py --data_folder /home/jovyan/shmelev/CR_agreed --hardware "0|1|2|3|4|5|6|7" --model_list ${JSON_PATH}/model_list.json --running_params ${JSON_PATH}/running_params.json \
--models_per_gpu 1
