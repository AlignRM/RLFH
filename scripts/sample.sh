set -x
export PYTHONPATH="${PYTHONPATH}:./trlx"
cd ${WORKSPACE_PATH:-"./"} || exit
N_GPUS=${KUBERNETES_CONTAINER_RESOURCE_GPU:-$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)}
export NUM_PROCESSES="${NUM_PROCESSES:-"$(expr $N_GPUS \* $WORLD_SIZE)"}"
export PYTHONPATH=${PYTHONPATH}:./:./repos/trlx
export VLLM_WORKER_MULTIPROC_METHOD=spawn

source ../miniconda3/bin/activate rlfh
python runs/sample.py \
  ${DATA_PATH:+--data_path="${DATA_PATH}"} \
  --dataset_config.dataset_path="${DATASET_PATH:-"hotpotqa/hotpot_qa"}" \
  --dataset_config.dataset_name="$DATASET_NAME" \
  --dataset_config.template_name="$TEMPLATE_NAME" \
  --dataset_config.test_size="${TEST_SIZE:-""}" \
  --dataset_config.n_questions="${N_QUESTIONS:-""}" \
  --dataset_config.db_path="${DB_PATH:-"../cache/enwiki-20230401.db"}" \
  --dataset_config.split="${SPLIT:-""}" \
  --model_path="${MODEL_PATH:-"meta-llama/Llama-3.1-8B-Instruct"}" \
  --output_path="${OUTPUT_PATH:-"./outputs/qa-train.jsonl"}" \
  --ans_args.n="${N_ANSWERS:-"1"}" \
  --ans_args.top_p="${TOP_P:-"1"}" \
  --ans_args.temperature="${TEMPERATURE:-"0."}" \
  --tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-"1"}" \
  --num_instances="${NUM_INSTANCES:-"1"}" \
  --batch_size="${BATCH_SIZE:-"8"}"
