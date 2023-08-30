cd "${WORK_DIR}/agi-rlfh/" || exit
source ${WORK_DIR}/miniconda3/bin/activate rlfh

set -x
VERIFICATION_MAP_DEFAULT="{'Correct':0.3,'Hedged correct':0.1,'Vague':-0.8,'Hedged wrong':-0.9,'Wrong':-1.2}"
export VERIFICATION_MAP="${VERIFICATION_MAP:-"$VERIFICATION_MAP_DEFAULT"}"
INFORMATIVE_MAP_DEFAULT="{'5':1.3,'4':1.2,'3':1.1,'2':1,'1':-0.1}"
export INFORMATIVE_MAP="${INFORMATIVE_MAP:-"$INFORMATIVE_MAP_DEFAULT"}"
export SAVE_PATH="${SAVE_PATH:-"./models/Qwen2.5-7B-Instruct"}"
export CKPT_PATH="${CKPT_PATH:-$SAVE_PATH}"
python runs/train_ray.py \
  --ref_num_nodes="${REF_NUM_NODES:-"1"}" \
  --ref_num_gpus_per_node="${REF_NUM_GPUS_PER_NODE:-"1"}" \
  --actor_num_nodes="${ACTOR_NUM_NODES:-"1"}" \
  --actor_num_gpus_per_node="${ACTOR_NUM_GPUS_PER_NODE:-"1"}" \
  --critic_num_nodes="${CRITIC_NUM_NODES:-"1"}" \
  --critic_num_gpus_per_node="${CRITIC_NUM_GPUS_PER_NODE:-"1"}" \
  --vllm_num_engines="${VLLM_NUM_ENGINES:-"1"}" \
  --vllm_tensor_parallel_size="${VLLM_TENSOR_PARALLEL_SIZE:-"1"}" \
  --vllm_gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-"0.9"}" \
  ${COLOCATE_ACTOR_REF:+--colocate_actor_ref} \
  ${COLOCATE_ALL_MODELS:+--colocate_all_models} \
  ${COLOCATE_ALL_MODELS:+--vllm_enable_sleep} \
  ${COLOCATE_ALL_MODELS:+--vllm_sync_with_ray} \
  ${ENFORCE_EAGER:+--enforce_eager} \
  ${REF_REWARD_OFFLOAD:+--ref_reward_offload} \
  ${ADAM_OFFLOAD:+--adam_offload} \
  ${OFFLOAD:+--offload} \
  ${ENABLE_PREFIX_CACHEING:+--enable_prefix_caching} \
  --flash_attn \
  --packing_samples \
  --gradient_checkpointing \
  --zero_stage 3 \
  --bf16 \
  --vllm_sync_backend="${VLLM_SYNC_BACKEND:-"nccl"}" \
  --pretrain="${MODEL_PATH:-"meta-llama/Llama-3.1-8B-Instruct"}" \
  --save_path=$SAVE_PATH \
  --ckpt_path=$CKPT_PATH \
  --save_steps="${SAVE_STEP:-"5"}" \
  --max_ckpt_num="${MAX_CKPT_NUM:-"5"}" \
  --save_value_network \
  --save_hf_ckpt \
  --load_checkpoint \
  --use_tensorboard=$SAVE_PATH \
  ${USE_WANDB:+--use_wandb=$USE_WANDB} \
  ${WANDB_GROUP:+--wandb_group=$WANDB_GROUP} \
  ${WANDB_PROJECT:+--wandb_project=$WANDB_PROJECT} \
  ${WANDB_RUN_NAME:+--wandb_run_name=$WANDB_RUN_NAME} \
  --micro_train_batch_size="${MICRO_TRAIN_BATCH_SIZE:-"1"}" \
  --train_batch_size="${TRAIN_BATCH_SIZE:-"1"}" \
  --micro_rollout_batch_size="${MICRO_ROLLOUT_BATCH_SIZE:-"2"}" \
  --rollout_batch_size="${ROLLOUT_BATCH_SIZE:-"1"}" \
  --n_samples_per_prompt="${N_SAMPLES_PER_PROMPT:-"1"}" \
  --max_samples="${SAMPLES:-"100000"}" \
  --max_epochs="${EPOCHS:-"1"}" \
  --num_episodes="${EPISODES:-"1"}" \
  --prompt_max_len="${PROMPT_MAX_LEN:-"2048"}" \
  --generate_max_len="${GENERATE_MAX_LEN:-"1024"}" \
  --actor_learning_rate="${ACTOR_LR:-"5e-7"}" \
  --critic_learning_rate="${CRITIC_LR:-"9e-6"}" \
  --init_kl_coef="${KL_COEF:-"1e-2"}" \
  --freezing_actor_steps="${FREEZING_ACTOR_STEPS:-"0"}" \
  --advantage_estimator="${ADVANTAGE_ESTIMATOR:-"gae"}" \
  --lambd="${LAMBD:-"0.95"}" \
  --use_kl_estimator_k3 \
  --prompt_data="${PROMPT_DATA:-"data/hotpot_qa_train_1w.jsonl"}" \
  --reward_config.verification_map="$VERIFICATION_MAP" \
  --reward_config.informative_map="$INFORMATIVE_MAP" \
  --reward_config.truth_weight="${TRUTH_WEIGHT:-"1"}" \
  --reward_config.info_weight="${INFO_WEIGHT:-"1"}" \
  --reward_config.epsilon="${EPSILON:-"-0.9"}" \
  --reward_config.mu="${MU:-"1"}" \
  --reward_config.granularity="${GRANULARITY:-"token"}" \
  --reward_config.annotator_config.retrieval_config.max_secs="${MAX_SECS:-"3"}" \
  --reward_config.annotator_config.retrieval_config.max_words_per_sec="${MAX_WORDS_PER_SEC:-"64"}" \
  ${ANNOTATE_MODEL:+--reward_config.annotator_config.completion_config.model=$ANNOTATE_MODEL} \
  ${MODEL_URL:+--reward_config.annotator_config.completion_config.model_url=$MODEL_URL} \
  --reward_clip_range "${REWARD_CLIP_RANGE_LOWER:-"-10"}" "${REWARD_CLIP_RANGE_UPPER:-"10"}" \
  --perf 2>&1
