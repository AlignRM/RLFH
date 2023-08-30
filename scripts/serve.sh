source ../miniconda3/bin/activate rlfh

set -x

MODEL="${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}"
cd ${WORK_DIR:-"."}/runs || exit
serve start --http-host ${SERVE_HOST:-"0.0.0.0"} --http-port ${SERVE_PORT:-"8000"}
serve run \
    ${BLOCKING:+--blocking} \
    serve:build_app \
    model="$MODEL" \
    tensor-parallel-size="${TENSOR_PARALLEL_SIZE:-"1"}" \
    served_model_name="${SERVED_MODEL_NAME:-$MODEL}" \
    accelerator="GPU"
