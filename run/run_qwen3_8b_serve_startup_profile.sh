#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspaces/tensorrt_llm}
WORKSPACE=${WORKSPACE:-${ROOT_DIR}/tmp/qwen3_8b_serve_startup_profile}
MODELS_DIR=${MODELS_DIR:-${ROOT_DIR}/llm-models}
MODEL_DIR=${MODEL_DIR:-${MODELS_DIR}/Qwen3/Qwen3-8B}

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-128}
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-1024}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
KV_CACHE_FREE_GPU_MEM_FRACTION=${KV_CACHE_FREE_GPU_MEM_FRACTION:-0.6}

CUDA_GRAPH_BATCH_SIZES=${CUDA_GRAPH_BATCH_SIZES:-1,2,4,8,16,32,64,96,128}
PIECEWISE_CAPTURE_NUM_TOKENS=${PIECEWISE_CAPTURE_NUM_TOKENS:-1,2,4,8,16,32,64,128,256,512,1024}
ENABLE_INDUCTOR=${ENABLE_INDUCTOR:-false}
ENABLE_USERBUFFERS=${ENABLE_USERBUFFERS:-true}
MAX_NUM_STREAMS=${MAX_NUM_STREAMS:-3}
ENABLE_LAYERWISE_NVTX_MARKER=${ENABLE_LAYERWISE_NVTX_MARKER:-false}
PROFILE_WITH_NSYS=${PROFILE_WITH_NSYS:-0}

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export TLLM_LOG_LEVEL=${TLLM_LOG_LEVEL:-INFO}
export TLLM_NVTX_DEBUG=${TLLM_NVTX_DEBUG:-1}

mkdir -p "${WORKSPACE}"

CONFIG_PATH=${CONFIG_PATH:-${WORKSPACE}/qwen3_8b_piecewise_compile.yml}
LOG_PATH=${LOG_PATH:-${WORKSPACE}/trtllm-serve-startup.log}
TIMELINE_PATH=${TIMELINE_PATH:-${WORKSPACE}/startup-timeline.jsonl}
SUMMARY_PATH=${SUMMARY_PATH:-${WORKSPACE}/startup-summary.txt}
PARSER_PATH=${PARSER_PATH:-${ROOT_DIR}/run/parse_trtllm_serve_startup.py}
NSYS_OUTPUT=${NSYS_OUTPUT:-${WORKSPACE}/qwen3_8b_serve_startup}
COMPARE_PATH=${COMPARE_PATH:-${ROOT_DIR}/run/compare_startup_nvtx.py}
NSYS_REP_PATH=${NSYS_REP_PATH:-${NSYS_OUTPUT}.nsys-rep}

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "MODEL_DIR does not exist: ${MODEL_DIR}" >&2
    echo "Override MODEL_DIR=/path/to/Qwen3-8B or MODELS_DIR=/path/to/llm-models." >&2
    exit 1
fi

yaml_list() {
    local value=${1// /}
    if [[ "${value}" == \[*\] ]]; then
        printf "%s" "${value}"
    else
        printf "[%s]" "${value}"
    fi
}

CUDA_GRAPH_BATCH_SIZES_YAML=$(yaml_list "${CUDA_GRAPH_BATCH_SIZES}")
PIECEWISE_CAPTURE_NUM_TOKENS_YAML=$(yaml_list "${PIECEWISE_CAPTURE_NUM_TOKENS}")

cat >"${CONFIG_PATH}" <<EOF
backend: pytorch
print_iter_log: true
enable_iter_perf_stats: true
enable_layerwise_nvtx_marker: ${ENABLE_LAYERWISE_NVTX_MARKER}
max_batch_size: ${MAX_BATCH_SIZE}
max_num_tokens: ${MAX_NUM_TOKENS}
max_seq_len: ${MAX_SEQ_LEN}
tensor_parallel_size: ${TP_SIZE}
pipeline_parallel_size: ${PP_SIZE}
moe_expert_parallel_size: ${EP_SIZE}
kv_cache_config:
  free_gpu_memory_fraction: ${KV_CACHE_FREE_GPU_MEM_FRACTION}
cuda_graph_config:
  enable_padding: true
  batch_sizes: ${CUDA_GRAPH_BATCH_SIZES_YAML}
torch_compile_config:
  enable_fullgraph: true
  enable_inductor: ${ENABLE_INDUCTOR}
  enable_piecewise_cuda_graph: true
  enable_userbuffers: ${ENABLE_USERBUFFERS}
  max_num_streams: ${MAX_NUM_STREAMS}
  capture_num_tokens: ${PIECEWISE_CAPTURE_NUM_TOKENS_YAML}
EOF

SERVE_CMD=(
    trtllm-serve "${MODEL_DIR}"
    --host "${HOST}"
    --port "${PORT}"
    --backend pytorch
    --trust_remote_code
    --config "${CONFIG_PATH}"
)

if [[ "${PROFILE_WITH_NSYS}" == "1" ]]; then
    SERVE_CMD=(
        nsys profile
        --output "${NSYS_OUTPUT}"
        --force-overwrite true
        --trace cuda,nvtx,mpi,python-gil
        --cuda-graph-trace node
        --trace-fork-before-exec true
        "${SERVE_CMD[@]}"
    )
fi

echo "Config: ${CONFIG_PATH}"
echo "Log: ${LOG_PATH}"
echo "Timeline: ${TIMELINE_PATH}"
echo "Summary: ${SUMMARY_PATH}"
if [[ "${PROFILE_WITH_NSYS}" == "1" ]]; then
    echo "Nsight Systems output prefix: ${NSYS_OUTPUT}"
fi
echo "Starting: ${SERVE_CMD[*]}"

"${SERVE_CMD[@]}" 2>&1 | python3 "${PARSER_PATH}" "${TIMELINE_PATH}" "${SUMMARY_PATH}" "${LOG_PATH}" &

wait_for_port() {
    while ! nc -z "$HOST" "$PORT"; do
        sleep 1
    done
}

# Wait for the server to start up and be ready to accept connections.
sleep 100
wait_for_port

pkill -f "trtllm-serve" || true

if [[ "${PROFILE_WITH_NSYS}" == "1" && -f "${NSYS_REP_PATH}" ]]; then
    echo ""
    echo "Running NVTX vs log comparison for ${NSYS_REP_PATH} ..."
    python3 "${COMPARE_PATH}" "${NSYS_REP_PATH}" "${SUMMARY_PATH}"
fi
