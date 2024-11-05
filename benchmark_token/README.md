# Triton-Inference-Server (TensorRT-LLM)

## Launching triton server

- Triton-Inference-Server [Tutorial](https://squeezebits.atlassian.net/l/cp/jfw1ver0)
- With `--update-config`, overwrites `config.pbtxt` each models (e.g., ensemble, tensorrt_llm, ...)
- [model configuration](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/model_config.md)
- For triton model repo, use `triton_llm_model` for decoder model or use `triton_draft_target_model` for speculative decoding.

```shell
MODEL_DIR=/hf/model/dir
TRITON_MODEL_DIR=/triton/model/dir
TRTLLM_ENGINE_DIR=/trtllm/engine/dir
TRTLLM_BACKEND_REPO_DIR=/tensorrtllm_backend # from https://github.com/triton-inference-server/tensorrtllm_backend

BATCH_SIZE=256
LOGFILE_PATH=/log/file/path

python3 launch_server.py tritonserver \
    --tensorrtllm-backend-repo $TRTLLM_BACKEND_REPO_DIR \
    --model-repo $TRITON_MODEL_DIR \
    --world-size 1 \
    --log --log-file $LOGFILE_PATH \
    --triton-backend tensorrtllm \
    --triton-max-batch-size $BATCH_SIZE \
    --decoupled-mode \
    --gpt-model-path $TRTLLM_ENGINE_DIR \
    --gpt-model-type inflight_fused_batching \
    --tokenizer-dir $MODEL_DIR \
    --update-config
```

## Benchmarking

```shell
RESULT_FILE_PATH=result.json
QPS=inf

python3 benchmark_serving_token.py \
    --backend tensorrt-llm \
    --endpoint /v2/models/tensorrt_llm/generate_stream \
    --tokenizer $MODEL_DIR \
    --request-rate $QPS \
    --dataset random --input-len 128 --output-len 128 --num-requests 256 \
    --ignore-eos \
    --save-result $RESULT_FILE_PATH
```

## Use Draft-Target Models for Speculative Decoding

```shell
MODEL_DIR=/hf/model/dir
TRITON_MODEL_DIR=triton_draft_target_model
TARGET_MODEL_DIR=/trtllm/target/model/dir
DRAFT_MODEL_DIR=/trtllm/draft/model/dir

# launch server
python3 launch_server.py tritonserver --tensorrtllm-backend-repo tensorrtllm_backend --model-repo $TRITON_MODEL_DIR --world-size 1 --update-config --triton-backend tensorrtllm --triton-max-batch-size $BATCH_SIZE --decoupled-mode --gpt-model-path $TARGET_MODEL_DIR --draft-model-path $DRAFT_MODEL_DIR --tokenizer-dir $MODEL_DIR --use-speculative-decoding  --gpu-device-ids 0 --draft-gpu-device-ids 1 --enable-kv-cache-reuse

# benchmark
python3 benchmark_serving_token.py --backend tensorrt-llm --tokenizer $MODEL_DIR --endpoint /v2/models/tensorrt_llm_bls/generate_stream --dataset random --input-len 128 --output-len 128 --num-requests 256 --num-draft-tokens $NUM_DRAFT_TOKENS
```

# vLLM

## Launching vLLM OpenAI API Server

- Arguments are same with `vllm.entrypoints.openai.api_server`.
- `--skip-tokenizer-init` is optional to disable decoding tokens.

```shell
MODEL_DIR=/hf/model/dir
BATCH_SIZE=256

python3 launch_server.py vllm-openai \
    --model $MODEL_DIR \
    --max-num-seqs $BATCH_SIZE \
    --disable-log-requests \
    --disable-log-stats \
    --uvicorn-log-level warning \
    --skip-tokenizer-init
```

## Benchmarking

```shell
RESULT_FILE_PATH=result.json
QPS=inf

python3 benchmark_serving_token.py \
    --backend vllm-openai \
    --endpoint /v1/completions \
    --tokenizer $MODEL_DIR \
    --request-rate $QPS \
    --dataset random --input-len 128 --output-len 128 --num-requests 256 \
    --ignore-eos \
    --save-result $RESULT_FILE_PATH
```

# Quitting the launched server

```shell
# for tritonserver
python3 launch_server.py --kill tritonserver

# for vllm-openai
python3 launch_server.py --kill vllm-openai
```

# Use Case

- It is possible to run benchmark script on single terminal

```shell
MODEL_DIR=/hf/model/dir
TRITON_MODEL_DIR=/triton/model/dir
TRTLLM_CKPT_DIR=/trtllm/ckpt/dir
TRTLLM_ENGINE_DIR=/trtllm/engine/dir
TRTLLM_BACKEND_REPO_DIR=/tensorrtllm_backend # from https://github.com/triton-inference-server/tensorrtllm_backend

NUM_REQUESTS=1000
INPUT_LEN=256
OUTPUT_LEN=256

# tensorrt-llm benchmark
for parallel in 4,1 2,2 1,4 2,1
do
    IFS=","; set -- $parallel
    TP=$1
    PP=$2
    TP_PP=$((TP * PP))


    # convert checkpoint
    python3 /tensorrtllm_backend/tensorrt_llm/examples/llama/convert_checkpoint.py \
        --model_dir $MODEL_DIR \
        --output_dir $TRTLLM_CKPT_DIR \
        --dtype bfloat16 \
        --tp_size $TP --pp_size $PP

    for BATCH_SIZE in 256
    do
        # build engine
        trtllm-build \
            --checkpoint_dir $TRTLLM_CKPT_DIR \
            --output_dir $TRTLLM_ENGINE_DIR \
            --max_batch_size $BATCH_SIZE \
            --kv_cache_type paged \
            --log_level info

        for QPS in inf 16 14 12 10 8 7 6 5 4 3 2 1
        do
            LOGFILE_PATH=/log/file/path
            # launch tritonserver
            python3 launch_server.py \
                tritonserver \
                --tensorrtllm-backend-repo $TRTLLM_BACKEND_REPO_DIR \
                --model-repo $TRITON_MODEL_DIR \
                --world-size $TP_PP \
                --log --log-file $LOGFILE_PATH \
                --triton-backend tensorrtllm \
                --triton-max-batch-size $BATCH_SIZE \
                --decoupled-mode \
                --gpt-model-path $TRTLLM_ENGINE_DIR \
                --gpt-model-type inflight_fused_batching \
                --tokenizer-dir $MODEL_DIR \
                --preprocessing-instance-count 1 \
                --postprocessing-instance-count 1 \
                --update-config
            
            # run benchmark
            RESULT_JSON_PATH=/result/json/file/path
            python3 benchmark_serving_token.py \
                --backend tensorrt-llm \
                --endpoint /v2/models/tensorrt_llm/generate_stream \
                --tokenizer $MODEL_DIR \
                --request-rate $QPS \
                --dataset random --input-len $INPUT_LEN --output-len $OUTPUT_LEN --num-requests $NUM_REQUESTS \
                --ignore-eos \
                --save-result $RESULT_JSON_PATH

            # kill tritonserver
            python3 launch_server.py --kill tritonserver
            sleep 60
        done

        rm -rf $ENGINE_PATH
    done

    rm -rf $CKPT_PATH
done


# vllm benchmark
conda activate

for parallel in 2,2 1,4 4,1 2,1
do
    IFS=","; set -- $parallel
    TP=$1
    PP=$2
    TP_PP=$((TP * PP))

    for BATCH_SIZE in 256
    do
        for QPS in inf 16 14 12 10 8 7 6 5 4 3 2 1
        do
            # launch vllm-openai server
            python3 launch_server.py \
                vllm-openai \
                --model $MODEL_DIR \
                --max-num-seqs $BATCH_SIZE \
                --disable-log-requests \
                --disable-log-stats \
                --uvicorn-log-level warning \
                -tp $TP -pp $PP
                # --skip-tokenizer-init \
            
            # run benchmark
            RESULT_JSON_PATH=/result/json/file/path
            python3 benchmark_serving_token.py \
                --backend vllm \
                --endpoint /v1/completions \
                --tokenizer $MODEL_DIR \
                --request-rate $QPS \
                --dataset random --input-len $INPUT_LEN --output-len $OUTPUT_LEN --num-requests $NUM_REQUESTS \
                --ignore-eos \
                --save-result $RESULT_JSON_PATH

            # kill vllm-openai server
            python3 launch_server.py --kill vllm-openai
            sleep 60
        done
    done
done
```