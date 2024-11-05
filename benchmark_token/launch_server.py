import warnings
import time
import re
import subprocess
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import requests

def update_configuration_file(target: str, args: Namespace):
    config_file_path = Path(args.model_repo) / target / "config.pbtxt"

    if os.path.exists(config_file_path):
        with open(config_file_path, "r") as f:
            lines = f.readlines()
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if "parameters" in line:
                key_line = lines[i + 1].strip()
                m = re.match(r'key:\s*"(.*?)"', key_line)
                if m:
                    key = m.group(1)
                    for target_key, value in vars(args).items():
                        if value is not None and (key == target_key or key in ["gpu_device_ids"]):
                            if key != target_key and target == "tensorrt_llm_draft":
                                value = args.draft_gpu_device_ids
                            if key == "gpt_model_path" and target == "tensorrt_llm_draft":
                                value = args.draft_model_path
                            new_lines.append(lines[i]) # parameters: {
                            new_lines.append(lines[i+1]) # key: "..."
                            new_lines.append(lines[i+2]) # value: {
                            if isinstance(value, bool):
                                if value:
                                    value = "true"
                                else:
                                    value= "false"
                            new_lines.append(f'    string_value: "{value}"\n')
                            i += 4
                            break
                    else:
                        new_lines.append(line)
                        i += 1
                else:
                    new_lines.append(line)
                    i += 1

            elif (target == "tensorrt_llm" or target == "tensorrt_llm_draft") and "backend:" in line:
                line = line[:line.rfind(':') + 1] + f' "{args.triton_backend}"\n'
                new_lines.append(line)
                i += 1
            elif "max_batch_size:" in line:
                line = line[:line.rfind(':') + 1] + f" {args.triton_max_batch_size}\n"
                new_lines.append(line)
                i += 1
            elif "preferred_batch_size:" in line:
                line = line[:line.rfind(':') + 1] + f" [ {args.triton_max_batch_size} ]\n"
                new_lines.append(line)
                i += 1
            elif "decoupled:" in line:
                val = args.decoupled_mode
                if args.use_speculative_decoding:
                    if not args.decoupled_mode:
                        raise ValueError("Decoupled Mode must be enabled for Speculative Decoding.")
                    if target == "tensorrt_llm" or target == "tensorrt_llm_draft":
                        val = False
                val_str = " true\n" if val else " false\n"
                line = line[:line.rfind(':') + 1] + val_str
                new_lines.append(line)
                i += 1
            elif "max_queue_delay_microseconds:" in line:
                line = line[:line.rfind(':') + 1] + f" {args.max_queue_delay_microseconds}\n"
                new_lines.append(line)
                i += 1
            elif "default_queue_policy:" in line:
                line = line[:line.rfind(':') + 1] + f" {args.max_queue_size} " + "}\n"
                new_lines.append(line)
                i += 1
            elif (target == "preprocessing" or target == "postprocessing" or target == "tensorrt_llm_bls") and "instance_group" in line:
                if target == "preprocessing":
                    count = args.preprocessing_instance_count
                elif target == "postprocessing":
                    count = args.postprocessing_instance_count
                else:
                    count = 1
                new_lines.append(lines[i]) # instance_group [
                new_lines.append(lines[i+1]) # {
                cnt_line = lines[i+2]
                cnt_line = cnt_line[:cnt_line.rfind(':') + 1] + f" {count}\n"
                new_lines.append(cnt_line)
                new_lines.append(lines[i+3]) # kind: KIND_CPU
                new_lines.append(lines[i+4]) # }
                new_lines.append(lines[i+5]) # ]
                i += 6
            else:
                new_lines.append(line)
                i += 1
        with open(config_file_path, "w") as f:
            f.writelines(new_lines)

def launch_server(args: Namespace):
    def wait_server(api_url: str, timeout=120) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(api_url)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        
        return False

    if args.backend == "tritonserver":
        launch_script = Path(args.tensorrtllm_backend_repo) / "scripts" / "launch_triton_server.py"
        model_api_url = f"http://{args.host}:{args.port}/v2/models/tensorrt_llm"
        cmdline = ["python3", launch_script,
                   "--model_repo", args.model_repo,
                   "--world_size", str(args.world_size)]
        if args.log:
            cmdline.append("--log")
            cmdline.append("--log-file")
            cmdline.append(args.log_file)
        if args.use_speculative_decoding:
            cmdline.append("--tensorrt_llm_model_name")
            cmdline.append(f"{args.tensorrt_llm_model_name},{args.tensorrt_llm_draft_model_name}")
            cmdline.append("--multi-model")
        
        if args.update_config:
            for target in ["ensemble", "preprocessing", "postprocessing", "tensorrt_llm", "tensorrt_llm_draft", "tensorrt_llm_bls"]:
                update_configuration_file(target, args)

    elif args.backend == "vllm-openai":
        model_api_url = f"http://{args.host}:{args.port}/v1/models"
        cmdline = ["python", "-m", "vllm.entrypoints.openai.api_server"]

        for key, value in vars(args).items():
            if key in ["backend", "log", "log_file", "allowed_origins", "allowed_methods", "allowed_headers"]:
                #  "['*']" causes an error
                continue
            arg_key = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmdline.append(arg_key)
            elif value is not None and value:
                cmdline.append(arg_key)
                cmdline.append(str(value))

    else:
        raise ValueError(f"unknown backend name: {args.backend}")

    subprocess.Popen(cmdline)
    
    if not wait_server(model_api_url):
        raise RuntimeError(f"failed to launch server ({args.backend})")

def quit_server(backend: str) -> None:
    if backend == "tritonserver":
        subprocess.run(["pkill", "-9", "mpirun"])
        subprocess.run(["pkill", "-9", "tritonserver"])

    elif backend == "vllm-openai":
        subprocess.run(["pkill", "-9", "pt_main_thread"])

    else:
        raise ValueError(f"unknown backend name: {backend}")

def parse_args() -> Namespace:
    parser = ArgumentParser("")
    parser.add_argument("--kill", type=str, choices=["vllm-openai", "tritonserver"])
    subparser = parser.add_subparsers(title="servers", dest="backend")
    
    vllm_parser = subparser.add_parser("vllm-openai", help="vllm server options")
    try:
        from vllm.entrypoints.openai.cli_args import make_arg_parser
        vllm_parser = make_arg_parser(vllm_parser)
    except ModuleNotFoundError:
        warnings.warn("vllm not found.")
    except Exception as e:
        raise RuntimeError(f"{e}")
    
    triton_parser = subparser.add_parser("tritonserver", help="triton server options")
    triton_parser.add_argument('--host', type=str)
    triton_parser.add_argument('--port', type=str, default="8000")
    triton_parser.add_argument('--tensorrtllm-backend-repo', type=str, required=True)
    triton_parser.add_argument('--model-repo', type=str, required=True)
    triton_parser.add_argument("--world-size", type=int, default=1)
    triton_parser.add_argument("--update-config", action="store_true", help="overwrites configuration file with passed arguments")
    triton_parser.add_argument("--log", action="store_true", help="log triton server stats into log_file")
    triton_parser.add_argument("--log-file", type=str, help="path to triton log gile", default="triton_log.txt")
    
    mandatory_config_group = triton_parser.add_argument_group("Mandatory parameters for triton server model")
    mandatory_config_group.add_argument("--triton-backend", type=str, default="tensorrtllm", choices=["tensorrtllm", "python"], help="Backend for the model (set to tensorrtllm or python).")
    mandatory_config_group.add_argument("--triton-max-batch-size", type=int, required=True, help="Maximum batch size for Triton model.")
    mandatory_config_group.add_argument("--decoupled-mode", action="store_true", help="Use decoupled mode for requests with stream tensor set to true.")
    mandatory_config_group.add_argument("--max-queue-delay-microseconds", type=int, default=0, help="Maximum queue delay in microseconds to improve request batching.")
    mandatory_config_group.add_argument("--max-queue-size", type=int, default=0, help="Maximum number of requests in TRT-LLM queue before rejecting new requests.")
    mandatory_config_group.add_argument("--gpt-model-path", type=str, required=True, help="Path to the engine for the model.")
    mandatory_config_group.add_argument("--gpt-model-type", type=str, default="inflight_fused_batching", choices=["inflight_fused_batching", "v1"], help="Batching strategy (default: inflight_fused_batching).")

    mandatory_config_group.add_argument("--tokenizer-dir", type=str, help="The path to the tokenizer for the model.", required=True)
    mandatory_config_group.add_argument("--preprocessing-instance-count", type=int, default=1, help="The number of instances of the model to run.")
    mandatory_config_group.add_argument("--postprocessing-instance-count", type=int, default=1, help="The number of instances of the model to run.")

    optional_config_group = triton_parser.add_argument_group("Optional parameters for triton server model")
    # General
    optional_config_group.add_argument("--encoder-model-path", type=str, help="Path to the fol`der with model configuration and encoder engine.")
    optional_config_group.add_argument("--max-attention-window-size", type=int, default=None, help="Maximum number of tokens in attention window (default: max_sequence_length).")
    optional_config_group.add_argument("--sink-token-length", type=int, help="Number of sink tokens to keep in the attention window.")
    optional_config_group.add_argument("--exclude-input-in-output", action="store_true", help="Only return completion tokens in response (default: False).")
    optional_config_group.add_argument("--cancellation-check-period-ms", type=int, help="Time for cancellation check thread to sleep (default: 100).")
    optional_config_group.add_argument("--stats-check-period-ms", type=int, help="Time for statistics reporting thread to sleep (default: 100).")
    optional_config_group.add_argument("--recv-poll-period-ms", type=int, help="Time for receiving thread in orchestrator mode to sleep (default: 0).")
    optional_config_group.add_argument("--iter-stats-max-iterations", type=int, help="Maximum number of iterations to keep statistics.")
    optional_config_group.add_argument("--request-stats-max-iterations", type=int, help="Maximum number of iterations for per-request statistics.")
    optional_config_group.add_argument("--normalize-log-probs", action="store_true", help="Normalize log probabilities (default: True).")
    optional_config_group.add_argument("--gpu-device-ids", type=str, help="Comma-separated list of GPU IDs for the model.")
    optional_config_group.add_argument("--participant-ids", type=str, help="Comma-separated list of MPI ranks for orchestrator mode.")
    optional_config_group.add_argument("--gpu-weights-percent", type=float, help="Percentage of weights on GPU for engine with weight_streaming on (default: 1.0).")
    # KV cache
    optional_config_group.add_argument("--max-tokens-in-paged-kv-cache", type=int, help="Maximum size of KV cache in tokens (default: infinite).")
    optional_config_group.add_argument("--kv-cache-free-gpu-mem-fraction", type=float, help="Fraction of GPU memory for KV cache (default: 0.9).")
    optional_config_group.add_argument("--kv-cache-host-memory-bytes", type=int, help="Enable offloading to host memory for given byte size.")
    optional_config_group.add_argument("--enable-kv-cache-reuse", action="store_true", help="Reuse previously computed KV cache values.")
    # LoRA cache
    optional_config_group.add_argument("--lora-cache-optimal-adapter-size", type=int, help="Optimal adapter size for LoRA cache pages (default: 8).")
    optional_config_group.add_argument("--lora-cache-max-adapter-size", type=int, help="Minimum size of LoRA cache page (default: 64).")
    optional_config_group.add_argument("--lora-cache-gpu-memory-fraction", type=float, help="Fraction of GPU memory for LoRA cache (default: 0.05).")
    optional_config_group.add_argument("--lora-cache-host-memory-bytes", type=int, help="Size of host LoRA cache in bytes (default: 1G).")
    # Decoding mode
    optional_config_group.add_argument("--max-beam-width", type=int, help="Beam width value for requests (default: 1).")
    optional_config_group.add_argument("--decoding-mode", type=str, choices=["top_k", "top_p", "top_k_top_p", "beam_search", "medusa"], help="Decoding mode for requests.")
    # Optimization
    optional_config_group.add_argument("--enable-chunked-context", action="store_true", help="Enable context chunking (default: False).")
    # Scheduling
    optional_config_group.add_argument("--batch-scheduler-policy", type=str, choices=["max_utilization", "guaranteed_no_evict"], default="guaranteed_no_evict", help="Policy for batch scheduling (default: guaranteed_no_evict).")
    # Speculative Decoding / Medusa
    optional_config_group.add_argument("--use-speculative-decoding", action="store_true", help="Use a draft and a target models.")
    optional_config_group.add_argument("--draft-model-path", type=str, help="Path to the engine for the draft model (Use --gpt-model-path for the target model).")
    optional_config_group.add_argument("--draft-gpu-device-ids", type=str, help="Comma-separated list of GPU Device IDs for a draft model (use --gpu-device-ids for the target model).", default="1")
    optional_config_group.add_argument("--use-draft-logits", action="store_true", help="Use draft logits.")
    optional_config_group.add_argument("--medusa-choices", type=str, help="Medusa choices tree in format e.g. '{0, 0, 0, 0, 1}'.")
    optional_config_group.add_argument("--tensorrt-llm-model-name", type=str, default="tensorrt_llm", help="Target Model Name")
    optional_config_group.add_argument("--tensorrt-llm-draft-model-name", type=str, default="tensorrt_llm_draft", help="Draft Model Name")
    
    args = parser.parse_args()

    if "host" in vars(args).keys() and args.host is None:
        args.host = "localhost"

    # disable logging (vllm)
    if "disable_log_stats" in vars(args).keys():
        args.disable_log_stats = True
    if "disable_log_requests" in vars(args).keys():
        args.disable_log_requests = True

    # always enable decoupled_mode (tritonserver)
    if "decoupled_mode" in vars(args).keys():
        args.decoupled_mode = True

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.kill is None:
        # launch server
        launch_server(args)
    else:
        # quit server
        quit_server(args.kill)