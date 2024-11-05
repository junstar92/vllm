import sys
import time
import json
import traceback
import random
import warnings
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import requests
import asyncio
import aiohttp
from tqdm.asyncio import tqdm

import numpy as np
from transformers import AutoTokenizer

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

@dataclass
class RequestFuncInput:
    request_id: str
    token_ids: list[int]
    input_len: int
    output_len: int
    ignore_eos: bool
    end_id: int
    pad_id: int
    model: str = ""

@dataclass
class SamplingParams:
    temperature: float = None
    top_k: int = None
    top_p: float = None

@dataclass
class SpeculativeDecodingParams:
    num_draft_tokens: int = None
    use_draft_logits: bool = None

@dataclass
class RequestFuncOutput:
    success: bool = False
    token_ids: list[int] = field(default_factory=list)
    error: str = ""
    start_ts: float = 0.0
    token_ts: list[float] = field(default_factory=list)
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)

@dataclass
class BenchmarkMetrics:
    num_requests: int
    num_successes: int
    total_input_tokens: int
    total_output_tokens: int
    output_token_throughput: float
    total_token_throughput: float
    sequence_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    mean_e2e_ms: float
    median_e2e_ms: float
    std_e2e_ms: float
    percentiles_e2e_ms: list[tuple[float, float]]


def sample_random_requests(
    num_requests: int,
    prefix_len: int,
    input_len: int,
    output_len: int,
    tokenizer: AutoTokenizer,
) -> list[tuple[list[int], int, int]]:
    prefix_token_ids = np.random.randint(0, tokenizer.vocab_size, size=prefix_len).tolist()
    input_requests = []
    for i in range(num_requests):
        input_ids = [tokenizer.bos_token_id] + prefix_token_ids + np.random.randint(0, tokenizer.vocab_size, size=input_len - 1).tolist()
        input_requests.append(
            (input_ids, prefix_len + input_len, output_len)
        )
    
    return input_requests

DATASET_SAMPLE_FUNCS = {
    "random": sample_random_requests,
    "sharegpt": "",
}

def get_model_id(api_url: str) -> str:
    response = requests.get(api_url).json()
    return response["data"][0]["id"]

async def async_request_vllm(
    api_url: str,
    request_func_input: RequestFuncInput,
    sampling_params: SamplingParams,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    assert api_url.endswith("completions")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.token_ids,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
            "best_of": 1,
        }
        if sampling_params.temperature:
            payload["temperature"] = sampling_params.temperature
        if sampling_params.top_k:
            payload["top_k"] = sampling_params.top_k
        if sampling_params.top_p:
            payload["top_p"] = sampling_params.top_p
        
        output = RequestFuncOutput()
        output.start_ts = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        timestamp = time.perf_counter()
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                        if chunk == "[DONE]":
                            continue
                        data = json.loads(chunk)

                        if data["choices"][0]["token_ids"]:
                            output.token_ts.append(timestamp)
                            output.token_ids += data["choices"][0]["token_ids"]

                    output.success = True
                
                else:
                    output.error = response.reason or ""
                    output.success = False

        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
        
        if pbar:
            pbar.update(1)

        return output

async def async_request_trtllm(
    api_url: str,
    request_func_input: RequestFuncInput,
    sampling_params: SamplingParams,
    speculative_decoding_params: SpeculativeDecodingParams,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "id": request_func_input.request_id,
            "input_ids": request_func_input.token_ids,
            "input_lengths": request_func_input.input_len,
            "request_output_len": request_func_input.output_len,
            "streaming": True,
            "end_id": request_func_input.end_id,
            "pad_id": request_func_input.pad_id,
            # "decoder_input_ids": [request_func_input.pad_id],
            # "decoder_input_lengths": 1,
            # "stop_words_list": [],
            # "bad_words_list": [],
            # "embedding_bias": [],
        }
        if request_func_input.ignore_eos:
            payload["min_length"] = request_func_input.output_len
        if sampling_params.temperature:
            payload["temperature"] = sampling_params.temperature
        if sampling_params.top_k:
            payload["runtime_top_k"] = sampling_params.top_k
        if sampling_params.top_p:
            payload["runtime_top_p"] = sampling_params.top_p
        if speculative_decoding_params.num_draft_tokens:
            payload["num_draft_tokens"] = speculative_decoding_params.num_draft_tokens
            payload["streaming"] = False
        if speculative_decoding_params.use_draft_logits:
            payload["use_draft_logits"] = True
        
        output = RequestFuncOutput()
        output.start_ts = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        timestamp = time.perf_counter()
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data:")
                        data = json.loads(chunk)
                        if data['sequence_length'] == 1:
                            output.token_ts.append(timestamp)
                            output.token_ids.append(data['output_ids'])
                        elif data['sequence_length'] > 1:
                            output.token_ts.extend([timestamp] * data['sequence_length'])
                            output.token_ids.extend(data['output_ids'])
                    
                    output.success = True
                
                else:
                    output.error = response.reason or ""
                    output.success = False

        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
        
        if pbar:
            pbar.update(1)
        
        return output

REQUEST_FUNCS = {
    "vllm": async_request_vllm,
    "tensorrt-llm": async_request_trtllm,
}

async def get_request(
    input_requests: list[tuple[list[int], int, int]],
    request_rate: float,
) -> AsyncGenerator[tuple[list[int], int, int], None]:
    input_requests = iter(input_requests)
    for i, request in enumerate(input_requests):
        request_id = str(i).zfill(8)
        yield request_id, request

        if request_rate == float("inf"):
            continue
        
        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)

def calculate_metrics(
    input_requests: list[tuple[list[int], int, int]],
    outputs: list[RequestFuncOutput],
    duration_sec: float,
    tokenizer: AutoTokenizer,
    percentiles: list[float]
) -> tuple[BenchmarkMetrics, list[str]]:
    assert len(input_requests) == len(outputs)

    successes = 0
    total_input_tokens = 0
    total_output_tokens = 0
    itls: list[float] = []
    tpots: list[float] = []
    ttfts: list[float] = []
    e2es: list[float] = []
    generation_texts = []

    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = len(outputs[i].token_ids)
            total_input_tokens += input_requests[i][1]
            total_output_tokens += output_len
            latency = outputs[i].token_ts[-1] - outputs[i].start_ts
            outputs[i].ttft = outputs[i].token_ts[0] - outputs[i].start_ts
            if output_len > 1:
                tpots.append((latency - outputs[i].ttft) / (output_len - 1))
            for j in range(1, len(outputs[i].token_ts)):
                outputs[i].itl.append(outputs[i].token_ts[j] - outputs[i].token_ts[j-1])
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2es.append(latency)
            generation_texts.append(tokenizer.decode(outputs[i].token_ids))
            
            successes += 1

    metrics = BenchmarkMetrics(
        num_requests=len(input_requests),
        num_successes=successes,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        output_token_throughput=total_output_tokens / duration_sec,
        total_token_throughput=(total_input_tokens + total_output_tokens) / duration_sec,
        sequence_throughput=successes / duration_sec,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000) for p in percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000) for p in percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000) for p in percentiles],
        mean_e2e_ms=np.mean(e2es or 0) * 1000,
        median_e2e_ms=np.median(e2es or 0) * 1000,
        std_e2e_ms=np.std(e2es or 0) * 1000,
        percentiles_e2e_ms=[(p, np.percentile(e2es or 0, p) * 1000) for p in percentiles],
    )

    return metrics, generation_texts

async def benchmark(
    model_id: str,
    backend: str,
    base_url: str,
    api_url: str,
    request_rate: float,
    input_requests: list[tuple[list[int], int, int]],
    ignore_eos: bool,
    end_id: int,
    pad_id: int,
    sampling_params: SamplingParams,
    speculative_decoding_params: SpeculativeDecodingParams,
    percentiles: list[float],
    tokenizer: AutoTokenizer,
):
    if end_id is None or end_id < 0:
        end_id = tokenizer.eos_token_id
        warnings.warn(f"end_id is not defined. Use default value: {end_id}.")
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        warnings.warn(f"pad_id is not defined. Use default value: {pad_id}.")
    request_func = REQUEST_FUNCS[backend]
    pbar = tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        request_id, (input_ids, input_len, output_len) = request
        request_input = RequestFuncInput(
            request_id=request_id,
            token_ids=input_ids,
            input_len=input_len,
            output_len=output_len,
            ignore_eos=ignore_eos,
            end_id=end_id,
            pad_id=pad_id,
            model=model_id,
        )
        tasks.append(
            asyncio.create_task(
                request_func(
                    api_url,
                    request_input,
                    sampling_params,
                    speculative_decoding_params,
                    pbar,
                )
            )
        )
    
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)
    pbar.close()
    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, generation_texts = calculate_metrics(
        input_requests,
        outputs,
        benchmark_duration,
        tokenizer,
        percentiles
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Total requests:", metrics.num_requests))
    print("{:<40} {:<10}".format("Successful requests:", metrics.num_successes))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input_tokens))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output_tokens))
    print("{:<40} {:<10.2f}".format("Sequence throughput (seq/s):", metrics.sequence_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_token_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", metrics.total_token_throughput))

    if backend == "vllm":
        try:
            iteration_data = requests.get(base_url + "/iteration_data").json()
        except:
            iteration_data = None
    else:
        iteration_data = None

    result = {
        "latency": benchmark_duration,
        "num_requests": metrics.num_requests,
        "num_successes": metrics.num_successes,
        "total_input_tokens": metrics.total_input_tokens,
        "total_output_tokens": metrics.total_output_tokens,
        "output_token_throughput": metrics.output_token_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "sequence_throughput": metrics.sequence_throughput,
        "input_lens": [input_request[1] for input_request in input_requests],
        "output_lens": [len(output.token_ids) for output in outputs],
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": generation_texts,
        "errors": [output.error for output in outputs],
        "token_ts": [{"start_ts": output.start_ts, "timestamps": output.token_ts} for output in outputs],
        "log_iterations": iteration_data,
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms")))
        print("{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms")))
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics,
                                f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                            value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT",
                       "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2e", "E2E", "End-to-end Latency")

    print("=" * 50)
    
    return result

def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark an online serving system.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=list(REQUEST_FUNCS.keys()),
        required=True,
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Path of the tokenizer",
        required=True,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )
    parser.add_argument("--random-seed", type=int, default=0)

    benchmark_group = parser.add_argument_group("benchmark options")
    benchmark_group.add_argument(
        "--endpoint",
        type=str,
        help="API endpoint",
    )
    benchmark_group.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second.",
    )
    benchmark_group.add_argument("--end-id", type=int, default=None)
    benchmark_group.add_argument("--pad-id", type=int, default=None)
    benchmark_group.add_argument("--ignore-eos", action="store_true")
    benchmark_group.add_argument(
        "--metric-percentiles",
        type=str,
        default="1,99",
        help="Comma-seperated list of percentiles for metrics. Default value is \"1,99\"."
    )
    benchmark_group.add_argument(
        "--save-result",
        type=str,
        default=None,
        help="Specify the file path to save benchmark json results. If not specified, results are not saved."
    )

    dataset_group = parser.add_argument_group("dataset options")
    dataset_group.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_SAMPLE_FUNCS.keys()),
        required=True,
    )
    dataset_group.add_argument(
        "--dataset-path",
        type=str,
        help="Path to the dataset."
    )
    dataset_group.add_argument(
        "--num-requests",
        type=int,
        default=1000,
        help="Number of requests to process.",
    )
    dataset_group.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before random context, used only for random dataset."
    )
    dataset_group.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Number of input tokens per request. Overrides the input length from selected dataset "
        "if it is smaller than request's input length."
    )
    dataset_group.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Number of output tokens per request. Overrides the output length from selected dataset."
    )

    sampling_group = parser.add_argument_group("sampling options")
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=None,
    )
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=None,
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=None,
    )
    sampling_group.add_argument(
        "--len-penalty",
        type=float,
        default=None,
    )
    sampling_group.add_argument(
        "--early-stopping",
        action="store_true"
    )
    sampling_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
    )
    sampling_group.add_argument(
        "--presense-penalty",
        type=float,
        default=None,
    )
    sampling_group.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
    )

    speculative_decoding_group = parser.add_argument_group("speculative decoding options")
    speculative_decoding_group.add_argument('--num-draft-tokens', type=int, default=None)
    speculative_decoding_group.add_argument('--use-draft-logits', action='store_true')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    backend = args.backend
    api_url = f"http://{args.host}:{args.port}{args.endpoint}"
    base_url = f"http://{args.host}:{args.port}"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model_id = get_model_id(f"http://{args.host}:{args.port}/v1/models") if backend == "vllm" else ""

    if args.dataset == "random":
        assert args.input_len is not None
        assert args.output_len is not None
        input_requests = sample_random_requests(
            args.num_requests,
            args.prefix_len,
            args.input_len,
            args.output_len,
            tokenizer
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    sampling_params = SamplingParams(
        args.temperature,
        args.top_k,
        args.top_p
    )
    speculative_decoding_params = SpeculativeDecodingParams(
        args.num_draft_tokens,
        args.use_draft_logits
    )
    
    result_json = asyncio.run(
        benchmark(
            model_id,
            backend,
            base_url,
            api_url,
            args.request_rate,
            input_requests,
            args.ignore_eos,
            args.end_id,
            args.pad_id,
            sampling_params,
            speculative_decoding_params,
            [float(p) for p in args.metric_percentiles.split(",")],
            tokenizer,
        )
    )

    if args.save_result:
        with open(args.save_result, "w", encoding='utf-8') as f:
            json.dump(result_json, f)