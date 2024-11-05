from collections.abc import Generator, Callable
from dataclasses import dataclass, replace
from typing import Optional

import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

class RequestValidationError(Exception):
    pass

def _validate_that(condition: bool, msg: str):
    if not condition:
        raise RequestValidationError(msg)

def _validate_non_empty(data, msg: str):
    if isinstance(data, torch.Tensor):
        _validate_that(data is not None and data.numel() > 0, msg)
    else:
        _validate_that(data is not None and data.size > 0, msg)

@dataclass
class LLMRequest:
    input_ids: np.ndarray = np.array([])
    input_lengths: np.ndarray = np.array([])
    request_output_len: np.ndarray = np.array([])
    draft_input_ids: Optional[np.ndarray] = None
    decoder_input_ids: Optional[np.ndarray] = None
    decoder_input_lengths: Optional[np.ndarray] = None
    draft_logits: Optional[np.ndarray] = None
    draft_acceptance_threshold: Optional[np.ndarray] = None
    end_id: Optional[np.ndarray] = None
    pad_id: Optional[np.ndarray] = None
    stop_words_list: Optional[np.ndarray] = None
    bad_words_list: Optional[np.ndarray] = None
    embedding_bias: Optional[np.ndarray] = None
    beam_width: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None
    runtime_top_k: Optional[np.ndarray] = None
    runtime_top_p: Optional[np.ndarray] = None
    runtime_top_p_min: Optional[np.ndarray] = None
    runtime_top_p_decay: Optional[np.ndarray] = None
    runtime_top_p_reset_ids: Optional[np.ndarray] = None
    len_penalty: Optional[np.ndarray] = None
    early_stopping: Optional[np.ndarray] = None
    repetition_penalty: Optional[np.ndarray] = None
    min_length: Optional[np.ndarray] = None
    beam_search_diversity_rate: Optional[np.ndarray] = None
    presence_penalty: Optional[np.ndarray] = None
    frequency_penalty: Optional[np.ndarray] = None
    random_seed: Optional[np.ndarray] = None
    return_log_probs: Optional[np.ndarray] = None
    return_context_logits: Optional[np.ndarray] = None
    return_generation_logits: Optional[np.ndarray] = None
    stop: Optional[np.ndarray] = None
    streaming: Optional[np.ndarray] = None
    prompt_embedding_table: Optional[np.ndarray] = None
    prompt_table_extra_ids: Optional[np.ndarray] = None
    prompt_vocab_size: Optional[np.ndarray] = None
    lora_task_id: Optional[np.ndarray] = None
    lora_weights: Optional[np.ndarray] = None
    lora_config: Optional[np.ndarray] = None
    num_draft_tokens: Optional[np.ndarray] = None
    use_draft_logits: Optional[np.ndarray] = None

    def validate(self):
        pass
    
    @classmethod
    def with_new_inputs(cls, other, input_ids: np.ndarray, input_lengths: np.ndarray):
        return replace(
            other,
            input_ids = input_ids,
            input_lengths = input_lengths,
        )

@dataclass
class LLMResponse:
    output_ids: np.ndarray = np.array([])
    sequence_length: np.ndarray = np.array([])
    cum_log_probs: Optional[np.ndarray] = None
    output_log_probs: Optional[np.ndarray] = None
    context_logits: Optional[np.ndarray] = None
    generation_logits: Optional[np.ndarray] = None
    batch_index: Optional[np.ndarray] = None

    @classmethod
    def slice_from_last(cls, other, last_idx: int):
        new_output_ids = other.output_ids[0][0][last_idx:]
        return cls(
            output_ids = np.array([[new_output_ids]]),
            sequence_length = np.array([[len(new_output_ids)]], dtype=np.int32),
            cum_log_probs = other.cum_log_probs,
            output_log_probs = other.cum_log_probs,
            context_logits = other.context_logits,
            generation_logits = other.generation_logits,
            batch_index = other.batch_index
        )

@dataclass
class DraftRequest:
    draft_input_ids: Optional[np.ndarray] = None
    draft_logits: Optional[np.ndarray] = None

class Decoder:
    def __init__(self,
                 streaming=False,
                 accumulate=False,
                 llm_model_name="tensorrt_llm",
                 draft_llm_model_name: Optional[str] = None):
        self._streaming = streaming
        self._accumulate = accumulate
        self.llm_model_name = llm_model_name
        self.draft_llm_model_name = draft_llm_model_name
        self._accumulated_tokens = []

        self._llm_inputs = [
            "input_ids",
            "input_lengths",
            "request_output_len",
            "draft_input_ids",
            "decoder_input_ids",
            "decoder_input_lengths",
            "draft_logits",
            "draft_acceptance_threshold",
            "end_id",
            "pad_id",
            "stop_words_list",
            "bad_words_list",
            "embedding_bias",
            "beam_width",
            "temperature",
            "runtime_top_k",
            "runtime_top_p",
            "runtime_top_p_min",
            "runtime_top_p_decay",
            "runtime_top_p_reset_ids",
            "len_penalty",
            "early_stopping",
            "repetition_penalty",
            "min_length",
            "beam_search_diversity_rate",
            "presence_penalty",
            "frequency_penalty",
            "random_seed",
            "return_log_probs",
            "return_context_logits",
            "return_generation_logits",
            "stop",
            "streaming",
            "prompt_embedding_table",
            "prompt_table_extra_ids",
            "prompt_vocab_size",
            "lora_task_id",
            "lora_weights",
            "lora_config",
            "num_draft_tokens",
            "use_draft_logits",
        ]
        self._llm_outputs = [
            "output_ids", "sequence_length", "cum_log_probs",
            "output_log_probs", "context_logits", "generation_logits",
            "batch_index"
        ]
        self._exclude_request_names_for_llm = [
            "num_draft_tokens", "use_draft_logits",
        ]
        self._exclude_request_names_for_spec = [
            "num_draft_tokens", "use_draft_logits", "draft_input_ids", "draft_logits",
            "return_context_logits", "return_generation_logits", "request_output_len"
        ]
        self._undo_reshape_whitelist = {
            "input_lengths",
            "request_output_len",
            "decoder_input_lengths",
            "draft_acceptance_threshold",
            "end_id",
            "pad_id",
            "beam_width",
            "temperature",
            "runtime_top_k",
            "runtime_top_p",
            "runtime_top_p_min",
            "runtime_top_p_decay",
            "runtime_top_p_reset_ids",
            "len_penalty",
            "early_stopping",
            "repetition_penalty",
            "min_length",
            "beam_search_diversity_rate",
            "presence_penalty",
            "frequency_penalty",
            "random_seed",
            "return_log_probs",
            "return_context_logits",
            "return_generation_logits",
            "stop",
            "streaming",
            "prompt_vocab_size",
            "lora_task_id",
            "num_draft_tokens",
            "use_draft_logits",
        }

    def _undo_reshape(self, x, name):
        if name in self._undo_reshape_whitelist and len(x.shape) == 1:
            return np.expand_dims(x, 0)
        else:
            return x

    def _exec_triton_request(self, request):
        responses = request.exec(decoupled=True)
        for r in responses:
            if r.has_error():
                raise pb_utils.TritonModelException(r.error().message())
            yield r

    def _exec_triton_request_single(self, request):
        response = request.exec(decoupled=False)
        if response.has_error():
            raise pb_utils.TritonModelException(response.error().message())
        return response

    def _get_llm_tensors_from_request(
        self,
        request: LLMRequest,
        num_output_tokens: Optional[int] = None,
        draft_request: Optional[DraftRequest] = None,
        is_draft_model_request: bool = False,
        exclude_request_name_list: Optional[list] = None,
    ):
        batch_size = request.input_ids.shape[0]
        tensors = self.create_triton_tensors(request, exclude_request_name_list)
        out_len_tensor = None
        if request.request_output_len is not None:
            out_len_tensor = request.request_output_len
        
        out_len = None
        if num_output_tokens is not None:
            out_len = num_output_tokens
        elif draft_request:
            out_len = len(draft_request.draft_input_ids[0]) + 1 if draft_request.draft_input_ids is not None else 1
        
        if out_len is not None:
            out_len_tensor = [[out_len]] * batch_size
        if out_len_tensor is None:
            raise Exception("Could not determine request_output_len")
        else:
            tensors.append(pb_utils.Tensor("request_output_len", np.array(out_len_tensor, dtype=np.int32)))

        if draft_request:
            if draft_request.draft_input_ids is not None:
                tensors.append(pb_utils.Tensor("draft_input_ids", draft_request.draft_input_ids))
                if draft_request.draft_logits is not None and request.use_draft_logits is not None and request.use_draft_logits[0]:
                    tensors.append(pb_utils.Tensor("draft_logits", draft_request.draft_logits))

        return_context_logits_data = [False]
        return_generation_logits_data = [False]
        if draft_request is None:
            if is_draft_model_request:
                return_generation_logits_data = request.use_draft_logits if request.use_draft_logits is not None else [
                    False
                ]
            else:
                return_context_logits_data = request.return_context_logits if request.return_context_logits is not None else [
                    False
                ]
                return_generation_logits_data = request.return_generation_logits if request.return_generation_logits is not None else [
                    False
                ]
        return_context_logits = np.array([return_context_logits_data] *
                                         batch_size,
                                         dtype=bool)
        return_generation_logits = np.array([return_generation_logits_data] *
                                            batch_size,
                                            dtype=bool)

        assert len(return_context_logits.shape) == 2
        assert len(return_generation_logits.shape) == 2

        tensors.append(
            pb_utils.Tensor("return_context_logits", return_context_logits))
        tensors.append(
            pb_utils.Tensor("return_generation_logits",
                            return_generation_logits))
        
        return tensors
    
    def _get_llm_response(self, triton_output):
        return self.convert_triton_response(triton_output, LLMResponse)

    def _spec_generate(self, request: LLMRequest) -> Generator[LLMResponse, None, None]:
        prompt_input_ids = request.input_ids[0]
        input_ids = prompt_input_ids
        output_len = request.request_output_len[0]
        last_input_ids: np.ndarray = None
        draft_output_ids: np.ndarray = None
        draft_logits: np.ndarray = None

        target_response: LLMResponse = None

        while True:
            num_draft_tokens = min(
                request.num_draft_tokens[0][0],
                len(prompt_input_ids) + output_len - len(input_ids) - 1
            )
            draft_request: DraftRequest = None
            if num_draft_tokens > 0:
                draft_response = self._draft_generate_non_streaming(request, num_draft_tokens)
                seq_len = draft_response.sequence_length[0][0]
                draft_output_ids = draft_response.output_ids[0][0]
                input_draft_tokens = draft_output_ids[len(input_ids):seq_len]
                draft_request = DraftRequest(draft_input_ids=np.expand_dims(input_draft_tokens, 0))
            else:
                draft_request = DraftRequest()
            
            target_response = self._generate_non_streaming(request, draft_request)
            last_input_ids = input_ids
            input_ids = target_response.output_ids[0][0]

            target_response = LLMResponse.slice_from_last(target_response, len(last_input_ids))
            if target_response.sequence_length[0][0] == 0:
                break
            yield target_response

            # if hit or exceed the max output length, should stop
            if (len(input_ids) >= len(prompt_input_ids) + output_len):
                break
            # if draft and target have same outputs, should stop.
            if draft_output_ids is not None and np.array_equal(draft_output_ids, input_ids):
                break
            # if tokens no longer change, should stop (early stopping)
            if np.array_equal(last_input_ids, input_ids):
                break
            # need to check if stop words was encounted
            if request.stop_words_list is not None and self.encountered_stop_words(input_ids, request.stop_words_list[0]):
                break

            request.input_ids = np.expand_dims(input_ids, 0)
            request.input_lengths = np.array([[len(input_ids)]], dtype=np.int32)
            

    def _generate_non_streaming(
        self,
        request: LLMRequest,
        draft_request: Optional[DraftRequest] = None,
    ) -> LLMResponse:
        input_tensors = self._get_llm_tensors_from_request(request, None, draft_request, False, self._exclude_request_names_for_spec)
        triton_req = pb_utils.InferenceRequest(
            model_name = self.llm_model_name,
            inputs = input_tensors,
            requested_output_names = self._llm_outputs
        )
        r = self._exec_triton_request_single(triton_req)
        return self.convert_triton_response(r, LLMResponse)

    def _draft_generate_non_streaming(
        self,
        request: LLMRequest,
        num_draft_tokens: int
    ) -> LLMResponse:
        input_tensors = self._get_llm_tensors_from_request(request, num_draft_tokens, None, True, self._exclude_request_names_for_spec)
        triton_req = pb_utils.InferenceRequest(
            model_name = self.draft_llm_model_name,
            inputs = input_tensors,
            requested_output_names = self._llm_outputs
        )
        triton_response = self._exec_triton_request_single(triton_req)
        return self._get_llm_response(triton_response)

    def _generate(
        self,
        request: LLMRequest,
    ):
        raise NotImplementedError()

    def create_triton_tensors(self, request: LLMRequest, exclude_name_list: list = None):
        if exclude_name_list is None:
            exclude_name_list = []
        tensors = []
        for name in vars(request).keys():
            if name in exclude_name_list:
                continue
            value = getattr(request, name)
            if value is None:
                continue
            if isinstance(value, np.ndarray):
                t = pb_utils.Tensor(name, self._undo_reshape(value, name))
            elif isinstance(value, torch.Tensor):
                t = pb_utils.Tensor.from_dlpack(name, to_dlpack(self._undo_reshape(value, name)))
            tensors.append(t)
        return tensors
    
    def create_triton_response(self, response: LLMResponse):
        tensors = self.create_triton_tensors(response)
        return pb_utils.InferenceResponse(output_tensors=tensors)

    def convert_triton_request(self, triton_request) -> LLMRequest:
        request = LLMRequest()
        for triton_name in self._llm_inputs:
            tensor = pb_utils.get_input_tensor_by_name(triton_request,
                                                       triton_name)
            target_name = triton_name
            if tensor is None:
                continue
            if not hasattr(request, target_name):
                raise AttributeError(
                    f"LLMRequest has no attribute '{target_name}'")
            setattr(request, target_name, tensor.as_numpy())
        return request

    def convert_triton_response(self,
                                triton_response,
                                response_factory: Callable,
                                name_map=None):
        response = response_factory()
        for tensor in triton_response.output_tensors():
            if tensor is None:
                continue
            triton_name = tensor.name()
            if tensor.is_cpu():
                value = tensor.as_numpy()
            else:
                # If the tensor is in GPU memory make it torch.Tensor type
                value = from_dlpack(tensor.to_dlpack())
            target_name = triton_name
            if name_map and triton_name in name_map:
                target_name = name_map[triton_name]
            if name_map and not triton_name in name_map:
                continue
            if target_name is None:
                # explicitly ignore this triton input
                continue
            if not hasattr(response, target_name):
                raise AttributeError(
                    f"response object has not attribute '{target_name}'")
            setattr(response, target_name, value)
        return response

    def encountered_stop_words(self, input_ids: np.ndarray, stop_words_ids: np.ndarray):
        for stop_word_ids in stop_words_ids:
            if np.array_equal(input_ids[-len(stop_word_ids):], stop_word_ids):
                return True
        return False

    def decode(self, request: LLMRequest, speculative_decoding: bool = False) -> Generator[LLMResponse, None, None]:
        batch_size = request.input_ids.shape[0]
        self._accumulated_tokens = [None] * batch_size
        
        if speculative_decoding:
            if batch_size > 1:
                raise Exception(
                    "Speculative decoding is not supported with batch size > 1"
                )
            
            for response in self._spec_generate(request):
                yield response
        
        else:
            if not self._streaming and batch_size == 1:
                yield self._generate_non_streaming(request)
            else:
                for response in self._generate(request):
                    yield response

    def reset_decoder(self):
        self._accumulated_tokens = []