"""Unmodified HF full-recompute reference for the queued non-prefix experiment.

Variable-length active requests are repacked into left-padded DynamicCache
batches. This overhead is timed and reported; this is not an optimized native
SGLang scheduler. At batch one the original DynamicCache is used directly.
The all-global paged control provides a second, same-backend comparison.
"""
from dataclasses import dataclass
import time

import torch
from transformers import DynamicCache


@dataclass
class NativeRequestState:
    request_id: str
    cache: object
    history_tokens: int
    max_suffix_tokens: int
    suffix_tokens: int = 0


class NativeQueuedEngine:
    def __init__(self, model, *, prefix_chunk_size=2048):
        if type(prefix_chunk_size) is not int or prefix_chunk_size <= 0:
            raise ValueError("prefix_chunk_size must be positive")
        self.model = model
        self.device = next(model.parameters()).device
        self.prefix_chunk_size = prefix_chunk_size
        self.documents = {}
        self.requests = {}
        self.forward_calls = 0
        self.batched_forward_calls = 0
        self.repacked_kv_bytes = 0
        self.unpacked_kv_bytes = 0
        self.max_batch_size = 0
        self.failed = False

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def register_document(self, document_id, token_ids):
        ids = torch.as_tensor(token_ids, dtype=torch.long, device="cpu").reshape(-1)
        if document_id in self.documents:
            if not torch.equal(ids, self.documents[document_id]):
                raise ValueError("document ID collision")
            return
        self.documents[document_id] = ids.clone()

    @torch.inference_mode()
    def admit(self, request_id, prefix_ids, document_ids, *, max_suffix_tokens):
        if type(max_suffix_tokens) is not int or max_suffix_tokens <= 0:
            raise ValueError("max_suffix_tokens must be positive")
        if self.failed or request_id in self.requests:
            raise RuntimeError("invalid native request admission")
        tokens = torch.cat([torch.as_tensor(prefix_ids, dtype=torch.long, device="cpu").reshape(-1)]
                           + [self.documents[key] for key in document_ids])
        if not len(tokens) or len(tokens) + max_suffix_tokens > self.model.config.max_position_embeddings:
            raise ValueError("native request exceeds model context")
        cache = DynamicCache()
        for start in range(0, len(tokens), self.prefix_chunk_size):
            out = self.model(input_ids=tokens[start:start+self.prefix_chunk_size].to(self.device)[None],
                             past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
        state = NativeRequestState(request_id, cache, len(tokens), max_suffix_tokens)
        self.requests[request_id] = state
        return state

    @torch.inference_mode()
    def forward(self, request_ids, input_ids):
        if self.failed or len(set(request_ids)) != len(request_ids):
            raise RuntimeError("invalid native active request set")
        states = [self.requests[key] for key in request_ids]
        if input_ids.ndim != 2 or input_ids.shape[0] != len(states) or input_ids.shape[1] < 1:
            raise ValueError("native batch shape mismatch")
        count = input_ids.shape[1]
        if any(state.suffix_tokens + count > state.max_suffix_tokens for state in states):
            raise ValueError("native private suffix capacity exceeded")
        lengths = [state.history_tokens + state.suffix_tokens for state in states]
        position_ids = torch.tensor(lengths, device=self.device)[:, None] + torch.arange(count, device=self.device)
        maximum = max(lengths)
        mask = None
        if len(states) == 1:
            cache = states[0].cache
        else:
            cache = DynamicCache()
            for layer in range(self.model.config.num_hidden_layers):
                pairs = [(state.cache.layers[layer].keys, state.cache.layers[layer].values) for state in states]
                kk, vv = [], []
                for length, (k, v) in zip(lengths, pairs):
                    if k.shape[2] != length:
                        raise AssertionError("native cache length mismatch")
                    kk.append(torch.nn.functional.pad(k, (0, 0, maximum-length, 0)))
                    vv.append(torch.nn.functional.pad(v, (0, 0, maximum-length, 0)))
                k, v = torch.cat(kk), torch.cat(vv)
                self.repacked_kv_bytes += k.numel() * k.element_size() * 2
                cache.update(k, v, layer)
            positions = torch.arange(maximum + count, device=self.device)
            mask = positions[None, :] >= (maximum - torch.tensor(lengths, device=self.device))[:, None]
        try:
            result = self.model(input_ids=input_ids.to(self.device), position_ids=position_ids,
                                attention_mask=mask, past_key_values=cache, use_cache=True,
                                logits_to_keep=1)
            if len(states) > 1:
                for row, (state, length) in enumerate(zip(states, lengths)):
                    private = DynamicCache()
                    for layer in range(self.model.config.num_hidden_layers):
                        # Independent storage: release must not retain other
                        # request rows through views of a packed batch slab.
                        k = result.past_key_values.layers[layer].keys[row:row+1, :, maximum-length:, :].clone()
                        v = result.past_key_values.layers[layer].values[row:row+1, :, maximum-length:, :].clone()
                        self.unpacked_kv_bytes += (k.numel()+v.numel())*k.element_size()
                        private.update(k, v, layer)
                    state.cache = private
            else:
                states[0].cache = result.past_key_values
            for state in states:
                state.suffix_tokens += count
            self.forward_calls += 1
            self.batched_forward_calls += len(states) > 1
            self.max_batch_size = max(self.max_batch_size, len(states))
            return result.logits
        except BaseException:
            self.failed = True
            raise

    def release(self, request_id):
        self.requests.pop(request_id, None)

    def stats(self):
        return dict(requests=len(self.requests), forward_calls=self.forward_calls,
                    batched_forward_calls=self.batched_forward_calls,
                    max_batch_size=self.max_batch_size, repacked_kv_bytes=self.repacked_kv_bytes,
                    unpacked_kv_bytes=self.unpacked_kv_bytes,
                    backend=f"unmodified HF {self.model.config._attn_implementation}; ragged DynamicCache repacking",
                    full_recomputation=True, offline_kv_reuse=False,
                    production_scheduler=False)

    def close(self):
        self.synchronize()
        self.requests.clear()
        self.documents.clear()
