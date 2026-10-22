"""Hybrid MoE backend: wraps SGLang's FusedMoE quantization methods (weight routing to host / GPU, decode
dispatch, prefill GPU path, CUDA-graph capture)."""
