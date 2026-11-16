"""
Unit tests and verification script for Stage 2: Transformer Split Inference & Activation Quantization.
"""

import sys
import os
import math

# Ensure project root and persistent pylib are in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
pylib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pylib"))
if os.path.exists(pylib_dir):
    sys.path.insert(0, pylib_dir)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from models.transformer_model import ModularTransformer, TransformerConfig
from models.quantization import ActivationQuantizer, QuantizationResult
from models.split_inference import SplitInferenceExecutor


def test_quantization_mechanisms():
    print("Testing ActivationQuantizer across FP32, INT16, INT8, and INT4 (with physical packing)...")
    quantizer = ActivationQuantizer()
    
    # Random activation tensor representing intermediate hidden state
    torch.manual_seed(42)
    sample_tensor = torch.randn(1, 64, 256, dtype=torch.float32)
    orig_bytes = sample_tensor.numel() * 4  # 64 * 256 * 4 = 65,536 bytes
    
    # 1. FP32
    res_fp32 = quantizer.quantize(sample_tensor, bitwidth=32)
    assert res_fp32.compressed_size_bytes == orig_bytes
    assert res_fp32.compression_ratio == 1.0
    deq_fp32 = quantizer.dequantize(res_fp32, as_torch_tensor=True)
    m_fp32 = quantizer.evaluate_quality(sample_tensor, deq_fp32)
    assert m_fp32["mse"] == 0.0
    assert m_fp32["quality_score"] == 1.0
    assert m_fp32["quality_degradation"] == 0.0
    print(f"  FP32: {res_fp32.compressed_size_bytes} B, Ratio: {res_fp32.compression_ratio}x, CosSim: {m_fp32['cosine_similarity']}")

    # 2. INT16
    res_int16 = quantizer.quantize(sample_tensor, bitwidth=16)
    assert res_int16.compression_ratio >= 1.95, f"Expected ~2x compression, got {res_int16.compression_ratio}"
    deq_int16 = quantizer.dequantize(res_int16, as_torch_tensor=True)
    m_int16 = quantizer.evaluate_quality(sample_tensor, deq_int16)
    assert m_int16["cosine_similarity"] > 0.9999
    assert m_int16["snr_db"] > 60.0
    print(f"  INT16: {res_int16.compressed_size_bytes} B, Ratio: {res_int16.compression_ratio}x, SNR: {m_int16['snr_db']} dB")

    # 3. INT8
    res_int8 = quantizer.quantize(sample_tensor, bitwidth=8)
    assert res_int8.compression_ratio >= 3.90, f"Expected ~4x compression, got {res_int8.compression_ratio}"
    deq_int8 = quantizer.dequantize(res_int8, as_torch_tensor=True)
    m_int8 = quantizer.evaluate_quality(sample_tensor, deq_int8)
    assert m_int8["cosine_similarity"] > 0.999
    assert m_int8["snr_db"] > 35.0
    print(f"  INT8: {res_int8.compressed_size_bytes} B, Ratio: {res_int8.compression_ratio}x, SNR: {m_int8['snr_db']} dB")

    # 4. INT4 (with physical 4-bit packing: 2 values per uint8 byte)
    res_int4 = quantizer.quantize(sample_tensor, bitwidth=4)
    # Check payload size: exactly ceil(num_elements / 2) bytes (+ 8 bytes scale/zero_point metadata)
    expected_packed_bytes = math.ceil(sample_tensor.numel() / 2.0) + 8
    assert res_int4.compressed_size_bytes == expected_packed_bytes
    assert res_int4.compression_ratio >= 7.80, f"Expected ~8x compression, got {res_int4.compression_ratio}"
    
    deq_int4 = quantizer.dequantize(res_int4, as_torch_tensor=True)
    assert deq_int4.shape == sample_tensor.shape
    m_int4 = quantizer.evaluate_quality(sample_tensor, deq_int4)
    assert m_int4["cosine_similarity"] > 0.96, f"INT4 Cosine similarity must be > 0.96, got {m_int4['cosine_similarity']}"
    assert m_int4["snr_db"] > 14.0
    print(f"  INT4 (Packed): {res_int4.compressed_size_bytes} B, Ratio: {res_int4.compression_ratio}x, SNR: {m_int4['snr_db']} dB, CosSim: {m_int4['cosine_similarity']}")
    print("  ActivationQuantizer passed.")


def test_modular_transformer():
    print("Testing ModularTransformer forward pass and layer profiling...")
    config = TransformerConfig.create_tiny_test_config(num_layers=4, hidden_dim=128)
    model = ModularTransformer(config)
    
    input_ids = torch.randint(0, config.vocab_size, (1, 32), dtype=torch.long)
    logits, profiles = model.forward_full(input_ids)
    
    assert logits.shape == (1, config.num_classes)
    assert "embeddings" in profiles
    assert "layer_0" in profiles
    assert "layer_3" in profiles
    assert "head" in profiles
    assert profiles["total"] > 0.0
    print(f"  Forward pass successful. Total execution time: {profiles['total'] * 1000.0:.2f} ms")


def test_split_inference_cases():
    print("Testing SplitInferenceExecutor for all boundary and split cases...")
    num_layers = 4
    config = TransformerConfig.create_tiny_test_config(num_layers=num_layers, hidden_dim=128)
    model = ModularTransformer(config)
    executor = SplitInferenceExecutor(model)
    
    input_ids = torch.randint(0, config.vocab_size, (1, 32), dtype=torch.long)
    
    # 1. Full Offloading: k = 0
    out_k0 = executor.execute(input_ids, split_point=0)
    assert out_k0.split_point == 0
    assert out_k0.t_device_measured_s == 0.0
    assert out_k0.t_mec_measured_s > 0.0
    assert out_k0.transmitted_bytes == 32 * 4  # raw token IDs
    assert out_k0.quality_metrics["quality_score"] == 1.0
    print(f"  k=0 (Offload): Device={out_k0.t_device_measured_s*1000:.1f}ms, Server={out_k0.t_mec_measured_s*1000:.1f}ms, Tx={out_k0.transmitted_bytes}B")

    # 2. Fully Local: k = L
    out_kL = executor.execute(input_ids, split_point=num_layers)
    assert out_kL.split_point == num_layers
    assert out_kL.t_device_measured_s > 0.0
    assert out_kL.t_mec_measured_s == 0.0
    assert out_kL.transmitted_bytes == 0
    assert out_kL.quality_metrics["quality_score"] == 1.0
    print(f"  k=L (Local): Device={out_kL.t_device_measured_s*1000:.1f}ms, Server={out_kL.t_mec_measured_s*1000:.1f}ms, Tx={out_kL.transmitted_bytes}B")

    # 3. Split Inference: k = 2 with INT8 Quantization
    out_k2_int8 = executor.execute(input_ids, split_point=2, bitwidth=8)
    assert out_k2_int8.split_point == 2
    assert out_k2_int8.bitwidth == 8
    assert out_k2_int8.t_device_measured_s > 0.0
    assert out_k2_int8.t_mec_measured_s > 0.0
    assert out_k2_int8.quantization_result is not None
    assert out_k2_int8.compression_ratio >= 3.8
    assert out_k2_int8.quality_metrics["cosine_similarity"] > 0.95
    print(f"  k=2 (INT8 Split): Compression={out_k2_int8.compression_ratio}x, CosSim={out_k2_int8.quality_metrics['cosine_similarity']:.4f}")

    # 4. Split Inference: k = 2 with INT4 Quantization
    out_k2_int4 = executor.execute(input_ids, split_point=2, bitwidth=4)
    assert out_k2_int4.bitwidth == 4
    assert out_k2_int4.compression_ratio >= 7.5
    assert out_k2_int4.quality_metrics["cosine_similarity"] > 0.85
    print(f"  k=2 (INT4 Split): Compression={out_k2_int4.compression_ratio}x, CosSim={out_k2_int4.quality_metrics['cosine_similarity']:.4f}")
    print("  SplitInferenceExecutor passed.")


if __name__ == "__main__":
    test_quantization_mechanisms()
    test_modular_transformer()
    test_split_inference_cases()
    print("\nALL STAGE 2 VERIFICATION TESTS PASSED SUCCESSFULLY!")
