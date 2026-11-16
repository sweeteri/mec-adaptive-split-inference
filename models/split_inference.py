"""
Split Inference Execution Engine.
Partitions Transformer execution between Mobile Device and MEC Server at split point k,
performing actual tensor slicing, intermediate activation quantization, and network offloading.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import time
import torch
import torch.nn.functional as F

from models.transformer_model import ModularTransformer, TransformerConfig
from models.quantization import ActivationQuantizer, QuantizationResult


@dataclass
class SplitInferenceOutput:
    """Complete results and performance metrics of a split inference pass."""
    logits: torch.Tensor
    predicted_class: int
    split_point: int
    bitwidth: int
    
    # Quantization payload and metrics
    quantization_result: Optional[QuantizationResult]
    quality_metrics: Dict[str, float]
    
    # Measured Empirical Profiling (CPU timestamps)
    t_device_measured_s: float
    t_mec_measured_s: float
    t_quant_overhead_s: float
    t_dequant_overhead_s: float
    
    # Transmitted payload size
    transmitted_bytes: int
    original_bytes: int
    compression_ratio: float


class SplitInferenceExecutor:
    """
    Executes split inference of a Transformer model across device and MEC server.
    """

    def __init__(self, model: Optional[ModularTransformer] = None):
        self.model = model or ModularTransformer()
        self.model.eval()
        self.quantizer = ActivationQuantizer()

    @torch.no_grad()
    def execute(
        self,
        input_ids: torch.Tensor,
        split_point: int,
        bitwidth: int = 8,
        quantization_enabled: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        compute_ground_truth_comparison: bool = True
    ) -> SplitInferenceOutput:
        """
        Executes partition at split_point k:
        - k = 0: Full offloading (MEC computes all layers)
        - k = L: Fully local (Device computes all layers)
        - 0 < k < L: Split inference (Device computes layers 0..k-1, MEC computes k..L-1)
        """
        total_layers = self.model.config.num_layers
        split_point = max(0, min(total_layers, split_point))

        # Ground truth FP32 baseline for exact quality calculation
        fp32_baseline_logits = None
        if compute_ground_truth_comparison:
            fp32_baseline_logits, _ = self.model.forward_full(input_ids, attention_mask)

        t_device = 0.0
        t_mec = 0.0
        t_quant = 0.0
        t_dequant = 0.0
        q_result: Optional[QuantizationResult] = None
        quality_metrics = {
            "mse": 0.0,
            "snr_db": 99.9,
            "cosine_similarity": 1.0,
            "quality_score": 1.0,
            "quality_degradation": 0.0,
            "top1_match": 1.0
        }

        # Case 1: Full Local Inference (k = total_layers)
        if split_point == total_layers:
            t0 = time.perf_counter()
            logits, profiles = self.model.forward_full(input_ids, attention_mask)
            t_device = time.perf_counter() - t0
            
            return SplitInferenceOutput(
                logits=logits,
                predicted_class=int(torch.argmax(logits, dim=-1)[0].item()),
                split_point=split_point,
                bitwidth=32,
                quantization_result=None,
                quality_metrics=quality_metrics,
                t_device_measured_s=round(t_device, 6),
                t_mec_measured_s=0.0,
                t_quant_overhead_s=0.0,
                t_dequant_overhead_s=0.0,
                transmitted_bytes=0,
                original_bytes=0,
                compression_ratio=1.0
            )

        # Case 2: Full Offloading (k = 0)
        elif split_point == 0:
            # Device computes 0 layers, only serializes raw input tokens
            raw_input_bytes = input_ids.numel() * 4  # int32 token IDs
            t0 = time.perf_counter()
            logits, profiles = self.model.forward_full(input_ids, attention_mask)
            t_mec = time.perf_counter() - t0
            
            return SplitInferenceOutput(
                logits=logits,
                predicted_class=int(torch.argmax(logits, dim=-1)[0].item()),
                split_point=split_point,
                bitwidth=32,
                quantization_result=None,
                quality_metrics=quality_metrics,
                t_device_measured_s=0.0,
                t_mec_measured_s=round(t_mec, 6),
                t_quant_overhead_s=0.0,
                t_dequant_overhead_s=0.0,
                transmitted_bytes=raw_input_bytes,
                original_bytes=raw_input_bytes,
                compression_ratio=1.0
            )

        # Case 3: Split Inference (0 < k < total_layers)
        else:
            # Phase A: Mobile Device Execution (Embedding + first k layers)
            t_dev_start = time.perf_counter()
            hidden, _ = self.model.forward_embeddings(input_ids)
            for i in range(split_point):
                hidden, _ = self.model.forward_layer(i, hidden, attention_mask)
            t_device = time.perf_counter() - t_dev_start

            # Phase B: Quantization of intermediate activations H_k
            t_q_start = time.perf_counter()
            q_result = self.quantizer.quantize(
                hidden,
                bitwidth=bitwidth,
                enabled=quantization_enabled
            )
            t_quant = time.perf_counter() - t_q_start

            # Phase C: Transmission & Dequantization on MEC Server
            t_dq_start = time.perf_counter()
            dequantized_hidden = self.quantizer.dequantize(q_result, as_torch_tensor=True)
            dequantized_hidden = dequantized_hidden.to(input_ids.device)
            t_dequant = time.perf_counter() - t_dq_start

            # Phase D: MEC Server Execution (remaining L - k layers + head)
            t_mec_start = time.perf_counter()
            current_hidden = dequantized_hidden
            for i in range(split_point, total_layers):
                current_hidden, _ = self.model.forward_layer(i, current_hidden, attention_mask)
            logits, _ = self.model.forward_head(current_hidden)
            t_mec = time.perf_counter() - t_mec_start

            # Compute empirical quality metrics relative to FP32 baseline
            if fp32_baseline_logits is not None:
                quality_metrics = self.quantizer.evaluate_quality(
                    fp32_baseline_logits,
                    logits
                )
                pred_split = int(torch.argmax(logits, dim=-1)[0].item())
                pred_base = int(torch.argmax(fp32_baseline_logits, dim=-1)[0].item())
                quality_metrics["top1_match"] = 1.0 if pred_split == pred_base else 0.0

            return SplitInferenceOutput(
                logits=logits,
                predicted_class=int(torch.argmax(logits, dim=-1)[0].item()),
                split_point=split_point,
                bitwidth=bitwidth,
                quantization_result=q_result,
                quality_metrics=quality_metrics,
                t_device_measured_s=round(t_device, 6),
                t_mec_measured_s=round(t_mec, 6),
                t_quant_overhead_s=round(t_quant, 6),
                t_dequant_overhead_s=round(t_dequant, 6),
                transmitted_bytes=q_result.compressed_size_bytes,
                original_bytes=q_result.original_size_bytes,
                compression_ratio=q_result.compression_ratio
            )
