"""
Activation Quantization and Dequantization Module for Transformer Split Inference.
Supports FP32, INT16, INT8, and INT4 (with real 4-bit uint8 packing/unpacking).
Computes exact compression ratios, MSE, SNR, and Cosine Similarity metrics.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, Union
import math

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


@dataclass
class QuantizationResult:
    """Represents the packed quantized activation payload ready for network transmission."""
    quantized_payload: Any         # Packed bytes / quantized array / tensor
    scale: float                   # Scaling factor for dequantization
    zero_point: float              # Zero point offset
    shape: Tuple[int, ...]         # Original tensor dimensions (e.g. batch, seq_len, hidden_dim)
    bitwidth: int                  # 4, 8, 16, or 32
    original_size_bytes: int       # Exact byte size in FP32
    compressed_size_bytes: int     # Exact transmitted payload size in bytes
    compression_ratio: float       # original_size / compressed_size
    is_enabled: bool = True


class ActivationQuantizer:
    """
    Performs real quantization, packing, dequantization and empirical fidelity evaluation.
    """

    SUPPORTED_BITWIDTHS = [4, 8, 16, 32]

    @staticmethod
    def quantize(
        tensor: Any,
        bitwidth: int = 8,
        enabled: bool = True
    ) -> QuantizationResult:
        """
        Quantizes intermediate activation tensor.
        Supported bitwidths: 32 (FP32), 16 (INT16), 8 (INT8), 4 (INT4 packed).
        """
        if bitwidth not in ActivationQuantizer.SUPPORTED_BITWIDTHS:
            raise ValueError(f"Unsupported bitwidth {bitwidth}. Must be one of {ActivationQuantizer.SUPPORTED_BITWIDTHS}")

        # Convert to numpy if PyTorch tensor
        if HAS_TORCH and isinstance(tensor, torch.Tensor):
            data = tensor.detach().cpu().to(torch.float32).numpy()
        elif HAS_NUMPY and isinstance(tensor, np.ndarray):
            data = tensor.astype(np.float32)
        else:
            # Fallback list of floats
            if HAS_NUMPY:
                data = np.array(tensor, dtype=np.float32)
            else:
                raise RuntimeError("NumPy is required for tensor quantization.")

        original_shape = data.shape
        num_elements = data.size
        original_size_bytes = num_elements * 4  # 4 bytes for FP32

        # If quantization disabled or FP32 requested
        if not enabled or bitwidth == 32:
            return QuantizationResult(
                quantized_payload=data,
                scale=1.0,
                zero_point=0.0,
                shape=original_shape,
                bitwidth=32,
                original_size_bytes=original_size_bytes,
                compressed_size_bytes=original_size_bytes,
                compression_ratio=1.0,
                is_enabled=False
            )

        flat = data.flatten()
        val_min = float(np.min(flat))
        val_max = float(np.max(flat))

        # Avoid zero division on constant activations
        if abs(val_max - val_min) < 1e-8:
            val_max = val_min + 1e-4

        if bitwidth == 16:
            # INT16: range [-32768, 32767]
            q_min, q_max = -32768, 32767
            scale = (val_max - val_min) / float(q_max - q_min)
            zero_point = round(-val_min / scale) + q_min
            zero_point = max(q_min, min(q_max, zero_point))
            
            q_data = np.round(flat / scale + zero_point)
            q_data = np.clip(q_data, q_min, q_max).astype(np.int16)
            
            payload = q_data.tobytes()
            # 2 bytes per element + 8 bytes header for scale and zero_point
            compressed_bytes = num_elements * 2 + 8
            
        elif bitwidth == 8:
            # INT8: range [-128, 127]
            q_min, q_max = -128, 127
            scale = (val_max - val_min) / float(q_max - q_min)
            zero_point = round(-val_min / scale) + q_min
            zero_point = max(q_min, min(q_max, zero_point))
            
            q_data = np.round(flat / scale + zero_point)
            q_data = np.clip(q_data, q_min, q_max).astype(np.int8)
            
            payload = q_data.tobytes()
            # 1 byte per element + 8 bytes header
            compressed_bytes = num_elements * 1 + 8

        elif bitwidth == 4:
            # INT4: range [0, 15] (unsigned 4-bit)
            # Physical packing: 2 values packed per uint8 byte
            q_min, q_max = 0, 15
            scale = (val_max - val_min) / float(q_max - q_min)
            zero_point = round(-val_min / scale) + q_min
            zero_point = max(q_min, min(q_max, zero_point))

            q_data = np.round(flat / scale + zero_point)
            q_data = np.clip(q_data, q_min, q_max).astype(np.uint8)

            # Pad odd length if needed
            pad_needed = len(q_data) % 2 != 0
            if pad_needed:
                q_data = np.pad(q_data, (0, 1), mode="constant")

            # Bitwise packing: high 4 bits and low 4 bits
            even_vals = q_data[0::2]
            odd_vals = q_data[1::2]
            packed_bytes = (even_vals | (odd_vals << 4)).astype(np.uint8)
            
            payload = packed_bytes.tobytes()
            # Exactly half bytes: ceil(num_elements / 2) + 8 bytes metadata
            compressed_bytes = math.ceil(num_elements / 2.0) + 8

        ratio = original_size_bytes / max(1, compressed_bytes)

        return QuantizationResult(
            quantized_payload=payload,
            scale=scale,
            zero_point=float(zero_point),
            shape=original_shape,
            bitwidth=bitwidth,
            original_size_bytes=original_size_bytes,
            compressed_size_bytes=int(compressed_bytes),
            compression_ratio=round(ratio, 2),
            is_enabled=True
        )

    @staticmethod
    def dequantize(
        result: QuantizationResult,
        as_torch_tensor: bool = False
    ) -> Any:
        """
        Reconstructs floating point activation tensor from QuantizationResult.
        """
        if not result.is_enabled or result.bitwidth == 32:
            data = result.quantized_payload
            if as_torch_tensor and HAS_TORCH:
                return torch.from_numpy(data)
            return data

        num_elements = math.prod(result.shape)
        scale = result.scale
        zero_point = result.zero_point

        if result.bitwidth == 16:
            raw_array = np.frombuffer(result.quantized_payload, dtype=np.int16)
            reconstructed = (raw_array.astype(np.float32) - zero_point) * scale

        elif result.bitwidth == 8:
            raw_array = np.frombuffer(result.quantized_payload, dtype=np.int8)
            reconstructed = (raw_array.astype(np.float32) - zero_point) * scale

        elif result.bitwidth == 4:
            packed_array = np.frombuffer(result.quantized_payload, dtype=np.uint8)
            # Unpack two 4-bit nibbles per byte
            even_vals = (packed_array & 0x0F).astype(np.float32)
            odd_vals = ((packed_array >> 4) & 0x0F).astype(np.float32)
            
            # Interleave back
            interleaved = np.empty(len(packed_array) * 2, dtype=np.float32)
            interleaved[0::2] = even_vals
            interleaved[1::2] = odd_vals
            
            # Truncate any padding added during packing
            raw_unpacked = interleaved[:num_elements]
            reconstructed = (raw_unpacked - zero_point) * scale

        reshaped = reconstructed.reshape(result.shape)
        if as_torch_tensor and HAS_TORCH:
            return torch.from_numpy(reshaped)
        return reshaped

    @staticmethod
    def evaluate_quality(
        original: Any,
        dequantized: Any
    ) -> Dict[str, float]:
        """
        Empirically measures quantization error:
        - MSE (Mean Squared Error)
        - SNR (Signal-to-Noise Ratio in dB)
        - Cosine Similarity (directional alignment of representation vectors)
        - Quality score Q in [0.0, 1.0]
        - Degradation D = 1.0 - Q
        """
        if HAS_TORCH and isinstance(original, torch.Tensor):
            orig_np = original.detach().cpu().to(torch.float32).numpy().flatten()
        else:
            orig_np = np.asarray(original, dtype=np.float32).flatten()

        if HAS_TORCH and isinstance(dequantized, torch.Tensor):
            deq_np = dequantized.detach().cpu().to(torch.float32).numpy().flatten()
        else:
            deq_np = np.asarray(dequantized, dtype=np.float32).flatten()

        diff = orig_np - deq_np
        mse = float(np.mean(diff ** 2))

        # Signal Power & Noise Power
        signal_power = float(np.mean(orig_np ** 2))
        noise_power = max(1e-12, mse)
        snr_db = 10.0 * math.log10(max(1e-12, signal_power) / noise_power)

        # Cosine Similarity
        norm_orig = float(np.linalg.norm(orig_np))
        norm_deq = float(np.linalg.norm(deq_np))
        if norm_orig > 1e-8 and norm_deq > 1e-8:
            cos_sim = float(np.dot(orig_np, deq_np) / (norm_orig * norm_deq))
        else:
            cos_sim = 1.0
        cos_sim = max(-1.0, min(1.0, cos_sim))

        # Normalized Quality Score (aligned with NLP representation quality)
        quality_score = max(0.0, min(1.0, (cos_sim + 1.0) / 2.0 if cos_sim < 0.9 else cos_sim))
        degradation = max(0.0, 1.0 - quality_score)

        return {
            "mse": round(mse, 6),
            "snr_db": round(snr_db, 2),
            "cosine_similarity": round(cos_sim, 5),
            "quality_score": round(quality_score, 4),
            "quality_degradation": round(degradation, 4)
        }
