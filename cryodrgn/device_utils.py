"""Device detection and management utilities for cross-platform GPU support.

This module provides utilities for detecting and configuring GPU devices across
different backends (CUDA, MPS/Metal, CPU) to enable cryoDRGN to run on various
hardware platforms including NVIDIA GPUs, Apple Silicon (M-Series), and CPU-only
systems.
"""

import logging
import torch
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_available_device(
    device: Optional[str] = None, verbose: bool = True
) -> Tuple[torch.device, str]:
    """
    Detect and return the best available compute device.

    Priority order (if device not specified):
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Metal/M-Series)
    3. CPU (fallback)

    Args:
        device: Optional device string ('cuda', 'mps', 'cpu', or None for auto-detect)
        verbose: Whether to log device selection information

    Returns:
        Tuple of (torch.device, device_type_string)
    """
    # If device explicitly specified, validate and return it
    if device is not None:
        device_lower = device.lower()
        if device_lower == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device requested but CUDA is not available. "
                    "Please check your PyTorch installation and GPU drivers."
                )
            device_obj = torch.device("cuda")
            if verbose:
                logger.info(
                    f"Using CUDA device: {torch.cuda.get_device_name(0)} "
                    f"(explicitly requested)"
                )
            return device_obj, "cuda"
        elif device_lower == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS device requested but MPS is not available. "
                    "Please ensure you are running on Apple Silicon with PyTorch >= 1.12."
                )
            device_obj = torch.device("mps")
            if verbose:
                logger.info("Using MPS (Metal) device (explicitly requested)")
            return device_obj, "mps"
        elif device_lower == "cpu":
            device_obj = torch.device("cpu")
            if verbose:
                logger.info("Using CPU device (explicitly requested)")
            return device_obj, "cpu"
        else:
            raise ValueError(
                f"Invalid device '{device}'. Must be 'cuda', 'mps', 'cpu', or None."
            )

    # Auto-detect best available device
    if torch.cuda.is_available():
        device_obj = torch.device("cuda")
        device_str = "cuda"
        if verbose:
            logger.info(
                f"Using CUDA device: {torch.cuda.get_device_name(0)} "
                f"({torch.cuda.device_count()} GPU(s) available)"
            )
    elif torch.backends.mps.is_available():
        device_obj = torch.device("mps")
        device_str = "mps"
        if verbose:
            logger.info("Using MPS (Metal) device on Apple Silicon")
    else:
        device_obj = torch.device("cpu")
        device_str = "cpu"
        if verbose:
            logger.info("Using CPU device (no GPU acceleration available)")

    return device_obj, device_str


def supports_mixed_precision(device_type: str) -> bool:
    """
    Check if the device supports automatic mixed precision training.

    Args:
        device_type: Device type string ('cuda', 'mps', 'cpu')

    Returns:
        True if AMP is supported and recommended for this device
    """
    if device_type == "cuda":
        # CUDA has mature AMP support
        return True
    elif device_type == "mps":
        # MPS has experimental AMP support in PyTorch >= 2.0
        # Check if the required functionality is available
        try:
            # Test if MPS autocast is available
            return hasattr(torch.amp, "autocast") and torch.backends.mps.is_available()
        except Exception:
            return False
    else:
        # CPU AMP exists but typically not recommended
        return False


def get_autocast_context(device_type: str, enabled: bool = True):
    """
    Get the appropriate autocast context manager for the device.

    Args:
        device_type: Device type string ('cuda', 'mps', 'cpu')
        enabled: Whether to enable autocasting

    Returns:
        Autocast context manager or a no-op context
    """
    if not enabled or device_type == "cpu":
        # Return a no-op context manager
        from contextlib import nullcontext

        return nullcontext()

    if device_type == "cuda":
        # Try newer PyTorch API first, fall back to older
        try:
            return torch.amp.autocast("cuda")
        except (AttributeError, TypeError):
            try:
                return torch.cuda.amp.autocast()
            except AttributeError:
                from contextlib import nullcontext

                return nullcontext()
    elif device_type == "mps":
        # MPS autocast support (PyTorch >= 2.0)
        try:
            return torch.amp.autocast("mps")
        except (AttributeError, TypeError):
            # MPS autocast not available, fall back to no-op
            logger.warning(
                "MPS autocast not available in this PyTorch version. "
                "Training will use full precision."
            )
            from contextlib import nullcontext

            return nullcontext()
    else:
        from contextlib import nullcontext

        return nullcontext()


def get_grad_scaler(device_type: str, enabled: bool = True):
    """
    Get the appropriate gradient scaler for mixed precision training.

    Args:
        device_type: Device type string ('cuda', 'mps', 'cpu')
        enabled: Whether to enable gradient scaling

    Returns:
        GradScaler instance or None if not supported/disabled
    """
    if not enabled or device_type == "cpu":
        return None

    if device_type == "cuda":
        try:
            return torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            try:
                return torch.cuda.amp.GradScaler()
            except (AttributeError, TypeError):
                logger.warning("GradScaler not available. Disabling mixed precision.")
                return None
    elif device_type == "mps":
        # MPS GradScaler support (experimental)
        try:
            return torch.amp.GradScaler("mps")
        except (AttributeError, TypeError):
            logger.warning(
                "MPS GradScaler not available in this PyTorch version. "
                "Training will use full precision."
            )
            return None
    else:
        return None


def supports_multi_gpu(device_type: str) -> bool:
    """
    Check if multi-GPU training is supported for this device type.

    Args:
        device_type: Device type string ('cuda', 'mps', 'cpu')

    Returns:
        True if multi-GPU is supported
    """
    if device_type == "cuda":
        return torch.cuda.device_count() > 1
    elif device_type == "mps":
        # Apple Silicon unified memory architecture
        # Multi-GPU not supported in the traditional sense
        return False
    else:
        return False


def log_device_info(device_type: str):
    """
    Log detailed information about the selected device.

    Args:
        device_type: Device type string ('cuda', 'mps', 'cpu')
    """
    logger.info("=" * 60)
    logger.info("Device Information:")
    logger.info(f"  Device type: {device_type.upper()}")

    if device_type == "cuda":
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                logger.info(
                    f"      Memory: {props.total_memory / 1024**3:.2f} GB"
                )
    elif device_type == "mps":
        logger.info(f"  MPS available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            logger.info("  Running on Apple Silicon (M-Series)")
            logger.info("  Note: MPS uses unified memory architecture")
    else:
        logger.info("  No GPU acceleration available")

    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  Mixed precision support: {supports_mixed_precision(device_type)}")
    logger.info(f"  Multi-GPU support: {supports_multi_gpu(device_type)}")
    logger.info("=" * 60)
