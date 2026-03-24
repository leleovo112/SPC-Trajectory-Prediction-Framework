# src/utils/metrics.py
"""
Standard trajectory prediction evaluation metrics.

ADE – Average Displacement Error (metres)
FDE – Final Displacement Error (metres)
CR  – Collision Rate utility (see infer.py / eval.py for full implementation)
"""
import numpy as np


def ADE(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Average Displacement Error over the prediction horizon.
    pred, gt : (..., T, 2) position arrays.
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt   = np.asarray(gt,   dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch in ADE: pred={pred.shape}, gt={gt.shape}"
        )
    return float(np.mean(np.linalg.norm(pred - gt, axis=-1)))


def FDE(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Final Displacement Error (distance at the last predicted step).
    pred, gt : (..., T, 2) position arrays.
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt   = np.asarray(gt,   dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch in FDE: pred={pred.shape}, gt={gt.shape}"
        )
    return float(np.linalg.norm(pred[..., -1, :] - gt[..., -1, :]))


def MR(pred: np.ndarray, gt: np.ndarray, threshold: float = 2.0) -> float:
    """
    Miss Rate: fraction of samples where FDE > threshold metres.
    pred, gt : (N, T, 2)
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt   = np.asarray(gt,   dtype=np.float32)
    fde_per_sample = np.linalg.norm(pred[:, -1, :] - gt[:, -1, :], axis=-1)
    return float(np.mean(fde_per_sample > threshold))
