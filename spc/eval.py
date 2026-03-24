# eval.py
"""
Offline evaluation script for SPC framework inference results.

Usage:
  python eval.py --pred results/inference_detailed_results.pkl
  python eval.py --pred results/inference_detailed_results.pkl --verbose

Metrics reported (mirroring Section 5.2 of the paper):
  • ADE / FDE  – overall and per-subset (Normal / Accident)
  • CR          – Collision Rate (% of trajectories that leave the drivable area)
  • Intent Acc  – Fraction of samples where the predicted meta-action matches GT
"""
import argparse
import json
import pickle
import sys

import numpy as np

from src.utils.metrics import ADE, FDE


# -----------------------------------------------------------------------
# Collision rate
# -----------------------------------------------------------------------
def _collision_rate(records: list, lane_width: float = 3.5) -> float:
    """
    Estimates CR from saved inference records.
    A sample is flagged as a collision if the stored 'cr' field > 0,
    or if pred_traj deviates > lane_width from GT at any point.
    """
    count = 0
    for r in records:
        # Use pre-computed collision rate if available
        if 'cr' in r:
            if r['cr'] > 0:
                count += 1
            continue
        # Fallback: large positional deviation heuristic
        pred = np.array(r.get('pred_traj', []))
        gt   = np.array(r.get('gt_future', []))
        if pred.size == 0 or gt.size == 0:
            continue
        max_dev = np.max(np.linalg.norm(pred[..., :2] - gt[..., :2], axis=-1))
        if max_dev > lane_width:
            count += 1
    return count / len(records) if records else 0.0


# -----------------------------------------------------------------------
# Intent accuracy
# -----------------------------------------------------------------------
def _intent_accuracy(records: list) -> float:
    correct = sum(
        1 for r in records
        if r.get('pred_action') is not None and r.get('gt_action') is not None
        and r['pred_action'] == r['gt_action']
    )
    return correct / len(records) if records else 0.0


# -----------------------------------------------------------------------
# Per-subset stats
# -----------------------------------------------------------------------
def _subset_stats(records: list, verbose: bool = False) -> dict:
    if not records:
        return {'ade': 0.0, 'fde': 0.0, 'cr': 0.0, 'n': 0}

    ade_list, fde_list = [], []
    for r in records:
        pred = np.array(r.get('pred_traj', r.get('pred', [])))
        gt   = np.array(r.get('gt_future', np.zeros_like(pred)))
        if pred.size == 0:
            continue
        ade_list.append(ADE(pred[..., :2], gt[..., :2]))
        fde_list.append(FDE(pred[..., :2], gt[..., :2]))

    cr  = _collision_rate(records)
    ade = float(np.mean(ade_list)) if ade_list else 0.0
    fde = float(np.mean(fde_list)) if fde_list else 0.0

    if verbose:
        print(f"  ADE  = {ade:.4f} m  (±{np.std(ade_list):.4f})")
        print(f"  FDE  = {fde:.4f} m  (±{np.std(fde_list):.4f})")
        print(f"  CR   = {cr*100:.2f}%")

    return {'ade': ade, 'fde': fde, 'cr': cr, 'n': len(records)}


# -----------------------------------------------------------------------
# Main evaluate function
# -----------------------------------------------------------------------
def evaluate(pred_file: str, gt_dir: str = None, verbose: bool = False):
    # Load inference results
    if pred_file.endswith('.pkl'):
        with open(pred_file, 'rb') as f:
            records = pickle.load(f)
    elif pred_file.endswith('.json'):
        with open(pred_file, 'r') as f:
            records = json.load(f)
    else:
        print(f"[Error] Unsupported file format: {pred_file}", file=sys.stderr)
        return

    if not records:
        print("[Error] No records found in prediction file.", file=sys.stderr)
        return

    # Partition into Normal / Accident subsets
    normal   = [r for r in records if r.get('scene_type', 'normal') == 'normal']
    accident = [r for r in records if r.get('scene_type', 'normal') == 'accident']

    print(f"\n{'='*55}")
    print(f"  Evaluation Report  (total={len(records)})")
    print(f"{'='*55}")

    # ---- Overall ----
    print(f"\n[Overall  N={len(records)}]")
    overall = _subset_stats(records, verbose=verbose)
    print(f"  ADE  = {overall['ade']:.4f} m")
    print(f"  FDE  = {overall['fde']:.4f} m")
    print(f"  CR   = {overall['cr']*100:.2f}%")

    ia = _intent_accuracy(records)
    print(f"  Intent Accuracy = {ia*100:.2f}%")

    # ---- Normal subset ----
    if normal:
        print(f"\n[Normal   N={len(normal)}]")
        n_stats = _subset_stats(normal, verbose=verbose)
        print(f"  ADE  = {n_stats['ade']:.4f} m")
        print(f"  FDE  = {n_stats['fde']:.4f} m")
        print(f"  CR   = {n_stats['cr']*100:.2f}%")

    # ---- Accident subset ----
    if accident:
        print(f"\n[Accident N={len(accident)}]")
        a_stats = _subset_stats(accident, verbose=verbose)
        print(f"  ADE  = {a_stats['ade']:.4f} m")
        print(f"  FDE  = {a_stats['fde']:.4f} m")
        print(f"  CR   = {a_stats['cr']*100:.2f}%")

    print(f"\n{'='*55}\n")

    return {
        'overall':  overall,
        'normal':   _subset_stats(normal)   if normal   else None,
        'accident': _subset_stats(accident) if accident else None,
        'intent_accuracy': ia,
    }


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(description="SPC Framework – Offline Evaluation")
    parser.add_argument("--pred", required=True,
                        help="Path to inference results (.pkl or .json)")
    parser.add_argument("--gt",  default=None,
                        help="(optional) Ground truth directory (future use)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print std-dev statistics for ADE/FDE")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(args.pred, args.gt, verbose=args.verbose)
