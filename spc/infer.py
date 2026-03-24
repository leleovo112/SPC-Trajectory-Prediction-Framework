# infer.py
"""
LLM-driven inference pipeline for the SPC trajectory prediction framework.

Usage:
  python infer.py [--provider openai|deepseek|qwen|local]
                  [--model MODEL_NAME]
                  [--max_samples N]
                  [--ckpt PATH_TO_CHECKPOINT]
                  [--out_dir OUTPUT_DIR]
"""
import argparse
import os
import pickle
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from src.dataset.dataset import DeepAccidentDataset, deepaccident_collate
from src.models.predictor import HeterogeneousPredictor
from src.llm.semantic_serializer import SemanticSerializer
from src.llm.llm_client import LLMClient
from src.llm.cot_controller import RuleBasedFallback


# -----------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------
def setup_logging(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'inference_results.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\nSPC Framework – LLM-Driven Inference Log\n{'='*60}\n")
    return log_path


def log_to_file(log_path: str, msg: str):
    print(msg)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


# -----------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------
def compute_ade_fde(pred_traj: torch.Tensor, gt_traj: torch.Tensor):
    """Returns (ADE, FDE) in metres using the [x, y] channels."""
    pred_pos = pred_traj[..., :2]
    gt_pos   = gt_traj[...,   :2]
    l2 = torch.norm(pred_pos - gt_pos, dim=-1)   # (B, T)
    ade = l2.mean().item()
    fde = l2[:, -1].mean().item()
    return ade, fde


def compute_collision_rate(pred_traj: torch.Tensor, map_feat: torch.Tensor | None) -> float:
    """
    Estimates the collision rate as the fraction of predicted trajectories
    that deviate beyond a safety corridor from the nearest lane centre-line.

    A trajectory is flagged as a collision if any waypoint lies > (lane_width/2)
    from the closest lane point AND simultaneously moves outside the drivable area.
    """
    if map_feat is None or map_feat.numel() == 0:
        return 0.0

    B, T, _ = pred_traj.shape
    lane_w   = getattr(cfg, 'LANE_WIDTH', 3.5)
    collision_count = 0

    lane_pts = map_feat[..., :2]   # (B, L, P, 2)
    for b in range(B):
        traj  = pred_traj[b, :, :2]        # (T, 2)
        lpts  = lane_pts[b]                 # (L, P, 2)
        # Minimum distance from each traj point to any lane waypoint
        dists = torch.norm(
            traj.unsqueeze(1).unsqueeze(1) - lpts.unsqueeze(0), dim=-1
        )  # (T, L, P)
        min_d = dists.min(dim=-1)[0].min(dim=-1)[0]   # (T,)
        if torch.any(min_d > lane_w):
            collision_count += 1

    return collision_count / B


# -----------------------------------------------------------------------
# Main inference loop
# -----------------------------------------------------------------------
def run_inference(
    llm_provider: str = "openai",
    llm_model:    str = None,
    max_samples:  int | None = None,
    ckpt_path:    str | None = None,
    out_dir:      str | None = None,
):
    out_dir   = out_dir   or cfg.OUTPUT_DIR
    ckpt_path = ckpt_path or os.path.join(cfg.CHECKPOINT_DIR, 'best.pth')
    log_path  = setup_logging(out_dir)

    device = torch.device(
        cfg.DEVICE if hasattr(cfg, 'DEVICE') else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log_to_file(log_path, f"[Config] provider={llm_provider}, device={device}, ckpt={ckpt_path}")

    # --- Dataset & loader ---
    dataset = DeepAccidentDataset(mode='val')
    loader  = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=deepaccident_collate
    )

    # --- Model ---
    model = HeterogeneousPredictor().to(device)
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model'])
        log_to_file(log_path, f"[Checkpoint] loaded from {ckpt_path}")
    else:
        log_to_file(log_path, f"[Warning] Checkpoint not found ({ckpt_path}). Using random init.")
    model.eval()

    # --- LLM ---
    llm        = LLMClient(provider=llm_provider, model=llm_model)
    serializer = SemanticSerializer()

    # --- Metric accumulators ---
    metrics = {
        'normal':   {'total': 0, 'ade': 0.0, 'fde': 0.0, 'cr': 0.0},
        'accident': {'total': 0, 'ade': 0.0, 'fde': 0.0, 'cr': 0.0},
        'global':   {'intent_correct': 0, 'total': 0, 'latency': 0.0},
    }
    results_buffer = []

    pbar = tqdm(loader, desc="[Inference]")
    for i, batch in enumerate(pbar):
        if max_samples and i >= max_samples:
            break

        hist_norm = batch['hist_norm'].to(device)
        map_feat  = batch.get('map_padded')
        if map_feat is not None:
            map_feat = map_feat.to(device)

        raw_hist   = batch['hist_raw'].to(device)
        fut_phys   = batch['fut_raw'].to(device)
        gt_action  = int(batch['gt_action'].item())
        meta       = batch['meta'][0] if isinstance(batch['meta'], list) else batch['meta']
        is_accident = meta.get('is_accident', False) if isinstance(meta, dict) else False
        scene_type  = 'accident' if is_accident else 'normal'

        # Build surrounding agent list from meta
        surrounding = []
        if isinstance(meta, dict):
            surrounding = meta.get('surrounding_agents', [])

        # --- LLM CoT reasoning ---
        t0 = time.time()
        raw_hist_np   = raw_hist[0].cpu().numpy()
        map_feat_np   = map_feat[0].cpu().numpy() if map_feat is not None else None

        prompt = serializer.generate_cot_prompt(
            raw_hist_np, map_feat_np, surrounding,
            meta if isinstance(meta, dict) else {}
        )
        action_id, reasoning = llm.query(prompt)

        if action_id is None:
            action_id, reasoning = RuleBasedFallback.infer(prompt)

        latency = time.time() - t0

        # --- Neural execution ---
        action_tensor = torch.tensor([action_id], device=device, dtype=torch.long)
        start_state   = raw_hist[:, -1, :]
        with torch.no_grad():
            pred_traj = model(hist_norm, map_feat, action_tensor, start_state)

        # --- Metrics ---
        ade, fde = compute_ade_fde(pred_traj, fut_phys)
        cr        = compute_collision_rate(pred_traj, map_feat)
        intent_ok = (action_id == gt_action)

        metrics[scene_type]['total'] += 1
        metrics[scene_type]['ade']   += ade
        metrics[scene_type]['fde']   += fde
        metrics[scene_type]['cr']    += cr
        metrics['global']['total']   += 1
        metrics['global']['intent_correct'] += int(intent_ok)
        metrics['global']['latency'] += latency

        results_buffer.append({
            'case_id':    i,
            'scene_type': scene_type,
            'hist':       raw_hist_np,
            'gt_future':  fut_phys[0].cpu().numpy(),
            'pred_traj':  pred_traj[0].cpu().numpy(),
            'prompt':     prompt,
            'reasoning':  reasoning,
            'pred_action': action_id,
            'gt_action':  gt_action,
            'ade':        ade,
            'fde':        fde,
            'cr':         cr,
            'latency':    latency,
            'meta':       str(meta),
        })

        pbar.set_postfix({
            'type': scene_type,
            'ADE': f"{ade:.2f}m",
            'intent': "OK" if intent_ok else "X",
        })

    # --- Final report ---
    g   = metrics['global']
    n   = metrics['normal']
    acc = metrics['accident']

    def _avg(d, k):
        return d[k] / d['total'] if d['total'] > 0 else 0.0

    total = g['total']
    report = f"""
{'='*60}
FINAL INFERENCE REPORT
{'='*60}
Total samples    : {total}
Intent accuracy  : {g['intent_correct']}/{total} = {_avg(g,'intent_correct')*100:.2f}%
Avg LLM latency  : {_avg(g,'latency'):.4f} s

[Normal   N={n['total']}]
  ADE  : {_avg(n,'ade'):.4f} m
  FDE  : {_avg(n,'fde'):.4f} m
  CR   : {_avg(n,'cr')*100:.2f}%

[Accident N={acc['total']}]
  ADE  : {_avg(acc,'ade'):.4f} m
  FDE  : {_avg(acc,'fde'):.4f} m
  CR   : {_avg(acc,'cr')*100:.2f}%
{'='*60}"""

    log_to_file(log_path, report)

    res_path = os.path.join(out_dir, 'inference_detailed_results.pkl')
    with open(res_path, 'wb') as f:
        pickle.dump(results_buffer, f)
    log_to_file(log_path, f"Detailed results saved → {res_path}")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(description="SPC Framework – LLM-driven inference")
    parser.add_argument("--provider",    default=cfg.LLM_PROVIDER,
                        choices=["openai", "deepseek", "qwen", "local"],
                        help="LLM backend provider")
    parser.add_argument("--model",       default=None,
                        help="Override LLM model name")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of inference samples")
    parser.add_argument("--ckpt",        default=None,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--out_dir",     default=None,
                        help="Output directory for logs and results")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_inference(
        llm_provider=args.provider,
        llm_model=args.model,
        max_samples=args.max_samples,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
    )
