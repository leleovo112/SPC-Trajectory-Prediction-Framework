# data/preprocess/prepare_splits.py
"""
DeepAccidentProcessor: orchestrates the full preprocessing pipeline.

For production use:
  - Calls data_process.scan_deepaccident_structure + process_scene_to_samples
    to convert raw .txt annotation files into training samples.

For debugging (when the raw dataset is unavailable):
  - Generates physically consistent synthetic samples (12-action coverage)
    that exercise the full training pipeline.

Usage:
  python data/preprocess/prepare_splits.py [--split train] [--out_dir data/processed]
"""
import argparse
import os
import pickle
import sys

import numpy as np
from glob import glob
from tqdm import tqdm

try:
    from config import cfg
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from config import cfg

from data.preprocess.data_process import (
    scan_deepaccident_structure,
    process_scene_to_samples,
    generate_intent_label,
)

# -----------------------------------------------------------------------
# 12-action label → physical motion parameters
# -----------------------------------------------------------------------
_ACTION_PARAMS = {
    # (lateral_yaw_rate rad/s, longitudinal_accel m/s², stop)
    0:  (0.0,   0.0,  False),   # Straight_Keep
    1:  (0.0,   1.5,  False),   # Straight_Acc
    2:  (0.0,  -2.0,  False),   # Straight_Dec
    3:  (0.0,  -5.0,  True),    # Straight_Stop
    4:  (0.25,  0.0,  False),   # Left_Keep
    5:  (0.25,  1.5,  False),   # Left_Acc
    6:  (0.25, -2.0,  False),   # Left_Dec
    7:  (0.25, -5.0,  True),    # Left_Stop
    8:  (-0.25, 0.0,  False),   # Right_Keep
    9:  (-0.25, 1.5,  False),   # Right_Acc
    10: (-0.25,-2.0,  False),   # Right_Dec
    11: (-0.25,-5.0,  True),    # Right_Stop
}


class DeepAccidentProcessor:
    """Preprocessing orchestrator for the DeepAccident dataset."""

    def __init__(self):
        self.data_root = cfg.DATA_ROOT
        self.scalers   = {
            'pos_mean': np.zeros(2, dtype=np.float32),
            'pos_std':  np.ones(2,  dtype=np.float32),
            'vel_mean': np.zeros(2, dtype=np.float32),
            'vel_std':  np.ones(2,  dtype=np.float32),
        }

    def _data_available(self) -> bool:
        return os.path.isdir(self.data_root)

    @staticmethod
    def _gen_traj(times: np.ndarray, v0: float, yaw_rate: float,
                  accel: float, is_stop: bool) -> np.ndarray:
        """Generate a physically consistent 7-D trajectory array."""
        traj = np.zeros((len(times), 7), dtype=np.float32)
        for k, t in enumerate(times):
            v = max(0.0, v0 + accel * t) if not is_stop else max(0.0, v0 - 5.0 * t)
            theta = yaw_rate * t
            x = (v0 / max(abs(yaw_rate), 1e-4)) * np.sin(theta) if abs(yaw_rate) > 1e-3 \
                else v0 * t + 0.5 * accel * t**2
            y = (v0 / max(abs(yaw_rate), 1e-4)) * (1 - np.cos(theta)) if abs(yaw_rate) > 1e-3 \
                else 0.0
            vx = v * np.cos(theta)
            vy = v * np.sin(theta)
            ax = accel * np.cos(theta)
            ay = accel * np.sin(theta)
            traj[k] = [x, y, vx, vy, ax, ay, theta]
        return traj

    def _generate_mock_samples(self, n: int = 300) -> list:
        """
        Produce `n` physically consistent synthetic samples
        with balanced 12-action coverage.
        """
        samples = []
        T_obs  = cfg.OBS_LEN
        T_pred = cfg.PRED_LEN
        t_hist = np.linspace(-T_obs * cfg.DT, -cfg.DT, T_obs)
        t_fut  = np.linspace(0.0, (T_pred - 1) * cfg.DT, T_pred)

        for idx in range(n):
            action_id = idx % cfg.NUM_ACTIONS   # balanced coverage
            yaw_rate, accel, is_stop = _ACTION_PARAMS[action_id]
            v0 = 0.0 if is_stop else np.random.uniform(5.0, 15.0)

            hist = self._gen_traj(t_hist, v0, yaw_rate, accel, is_stop)
            fut  = self._gen_traj(t_fut,  v0, yaw_rate, accel, is_stop)

            # Small Gaussian noise for data augmentation
            hist[:, :2] += np.random.randn(*hist[:, :2].shape).astype(np.float32) * 0.05
            fut[:, :2]  += np.random.randn(*fut[:, :2].shape).astype(np.float32) * 0.05

            # Shift so that the last observed hist frame is at the origin
            # (ego-centric, consistent with what process_scene_to_samples produces)
            anchor_x, anchor_y = float(hist[-1, 0]), float(hist[-1, 1])
            hist[:, 0] -= anchor_x;  hist[:, 1] -= anchor_y
            fut[:, 0]  -= anchor_x;  fut[:, 1]  -= anchor_y

            map_feat = np.zeros(
                (cfg.MAP_MAX_LINES, cfg.MAP_POINTS_PER_LINE, cfg.MAP_DIM), dtype=np.float32
            )
            # Synthetic lane: straight lane along x-axis
            for l_idx in range(min(5, cfg.MAP_MAX_LINES)):
                for p_idx in range(cfg.MAP_POINTS_PER_LINE):
                    map_feat[l_idx, p_idx, 0] = p_idx * 2.0       # x spacing
                    map_feat[l_idx, p_idx, 1] = (l_idx - 2) * 3.5  # y lane offset

            # Verify label consistency
            derived_label = generate_intent_label(hist, fut)
            # Use action_id for mock data (ground-truth); derived_label is for validation
            is_accident = action_id in {3, 6, 7, 10, 11}

            samples.append({
                'hist_states':       hist.astype(np.float32),
                'future_traj':       fut[:, :4].astype(np.float32),
                'map_polylines':     map_feat,
                'signal_states':     {'light': 2},
                'meta_action_label': action_id,
                'collision_flag':    False,
                'surrounding_agents': [],
                'meta': {
                    'id':         idx,
                    'is_accident': is_accident,
                    'signal':     {'light': 2},
                },
            })

        return samples

    def run(self, split: str = 'train', out_dir: str = None, max_samples: int = None):
        out_dir = out_dir or os.path.dirname(cfg.PROCESSED_DATA_PATH)
        os.makedirs(out_dir, exist_ok=True)

        all_samples = []

        # --- Try real dataset first ---
        if self._data_available():
            try:
                scenes = scan_deepaccident_structure(self.data_root, split)
                for scene in tqdm(scenes, desc=f"[{split}] Parsing scenes"):
                    if max_samples and len(all_samples) >= max_samples:
                        break
                    all_samples.extend(process_scene_to_samples(scene))
                print(f"[Data] Parsed {len(all_samples)} real samples for split='{split}'")
            except Exception as e:
                print(f"[Warning] Real data processing failed: {e}")
                all_samples = []

        # --- Fall back to synthetic data ---
        if len(all_samples) == 0:
            print(
                f"[Warning] No real samples parsed for '{split}'. "
                "Generating synthetic debug data (300 samples)."
            )
            n_mock = min(max_samples, 300) if max_samples else 300
            all_samples = self._generate_mock_samples(n=n_mock)

        if max_samples:
            all_samples = all_samples[:max_samples]

        # --- Compute normalisation scalers (only from train split to avoid leakage) ---
        print("[Data] Computing normalisation statistics …")
        all_hist = np.stack([s['hist_states'] for s in all_samples])
        self.scalers = {
            'pos_mean': np.mean(all_hist[..., 0:2], axis=(0, 1)).astype(np.float32),
            'pos_std':  (np.std(all_hist[..., 0:2], axis=(0, 1)) + 1e-5).astype(np.float32),
            'vel_mean': np.mean(all_hist[..., 2:4], axis=(0, 1)).astype(np.float32),
            'vel_std':  (np.std(all_hist[..., 2:4], axis=(0, 1)) + 1e-5).astype(np.float32),
        }

        # --- Resolve split-specific output path ---
        template = getattr(cfg, 'PROCESSED_DATA_TEMPLATE', '')
        if template and '{split}' in template:
            data_path = template.format(split=split)
        else:
            data_path = cfg.PROCESSED_DATA_PATH  # legacy fallback
        scaler_path = cfg.SCALER_PATH

        os.makedirs(os.path.dirname(os.path.abspath(data_path)),   exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(scaler_path)), exist_ok=True)

        with open(data_path, 'wb') as f:
            pickle.dump(all_samples, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers, f)

        print(
            f"[Data] Saved {len(all_samples)} samples ({split}) → {data_path}\n"
            f"[Data] Saved scalers → {scaler_path}"
        )
        return all_samples


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepAccident – prepare training splits")
    parser.add_argument("--data_root",   default=cfg.DATA_ROOT)
    parser.add_argument("--out_dir",     default=os.path.dirname(cfg.PROCESSED_DATA_PATH))
    parser.add_argument("--split",       default="train", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    processor = DeepAccidentProcessor()
    processor.data_root = args.data_root
    processor.run(split=args.split, out_dir=args.out_dir, max_samples=args.max_samples)
