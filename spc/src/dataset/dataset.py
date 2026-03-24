# src/dataset/dataset.py
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from config import cfg

# -----------------------------------------------------------------------
# 12-action mapping (mirrors config.py)
# Lateral base: Straight=0, Left=4, Right=8
# Longitudinal offset: Keep=0, Acc=1, Dec=2, Stop=3
# -----------------------------------------------------------------------
_LAT_BASE   = {"straight": 0, "left": 4, "right": 8}
_LON_OFFSET = {"keep": 0, "acc": 1, "dec": 2, "stop": 3}


class DeepAccidentDataset(Dataset):
    """
    DeepAccident dataset loader built on pre-processed pickle output.

    Expected sample keys:
      'hist_states'      : np.ndarray (T_obs, D_hist)
      'future_traj'      : np.ndarray (T_pred, D_fut)
      'map_polylines'    : np.ndarray (L, P, D_map)
      'signal_states'    : dict  {'light': int}
      'meta_action_label': int   (optional, for offline-labelled LLM actions)
      'collision_flag'   : bool  (optional)
      'surrounding_agents': list  (optional)
      'meta'             : dict  (optional)
    """

    def __init__(self, mode: str = 'train'):
        super().__init__()
        self.mode = mode

        # Prefer split-specific file (avoids data leakage between train/val/test)
        split_path = getattr(cfg, 'PROCESSED_DATA_TEMPLATE', '').format(split=mode)
        if split_path and os.path.exists(split_path):
            data_path = split_path
        elif os.path.exists(cfg.PROCESSED_DATA_PATH):
            # Fallback to combined file with a warning
            data_path = cfg.PROCESSED_DATA_PATH
            if mode != 'train':
                print(
                    f"[Warning] Split-specific file not found for mode='{mode}'. "
                    f"Falling back to {data_path} — metrics may reflect training data. "
                    "Run prepare_splits.py with --split val/test to create proper splits."
                )
        else:
            raise FileNotFoundError(
                f"[错误] 找不到处理后的数据 (split={mode}).\n"
                "请先运行 data/preprocess/prepare_splits.py 进行预处理。"
            )

        if not os.path.exists(cfg.SCALER_PATH):
            raise FileNotFoundError(
                f"[错误] 找不到 scalers 文件: {cfg.SCALER_PATH}\n"
                "请先运行 data/preprocess/prepare_splits.py 以生成 scalers。"
            )

        with open(data_path, 'rb') as f:
            self.samples = pickle.load(f)
        with open(cfg.SCALER_PATH, 'rb') as f:
            self.scalers = pickle.load(f)

        self.pos_mean = torch.FloatTensor(np.atleast_1d(self.scalers['pos_mean']))
        self.pos_std  = torch.FloatTensor(np.atleast_1d(self.scalers['pos_std']))
        self.vel_mean = torch.FloatTensor(np.atleast_1d(self.scalers['vel_mean']))
        self.vel_std  = torch.FloatTensor(np.atleast_1d(self.scalers['vel_std']))

    # ------------------------------------------------------------------
    def _normalize(self, traj: torch.Tensor) -> torch.Tensor:
        traj_norm = traj.clone()
        D = traj_norm.shape[1]
        if D >= 2:
            traj_norm[:, 0:2] = (traj[:, 0:2] - self.pos_mean) / (self.pos_std + 1e-8)
        if D >= 4:
            traj_norm[:, 2:4] = (traj[:, 2:4] - self.vel_mean) / (self.vel_std + 1e-8)
        return traj_norm

    # ------------------------------------------------------------------
    def _get_gt_action(self, hist: np.ndarray, fut: np.ndarray) -> int:
        """
        Derive a 12-class meta-action label from trajectory kinematics.

        Lateral:
          - Straight: |avg yaw change| ≤ TURN_THRESH
          - Left:      avg yaw change  >  TURN_THRESH
          - Right:     avg yaw change  < -TURN_THRESH

        Longitudinal (compared to last observed speed):
          - Stop:  final speed < STOP_THRESH
          - Acc:   Δv > +ACC_THRESH
          - Dec:   Δv < -ACC_THRESH
          - Keep:  otherwise
        """
        STOP_THRESH = 0.5   # m/s
        ACC_THRESH  = 1.0   # m/s
        TURN_THRESH = 0.15  # rad cumulative yaw change

        try:
            # Lateral: heading change over the prediction horizon.
            # future_traj format: [x, y, vx, vy] (4-col) or [x,y,vx,vy,ax,ay,yaw] (7-col)
            # IMPORTANT: column 2 is vx, NOT yaw. Derive heading from velocity vector.
            if fut.shape[1] >= 4:
                # heading = arctan2(vy, vx) — correct even for 4-column future_traj
                headings = np.arctan2(fut[:, 3], fut[:, 2] + 1e-6)
                yaw_changes = np.diff(headings)
                avg_yaw = float(np.mean(yaw_changes)) if len(yaw_changes) > 0 else 0.0
            elif fut.shape[1] > 6:
                # 7-column: actual yaw stored in column 6
                yaw_changes = np.diff(fut[:, 6])
                avg_yaw = float(np.mean(yaw_changes)) if len(yaw_changes) > 0 else 0.0
            else:
                avg_yaw = float(np.arctan2(fut[-1, 1], fut[-1, 0])) if fut.shape[0] > 0 else 0.0

            if avg_yaw > TURN_THRESH:
                lateral = "left"
            elif avg_yaw < -TURN_THRESH:
                lateral = "right"
            else:
                lateral = "straight"

            # Longitudinal: speed comparison
            if hist.shape[1] >= 4:
                v_curr = float(np.linalg.norm(hist[-1, 2:4]))
            elif hist.shape[1] >= 3:
                v_curr = float(abs(hist[-1, 2]))
            else:
                v_curr = 0.0

            if fut.shape[1] >= 4:
                v_final = float(np.linalg.norm(fut[-1, 2:4]))
            elif fut.shape[1] >= 3:
                v_final = float(abs(fut[-1, 2]))
            else:
                v_final = 0.0

            if v_final < STOP_THRESH:
                longitudinal = "stop"
            elif v_final - v_curr > ACC_THRESH:
                longitudinal = "acc"
            elif v_curr - v_final > ACC_THRESH:
                longitudinal = "dec"
            else:
                longitudinal = "keep"

        except Exception:
            return 3  # Straight_Stop as safe default

        return _LAT_BASE[lateral] + _LON_OFFSET[longitudinal]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Flexible key lookup — use explicit None checks to avoid treating
        # valid empty numpy arrays as missing (the `or` operator would skip them).
        def _first_not_none(d, *keys):
            for k in keys:
                v = d.get(k)
                if v is not None:
                    return v
            return None

        hist_np = _first_not_none(sample, 'hist_states', 'hist', 'history')
        fut_np  = _first_not_none(sample, 'future_traj', 'future', 'fut')

        if hist_np is None or fut_np is None:
            raise ValueError(
                f"样本 idx={idx} 缺少历史或未来轨迹数据，请检查预处理结果。"
            )

        map_np = _first_not_none(sample, 'map_polylines', 'map') or []
        meta   = _first_not_none(sample, 'meta', 'meta_info') or {}

        hist = torch.FloatTensor(hist_np)
        fut  = torch.FloatTensor(fut_np)

        try:
            if isinstance(map_np, (list, np.ndarray)) and len(map_np) > 0:
                map_feat = torch.FloatTensor(np.array(map_np))
            else:
                map_feat = torch.zeros((1, 1, cfg.MAP_DIM), dtype=torch.float32)
        except Exception:
            map_feat = torch.zeros((1, 1, cfg.MAP_DIM), dtype=torch.float32)

        hist_norm = self._normalize(hist)

        # Prefer offline LLM-assigned label; fall back to kinematic derivation
        if 'meta_action_label' in sample and sample['meta_action_label'] is not None:
            gt_action = int(sample['meta_action_label'])
            # Clamp to valid range
            gt_action = max(0, min(gt_action, cfg.NUM_ACTIONS - 1))
        else:
            gt_action = self._get_gt_action(hist.numpy(), fut.numpy())

        # Enrich meta with collision and surrounding info from sample
        if isinstance(meta, dict):
            meta.setdefault('collision_flag', sample.get('collision_flag', False))
            meta.setdefault('is_accident',    sample.get('collision_flag', False))
            meta.setdefault('surrounding_agents', sample.get('surrounding_agents', []))
            meta.setdefault('signal_states',  sample.get('signal_states', {'light': 2}))

        return {
            'hist_norm': hist_norm,
            'map_feat':  map_feat,
            'fut_raw':   fut,
            'hist_raw':  hist,
            'gt_action': torch.tensor(gt_action, dtype=torch.long),
            'meta':      meta,
        }


# -----------------------------------------------------------------------
# Collate utilities
# -----------------------------------------------------------------------
def compute_line_centers(map_tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(map_tensor, torch.Tensor):
        if map_tensor.numel() == 0:
            return torch.zeros((0, 2), dtype=torch.float32)
        return map_tensor.mean(dim=1)[:, :2]
    else:
        if map_tensor is None or len(map_tensor) == 0:
            return torch.zeros((0, 2), dtype=torch.float32)
        arr = torch.from_numpy(np.array(map_tensor)).float()
        return arr.mean(dim=1)[:, :2]


def pad_maps_to_maxL(map_list, device=None, pad_value: float = 0.0):
    """Pad a list of (L_i, P, D) map tensors to a common L_max."""
    B = len(map_list)
    Ls = [m.shape[0] if (isinstance(m, torch.Tensor) and m.numel() > 0) else 0
          for m in map_list]
    max_L = max(Ls) if Ls else 0
    if max_L == 0:
        return None, None, None

    m0 = next((m for m in map_list if isinstance(m, torch.Tensor) and m.numel() > 0), None)
    P = m0.shape[1] if m0 is not None else 1
    D = m0.shape[2] if m0 is not None else cfg.MAP_DIM

    padded, mask, centers = [], [], []
    for m in map_list:
        dev = torch.device(device) if device is not None else torch.device("cpu")
        if isinstance(m, torch.Tensor) and m.numel() > 0:
            L = m.shape[0]
            m = m.to(dev)
            if L < max_L:
                pad = torch.full((max_L - L, P, D), pad_value, device=dev, dtype=m.dtype)
                m = torch.cat([m, pad], dim=0)
            else:
                m = m[:max_L]
            valid_mask = torch.tensor([1] * min(L, max_L) + [0] * (max_L - L),
                                      dtype=torch.bool, device=dev)
        else:
            m = torch.zeros((max_L, P, D), device=dev)
            valid_mask = torch.zeros((max_L,), dtype=torch.bool, device=dev)

        padded.append(m)
        mask.append(valid_mask)
        centers.append(compute_line_centers(m).to(dev))

    return torch.stack(padded), torch.stack(mask), torch.stack(centers)


def deepaccident_collate(batch: list) -> dict:
    hist_norm  = torch.stack([b['hist_norm']  for b in batch])
    fut_raw    = torch.stack([b['fut_raw']    for b in batch])
    hist_raw   = torch.stack([b['hist_raw']   for b in batch])
    gt_action  = torch.stack([b['gt_action']  for b in batch])
    map_feat   = [b['map_feat'] for b in batch]
    meta       = [b['meta']     for b in batch]
    agent_pos  = torch.stack([h[-1, 0:2] for h in hist_raw])

    padded_maps, map_mask, map_centers = pad_maps_to_maxL(map_feat)
    return {
        'hist_norm':   hist_norm,
        'map_feat':    map_feat,
        'map_padded':  padded_maps,
        'map_mask':    map_mask,
        'map_centers': map_centers,
        'fut_raw':     fut_raw,
        'hist_raw':    hist_raw,
        'gt_action':   gt_action,
        'agent_pos':   agent_pos,
        'meta':        meta,
    }


if __name__ == "__main__":
    ds = DeepAccidentDataset(mode='train')
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=deepaccident_collate)
    for batch in dl:
        print("hist_norm  :", batch['hist_norm'].shape)
        print("fut_raw    :", batch['fut_raw'].shape)
        print("gt_action  :", batch['gt_action'])
        break
