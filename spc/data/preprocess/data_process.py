# data/preprocess/data_process.py
"""
DeepAccident data processing utilities.

Key functions:
  scan_deepaccident_structure  – walk the raw dataset directory tree
  parse_label_file             – parse a single annotation .txt frame
  process_scene_to_samples     – sliding-window sample extraction from a scene
  generate_intent_label        – 12-class meta-action label from kinematics
"""
import os
import sys
import pickle
import argparse
from collections import Counter
from glob import glob
from tqdm import tqdm

import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from config import cfg
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from config import cfg

# Module-level constants from config
OBS_LEN              = cfg.OBS_LEN
PRED_LEN             = cfg.PRED_LEN
DATA_ROOT            = cfg.DATA_ROOT
MAP_MAX_LINES        = cfg.MAP_MAX_LINES
MAP_POINTS_PER_LINE  = cfg.MAP_POINTS_PER_LINE
MAP_DIM              = cfg.MAP_DIM


def info(*args):
    print(f"\033[94m[INFO]\033[0m {' '.join(map(str, args))}")


def warn(*args):
    print(f"\033[93m[WARN]\033[0m {' '.join(map(str, args))}")


# -----------------------------------------------------------------------
# 1. Directory scanner
# -----------------------------------------------------------------------
def scan_deepaccident_structure(root_dir: str, split: str) -> list:
    """
    Walk the DeepAccident directory layout:
      <split>/<category>/<role>/label/<scene_dir>/<frame>.txt

    Returns a list of dicts:
      {'frame_paths': [...], 'scene_id': str, 'category': str}
    """
    split_path = os.path.join(root_dir, split)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split path not found: {split_path}")

    info(f"Scanning: {split_path}")
    needed = OBS_LEN + PRED_LEN
    result = []

    for category in os.listdir(split_path):
        cat_path = os.path.join(split_path, category)
        if not os.path.isdir(cat_path):
            continue

        for role in ['ego_vehicle']:
            label_root = os.path.join(cat_path, role, "label")
            if not os.path.exists(label_root):
                continue

            for scene_dir in os.listdir(label_root):
                scene_path = os.path.join(label_root, scene_dir)
                if not os.path.isdir(scene_path):
                    continue
                frames = sorted(
                    [f for f in os.listdir(scene_path) if f.endswith('.txt')]
                )
                if len(frames) >= needed:
                    result.append({
                        'frame_paths': [os.path.join(scene_path, f) for f in frames],
                        'scene_id':   scene_dir,
                        'category':   category,
                    })

    info(f"Found {len(result)} valid scene sequences (≥ {needed} frames)")
    return result


# -----------------------------------------------------------------------
# 2. Single-frame label parser
# -----------------------------------------------------------------------
def parse_label_file(label_path: str) -> dict:
    """
    Parse one annotation .txt frame.

    Expected line format (space-separated):
      type x y z l w h yaw vx vy id lane_id role collision traffic_light ...

    Returns:
      {'timestamp': float, 'vehicles': list[dict]}
    """
    vehicles  = []
    raw_lines = []
    try:
        with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
            for ln in f:
                s = ln.strip()
                if s:
                    raw_lines.append(s)
    except Exception as e:
        warn(f"Cannot open {label_path}: {e}")
        return {'timestamp': 0.0, 'vehicles': [], 'raw_lines': []}

    # First line may be a timestamp scalar
    timestamp       = 0.0
    vehicle_lines   = raw_lines
    if raw_lines and len(raw_lines[0].split()) == 1:
        try:
            timestamp     = float(raw_lines[0])
            vehicle_lines = raw_lines[1:]
        except ValueError:
            pass

    for line in vehicle_lines:
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            role_str = parts[12].lower() if len(parts) > 12 else ""
            v = {
                'type':          parts[0],
                'x':             float(parts[1]),
                'y':             float(parts[2]),
                'z':             float(parts[3]) if len(parts) > 3 else 0.0,
                'length':        float(parts[4]) if len(parts) > 4 else 4.5,
                'width':         float(parts[5]) if len(parts) > 5 else 2.0,
                'yaw':           float(parts[7])  if len(parts) > 7  else 0.0,
                'vx':            float(parts[8])  if len(parts) > 8  else 0.0,
                'vy':            float(parts[9])  if len(parts) > 9  else 0.0,
                'id':            int(float(parts[10])) if len(parts) > 10 else -1,
                'lane_id':       int(float(parts[11])) if len(parts) > 11 else -1,
                'is_ego':        'ego' in role_str,
                'collision':     (parts[13].lower() == "true") if len(parts) > 13 else False,
                'traffic_light': (parts[14].lower()
                                  if len(parts) > 14
                                  and parts[14].lower() in ("red", "yellow", "green")
                                  else None),
            }
            vehicles.append(v)
        except Exception as e:
            warn(f"Parse error in '{label_path}', line '{line[:60]}': {e}")

    return {'timestamp': timestamp, 'vehicles': vehicles, 'raw_lines': raw_lines}


# -----------------------------------------------------------------------
# 3. Extract ego trajectory from a sequence of parsed frames
# -----------------------------------------------------------------------
def extract_ego_trajectory(frames: list, obs_len: int = OBS_LEN,
                            pred_len: int = PRED_LEN):
    """
    Build (hist, fut) from a list of frame dicts, each produced by parse_label_file.
    Each ego state: [x, y, vx, vy, ax, ay, yaw]  (7-D)
    """
    ego_states = []
    for frame in frames:
        vehicles = frame.get('vehicles', [])
        ego = next((v for v in vehicles if v.get('is_ego')), None)
        if ego is None:
            # Fallback: most common vehicle ID
            ids = [v['id'] for v in vehicles if v['id'] != -1]
            if ids:
                common_id = Counter(ids).most_common(1)[0][0]
                ego = next((v for v in vehicles if v['id'] == common_id), vehicles[0] if vehicles else None)
        if ego is None:
            continue
        ego_states.append([
            ego['x'], ego['y'],
            ego['vx'], ego['vy'],
            0.0, 0.0,   # ax, ay placeholder (will be computed below)
            ego['yaw'],
        ])

    traj = np.array(ego_states, dtype=np.float32)
    if len(traj) < obs_len + pred_len:
        return None, None

    # Compute approximate acceleration via finite differences
    if len(traj) >= 2:
        vel = traj[:, 2:4]
        acc = np.zeros_like(vel)
        acc[1:] = (vel[1:] - vel[:-1]) / cfg.DT
        traj[:, 4:6] = acc

    return traj[:obs_len], traj[obs_len: obs_len + pred_len]


# -----------------------------------------------------------------------
# 4. Extract surrounding agents
# -----------------------------------------------------------------------
def extract_surrounding_agents(frame: dict, ego_pose: tuple,
                                radius: float = 50.0) -> list:
    """
    Returns a list of nearby agent dicts (within `radius` metres).
    ego_pose = (x, y, yaw)
    """
    ex, ey, _ = ego_pose
    agents = []
    for v in frame.get('vehicles', []):
        if v.get('is_ego'):
            continue
        dx   = v['x'] - ex
        dy   = v['y'] - ey
        dist = float(np.hypot(dx, dy))
        if dist > radius:
            continue
        speed = float(np.hypot(v['vx'], v['vy'])) * 3.6
        # Rough direction using ego-centric y-axis
        direction = "ahead" if dy > 0 else "behind"
        agents.append({
            'id':       v['id'],
            'type':     v.get('type', 'vehicle'),
            'rel_dist': dist,
            'rel_dir':  direction,
            'v_kph':    speed,
            'action':   "moving" if speed > 1.0 else "stopped",
        })
    return agents


# -----------------------------------------------------------------------
# 5. Collision and signal flags
# -----------------------------------------------------------------------
def parse_collision_and_signals(frame: dict) -> tuple:
    """Returns (collision_flag: bool, signal_states: dict)."""
    collision = any(v.get('collision', False) for v in frame.get('vehicles', []))
    signal = {'light': 2}  # default = green
    for v in frame.get('vehicles', []):
        tl = v.get('traffic_light')
        if tl == 'red':
            signal['light'] = 0
            break
        elif tl == 'yellow':
            signal['light'] = 1
    return collision, signal


# -----------------------------------------------------------------------
# 6. 12-class intent label derivation (Lateral × Longitudinal)
# -----------------------------------------------------------------------
_LAT_BASE   = {"straight": 0, "left": 4, "right": 8}
_LON_OFFSET = {"keep": 0, "acc": 1, "dec": 2, "stop": 3}


def generate_intent_label(hist: np.ndarray, fut: np.ndarray) -> int:
    """
    Derives the 12-class meta-action ID from trajectory kinematics.

    Lateral  (yaw change over future horizon):
      straight  | yaw_delta | ≤ 0.1 rad
      left       yaw_delta   > 0.1 rad
      right      yaw_delta   < -0.1 rad

    Longitudinal (speed change hist[-1] → fut[-1]):
      stop   v_final < 0.5 m/s
      acc    Δv > +0.5 m/s
      dec    Δv < -0.5 m/s
      keep   otherwise
    """
    TURN_THRESH = 0.1   # rad
    STOP_THRESH = 0.5   # m/s
    VEL_THRESH  = 0.5   # m/s delta

    # --- Lateral ---
    if fut is None or len(fut) == 0:
        return _LAT_BASE["straight"] + _LON_OFFSET["stop"]

    try:
        # Heading from velocity (correct for 4-col future [x,y,vx,vy])
        # IMPORTANT: column 2 is vx NOT yaw — always derive heading from velocity.
        if fut.shape[1] >= 4:
            headings = np.arctan2(fut[:, 3], fut[:, 2] + 1e-6)
            yaw_changes = np.diff(headings)
        elif fut.shape[1] > 6:
            yaw_changes = np.diff(fut[:, 6])
        else:
            yaw_changes = np.array([0.0])
        avg_yaw = float(np.mean(yaw_changes)) if len(yaw_changes) > 0 else 0.0
    except Exception:
        avg_yaw = 0.0

    if avg_yaw > TURN_THRESH:
        lateral = "left"
    elif avg_yaw < -TURN_THRESH:
        lateral = "right"
    else:
        lateral = "straight"

    # --- Longitudinal ---
    try:
        if hist.shape[1] >= 4:
            v_curr  = float(np.linalg.norm(hist[-1, 2:4]))
        elif hist.shape[1] >= 3:
            v_curr  = float(abs(hist[-1, 2]))
        else:
            v_curr  = 0.0

        if fut.shape[1] >= 4:
            v_final = float(np.linalg.norm(fut[-1, 2:4]))
        elif fut.shape[1] >= 3:
            v_final = float(abs(fut[-1, 2]))
        else:
            v_final = 0.0
    except Exception:
        v_curr = v_final = 0.0

    if v_final < STOP_THRESH:
        longitudinal = "stop"
    elif v_final - v_curr > VEL_THRESH:
        longitudinal = "acc"
    elif v_curr - v_final > VEL_THRESH:
        longitudinal = "dec"
    else:
        longitudinal = "keep"

    return _LAT_BASE[lateral] + _LON_OFFSET[longitudinal]


# -----------------------------------------------------------------------
# 6b. Ego-centric coordinate transform
# -----------------------------------------------------------------------
def _to_ego_centric(traj: np.ndarray, anchor_x: float, anchor_y: float,
                    anchor_yaw: float) -> np.ndarray:
    """
    Transform a trajectory from world coordinates to the ego-centric frame
    defined by (anchor_x, anchor_y, anchor_yaw) — the last observed ego pose.

    Columns transformed:
      [0,1] → x,y positions  (rotation + translation)
      [2,3] → vx,vy          (rotation only, magnitudes preserved)
      [4,5] → ax,ay          (rotation only)
      [6]   → yaw            (subtract anchor_yaw, wrap to [-π,π])
    """
    out = traj.copy().astype(np.float32)
    cos_h = np.cos(-anchor_yaw)
    sin_h = np.sin(-anchor_yaw)
    R = np.array([[cos_h, -sin_h],
                  [sin_h,  cos_h]], dtype=np.float32)

    # Translate then rotate positions
    dx = out[:, 0] - anchor_x
    dy = out[:, 1] - anchor_y
    pos_rotated = R @ np.stack([dx, dy])          # (2, T)
    out[:, 0] = pos_rotated[0]
    out[:, 1] = pos_rotated[1]

    # Rotate velocity
    if traj.shape[1] >= 4:
        vel_rotated = R @ np.stack([out[:, 2], out[:, 3]])
        out[:, 2] = vel_rotated[0]
        out[:, 3] = vel_rotated[1]

    # Rotate acceleration
    if traj.shape[1] >= 6:
        acc_rotated = R @ np.stack([out[:, 4], out[:, 5]])
        out[:, 4] = acc_rotated[0]
        out[:, 5] = acc_rotated[1]

    # Wrap yaw
    if traj.shape[1] >= 7:
        out[:, 6] = (out[:, 6] - anchor_yaw + np.pi) % (2 * np.pi) - np.pi

    return out


# -----------------------------------------------------------------------
# 7. Scene → samples (sliding window)
# -----------------------------------------------------------------------
def process_scene_to_samples(scene_info: dict, stride: int = 5) -> list:
    """
    Parse an entire scene sequence and extract overlapping training samples
    using a sliding window of size OBS_LEN + PRED_LEN with step `stride`.

    Each sample dict:
      hist_states, future_traj, map_polylines, signal_states,
      meta_action_label, collision_flag, surrounding_agents, meta
    """
    frame_paths = scene_info['frame_paths']
    all_frames  = [parse_label_file(p) for p in frame_paths]
    all_frames  = [f for f in all_frames if f is not None]

    needed = OBS_LEN + PRED_LEN
    if len(all_frames) < needed:
        return []

    # Build continuous ego trajectory for the full scene
    ego_states = []
    for frame in all_frames:
        veh = frame.get('vehicles', [])
        ego = next((v for v in veh if v.get('is_ego')), None)
        if ego is None and veh:
            ids = [v['id'] for v in veh if v['id'] != -1]
            if ids:
                cid = Counter(ids).most_common(1)[0][0]
                ego = next((v for v in veh if v['id'] == cid), veh[0])
        if ego is None:
            ego_states.append(None)
            continue
        # compute acc via finite diff later
        ego_states.append([
            ego['x'], ego['y'],
            ego['vx'], ego['vy'],
            0.0, 0.0,
            ego['yaw'],
        ])

    # Filter out frames with missing ego (keep continuity)
    valid_indices = [i for i, s in enumerate(ego_states) if s is not None]
    if len(valid_indices) < needed:
        return []

    traj_full = np.array([ego_states[i] for i in valid_indices], dtype=np.float32)
    frames_valid = [all_frames[i] for i in valid_indices]

    # Compute acceleration columns
    vel = traj_full[:, 2:4]
    acc = np.zeros_like(vel)
    acc[1:] = (vel[1:] - vel[:-1]) / cfg.DT
    traj_full[:, 4:6] = acc

    samples = []
    total_len = len(traj_full)

    for start in range(0, total_len - needed + 1, stride):
        hist = traj_full[start: start + OBS_LEN]
        fut  = traj_full[start + OBS_LEN: start + needed]
        anchor_frame = frames_valid[start + OBS_LEN - 1]

        # Ego pose at the prediction start (= last observed frame)
        anchor_x   = float(hist[-1, 0])
        anchor_y   = float(hist[-1, 1])
        anchor_yaw = float(hist[-1, 6])
        ego_pose   = (anchor_x, anchor_y, anchor_yaw)

        # --- Transform to ego-centric coordinates (meters from ego, ego-aligned) ---
        # This is critical: world coordinates cause numerical instability in the
        # regression loss and break the distance-based interaction mask.
        hist = _to_ego_centric(hist, anchor_x, anchor_y, anchor_yaw)
        fut  = _to_ego_centric(fut,  anchor_x, anchor_y, anchor_yaw)

        surrounding = extract_surrounding_agents(anchor_frame, ego_pose)
        collision, signal = parse_collision_and_signals(anchor_frame)
        # Intent label is computed AFTER ego-centric transform (velocity angles unchanged)
        intent_label = generate_intent_label(hist, fut)

        is_accident = scene_info.get('category', '').lower().find('accident') >= 0

        samples.append({
            'hist_states':      hist.astype(np.float32),
            'future_traj':      fut[:, :4].astype(np.float32),   # [x,y,vx,vy]
            'map_polylines':    np.zeros(
                (MAP_MAX_LINES, MAP_POINTS_PER_LINE, MAP_DIM), dtype=np.float32
            ),
            'signal_states':    signal,
            'meta_action_label': intent_label,
            'collision_flag':   collision,
            'surrounding_agents': surrounding,
            'meta': {
                'scene':      scene_info['scene_id'],
                'category':   scene_info['category'],
                'is_accident': is_accident,
                'signal':     signal,
            },
        })

    return samples


# -----------------------------------------------------------------------
# 8. CLI entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepAccident data preprocessing")
    parser.add_argument("--data_root",   default=cfg.DATA_ROOT)
    parser.add_argument("--out_path",    default=cfg.PROCESSED_DATA_PATH,
                        help="Output pickle path")
    parser.add_argument("--scaler_path", default=cfg.SCALER_PATH)
    parser.add_argument("--split",       default="train")
    parser.add_argument("--stride",      type=int, default=5)
    parser.add_argument("--max_scenes",  type=int, default=None)
    args = parser.parse_args()

    scenes = scan_deepaccident_structure(args.data_root, args.split)
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    all_samples = []
    for scene in tqdm(scenes, desc="Processing scenes"):
        all_samples.extend(process_scene_to_samples(scene, stride=args.stride))

    info(f"Total samples: {len(all_samples)}")

    # Compute scalers
    if all_samples:
        all_hist = np.stack([s['hist_states'] for s in all_samples])
        scalers  = {
            'pos_mean': np.mean(all_hist[..., 0:2], axis=(0, 1)),
            'pos_std':  np.std(all_hist[...,  0:2], axis=(0, 1)) + 1e-5,
            'vel_mean': np.mean(all_hist[..., 2:4], axis=(0, 1)),
            'vel_std':  np.std(all_hist[...,  2:4], axis=(0, 1)) + 1e-5,
        }
        os.makedirs(os.path.dirname(args.out_path),    exist_ok=True)
        os.makedirs(os.path.dirname(args.scaler_path), exist_ok=True)
        with open(args.out_path,    'wb') as f:
            pickle.dump(all_samples, f)
        with open(args.scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        info(f"Saved {len(all_samples)} samples → {args.out_path}")
        info(f"Saved scalers → {args.scaler_path}")
    else:
        warn("No samples produced. Check dataset path and structure.")
