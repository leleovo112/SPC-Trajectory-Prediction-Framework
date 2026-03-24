# train.py
"""
Training entry point for the SPC neuro-symbolic trajectory prediction framework.

Usage:
  python train.py [--cfg configs/train.yaml]
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from data.preprocess.prepare_splits import DeepAccidentProcessor
from src.dataset.dataset import DeepAccidentDataset, deepaccident_collate
from src.models.predictor import HeterogeneousPredictor
from src.models.safety_loss import SafetyLoss


# -----------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------
def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def apply_yaml_overrides(yaml_cfg: dict):
    """Patch the global cfg object with values from the YAML file."""
    mapping = {
        ('model', 'd_model'):      ('HIDDEN_DIM',   int),
        ('model', 'intent_dim'):   ('INTENT_DIM',   int),
        ('model', 'n_actions'):    ('NUM_ACTIONS',  int),
        ('data',  'root'):         ('DATA_ROOT',    str),
        ('train', 'batch_size'):   ('BATCH_SIZE',   int),
        ('train', 'epochs'):       ('EPOCHS',       int),
        ('train', 'lr'):           ('LEARNING_RATE', float),
        ('train', 'device'):       ('DEVICE',       str),
        ('train', 'num_workers'):  ('NUM_WORKERS',  int),
        ('train', 'grad_clip'):    ('GRAD_CLIP',    float),
        ('llm',   'provider'):     ('LLM_PROVIDER', str),
        ('llm',   'model'):        None,          # handled separately
        ('llm',   'temperature'):  ('LLM_TEMPERATURE', float),
        ('llm',   'max_tokens'):   ('LLM_MAX_TOKENS',  int),
        ('logging', 'save_ckpt_dir'): ('CHECKPOINT_DIR', str),
        ('logging', 'output_dir'):    ('OUTPUT_DIR',     str),
        ('logging', 'llm_cache_dir'): ('LLM_CACHE_DIR',  str),
    }
    for (section, key), target in mapping.items():
        if section in yaml_cfg and isinstance(yaml_cfg[section], dict):
            val = yaml_cfg[section].get(key)
            if val is not None and target is not None:
                attr_name, cast = target
                setattr(cfg, attr_name, cast(val))

    # Loss weights
    if 'loss_weights' in yaml_cfg:
        for k, v in yaml_cfg['loss_weights'].items():
            cfg.LOSS_WEIGHTS[k] = float(v)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# -----------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------
def ensure_processed_data():
    template = getattr(cfg, 'PROCESSED_DATA_TEMPLATE', '')
    train_path = template.format(split='train') if '{split}' in template else cfg.PROCESSED_DATA_PATH
    val_path   = template.format(split='val')   if '{split}' in template else cfg.PROCESSED_DATA_PATH

    if (not os.path.exists(train_path)
            or not os.path.exists(val_path)
            or not os.path.exists(cfg.SCALER_PATH)):
        print("[Data] Processed data not found – running preprocessor …")
        out_dir = os.path.dirname(os.path.abspath(cfg.PROCESSED_DATA_PATH))
        processor = DeepAccidentProcessor()
        processor.run(split='train', out_dir=out_dir)
        processor.run(split='val',   out_dir=out_dir)


# -----------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for batch in val_loader:
            hist_norm  = batch['hist_norm'].to(device)
            map_padded = batch.get('map_padded')
            if map_padded is not None:
                map_padded = map_padded.to(device)
            gt_action = batch['gt_action'].to(device)
            fut_raw   = batch['fut_raw'].to(device)
            hist_raw  = batch['hist_raw'].to(device)

            start_state = hist_raw[:, -1, :]
            pred = model(hist_norm, map_padded, gt_action, start_state)
            loss, _ = criterion(pred, fut_raw, gt_action, map_padded)
            total_loss += loss.item()
            n_batches  += 1
    return total_loss / max(1, n_batches)


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------
def main(cfg_path: str = "configs/train.yaml"):
    # 1. Load & apply YAML overrides
    if os.path.exists(cfg_path):
        yaml_cfg = load_yaml(cfg_path)
        apply_yaml_overrides(yaml_cfg)
        print(f"[Config] Loaded YAML overrides from {cfg_path}")

    # 2. Deterministic seed (fixed to 42 per paper)
    seed = getattr(cfg, 'SEED', 42)
    set_seed(seed)
    print(f"[Config] Random seed = {seed}")

    device = torch.device(
        cfg.DEVICE if hasattr(cfg, 'DEVICE')
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Config] Device = {device}")

    # 3. Data — use split-specific datasets to avoid data leakage
    ensure_processed_data()
    train_ds = DeepAccidentDataset(mode='train')
    val_ds   = DeepAccidentDataset(mode='val')

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=deepaccident_collate,
        pin_memory=True, persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=deepaccident_collate,
    )
    print(f"[Data] train={len(train_ds)}, val={len(val_ds)}")

    # 4. Model & loss
    model     = HeterogeneousPredictor().to(device)
    criterion = SafetyLoss().to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {param_count:,}")

    # 5. Optimiser (AdamW with weight decay per paper: lr=1e-4, wd=0.01)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=getattr(cfg, 'WEIGHT_DECAY', 0.01),
    )

    # 6. Cosine LR scheduler
    epochs    = cfg.EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 7. Checkpoint dir
    ckpt_dir = cfg.CHECKPOINT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val  = float('inf')
    grad_clip = getattr(cfg, 'GRAD_CLIP', 1.0)

    print(f"[Train] Starting {epochs} epochs …")
    for epoch in range(epochs):
        model.train()
        epoch_loss   = 0.0
        n_steps      = 0
        t0           = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=120)
        for batch in pbar:
            hist_norm  = batch['hist_norm'].to(device)
            map_padded = batch.get('map_padded')
            if map_padded is not None:
                map_padded = map_padded.to(device)
            gt_action   = batch['gt_action'].to(device)
            fut_raw     = batch['fut_raw'].to(device)
            hist_raw    = batch['hist_raw'].to(device)
            start_state = hist_raw[:, -1, :]

            pred = model(hist_norm, map_padded, gt_action, start_state)
            loss, loss_dict = criterion(pred, fut_raw, gt_action, map_padded)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps    += 1
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'reg':  f"{loss_dict.get('reg', 0):.4f}",
                'int':  f"{loss_dict.get('intent', 0):.4f}",
            })

        scheduler.step()
        avg_train = epoch_loss / max(1, n_steps)
        val_loss  = validate(model, val_loader, criterion, device)
        elapsed   = time.time() - t0
        lr_now    = scheduler.get_last_lr()[0]

        print(f"[Epoch {epoch+1:03d}] "
              f"train={avg_train:.4f}  val={val_loss:.4f}  "
              f"lr={lr_now:.2e}  t={elapsed:.1f}s")

        # Save latest checkpoint
        latest = os.path.join(ckpt_dir, "latest.pth")
        torch.save({
            'epoch':     epoch,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss':  val_loss,
        }, latest)

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            best = os.path.join(ckpt_dir, "best.pth")
            torch.save({
                'epoch':    epoch,
                'model':    model.state_dict(),
                'val_loss': val_loss,
            }, best)
            print(f"  ✔ New best model → {best}  (val_loss={val_loss:.4f})")

    print(f"[Done] Best val_loss = {best_val:.4f}")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(description="SPC Framework – Training")
    parser.add_argument("--cfg", default="configs/train.yaml",
                        help="Path to YAML configuration file")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(cfg_path=args.cfg)
