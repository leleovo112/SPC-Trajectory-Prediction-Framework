# src/models/safety_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg

# -----------------------------------------------------------------------
# 12-action group indices (kept as tuples for tensor operations)
# Straight: 0-3 | Left: 4-7 | Right: 8-11
# *_Keep=+0  *_Acc=+1  *_Dec=+2  *_Stop=+3
# -----------------------------------------------------------------------
_STOP_IDS  = (3, 7, 11)
_DEC_IDS   = (2, 6, 10)
_ACC_IDS   = (1, 5, 9)
_LEFT_IDS  = (4, 5, 6, 7)
_RIGHT_IDS = (8, 9, 10, 11)


def _mask(action_ids: torch.Tensor, ids: tuple) -> torch.Tensor:
    """Return boolean mask where action_ids ∈ ids."""
    m = torch.zeros_like(action_ids, dtype=torch.bool)
    for v in ids:
        m |= (action_ids == v)
    return m


class SafetyLoss(nn.Module):
    """
    Composite loss implementing the Semantic-Physical Verification Rules (Eq. 14-19):
      L_total = L_reg + λ_int·L_intent + λ_phy·L_phy + λ_scene·L_scene
    """

    def __init__(self):
        super().__init__()
        self.reg_loss_fn = nn.SmoothL1Loss(reduction='none')
        self.weights = getattr(cfg, 'LOSS_WEIGHTS', {'reg': 1.0, 'intent': 0.5, 'phy': 0.1, 'scene': 0.1})
        self.max_acc        = getattr(cfg, 'MAX_ACC',    8.0)
        self.max_jerk       = getattr(cfg, 'MAX_JERK',  20.0)
        self.max_angular_vel = getattr(cfg, 'MAX_ANG_VEL', 0.6)
        self.scene_sample_points = getattr(cfg, 'SCENE_SAMPLE_POINTS', 8)
        self.dt = getattr(cfg, 'DT', 0.1)

    # ------------------------------------------------------------------
    # Rule 0: Regression (trajectory proximity to ground truth)
    # ------------------------------------------------------------------
    def _regression_loss(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
        pos_loss = self.reg_loss_fn(pred_traj[..., :2], gt_traj[..., :2]).mean()
        vel_loss = self.reg_loss_fn(pred_traj[..., 2:4], gt_traj[..., 2:4]).mean()
        return pos_loss + vel_loss

    # ------------------------------------------------------------------
    # Rule 1: Intention Consistency (Eq. 17)
    # Penalise deviations from the upper-level meta-action k:
    #   L_intent = ||v_T - v_ref^k||² + ReLU(|LatOffset| - δ_lat^k)
    # ------------------------------------------------------------------
    def _intent_consistency_loss(self, pred_traj: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        device = pred_traj.device
        loss_intent = torch.tensor(0.0, device=device)
        B, T, _ = pred_traj.shape

        vel_norm   = torch.norm(pred_traj[:, :, 2:4], dim=-1)   # (B, T)
        lat_offset = pred_traj[:, -1, 1]                       # final y displacement (ego frame)

        # --- Stop: final velocity → 0 ---
        m_stop = _mask(action_ids, _STOP_IDS)
        if m_stop.any():
            loss_intent = loss_intent + vel_norm[m_stop].mean()

        # --- Decelerate: v_end should be < v_start ---
        m_dec = _mask(action_ids, _DEC_IDS)
        if m_dec.any():
            # penalise non-decreasing velocity segments
            acc_seq = vel_norm[m_dec, 1:] - vel_norm[m_dec, :-1]
            loss_intent = loss_intent + F.relu(acc_seq).mean()

        # --- Accelerate: v_end should be > v_start ---
        m_acc = _mask(action_ids, _ACC_IDS)
        if m_acc.any():
            acc_seq = vel_norm[m_acc, 1:] - vel_norm[m_acc, :-1]
            loss_intent = loss_intent + F.relu(-acc_seq).mean()

        # --- Left manoeuvre: final lateral offset should be positive (δ_lat > 0) ---
        m_left = _mask(action_ids, _LEFT_IDS)
        if m_left.any():
            loss_intent = loss_intent + F.relu(-lat_offset[m_left]).mean()

        # --- Right manoeuvre: final lateral offset should be negative (δ_lat < 0) ---
        m_right = _mask(action_ids, _RIGHT_IDS)
        if m_right.any():
            loss_intent = loss_intent + F.relu(lat_offset[m_right]).mean()

        # --- Straight: lateral offset should remain small (|δ_lat| ≈ 0) ---
        m_straight = ~(m_left | m_right)
        if m_straight.any():
            lane_w = getattr(cfg, 'LANE_WIDTH', 3.5)
            loss_intent = loss_intent + F.relu(lat_offset[m_straight].abs() - lane_w / 4.0).mean()

        return loss_intent

    # ------------------------------------------------------------------
    # Rule 2: Dynamic Feasibility (Eq. 18)
    # Eliminate kinematically impossible trajectories via acc / jerk / ω limits
    # ------------------------------------------------------------------
    def _physics_feasibility_loss(self, pred_traj: torch.Tensor) -> torch.Tensor:
        vel = pred_traj[..., 2:4]
        acc = (vel[:, 1:, :] - vel[:, :-1, :]) / self.dt

        if acc.shape[1] >= 2:
            jerk = (acc[:, 1:, :] - acc[:, :-1, :]) / self.dt
        else:
            jerk = torch.zeros_like(acc)

        yaw = torch.atan2(pred_traj[..., 3], pred_traj[..., 2] + 1e-6)
        yaw_diff = (yaw[:, 1:] - yaw[:, :-1] + torch.pi) % (2 * torch.pi) - torch.pi
        angular_vel = yaw_diff / self.dt

        loss_acc     = F.relu(torch.norm(acc,  dim=-1) - self.max_acc).mean()
        loss_jerk    = F.relu(torch.norm(jerk, dim=-1) - self.max_jerk).mean()
        loss_angular = F.relu(angular_vel.abs()         - self.max_angular_vel).mean()
        return loss_acc + loss_jerk + loss_angular

    # ------------------------------------------------------------------
    # Rule 3: Scene Compliance (Eq. 19)
    # Penalise lane boundary violations (and red-light stop line crossing)
    # ------------------------------------------------------------------
    def _scene_compliance_loss(self, pred_traj: torch.Tensor, map_feat: torch.Tensor) -> torch.Tensor:
        device = pred_traj.device
        if map_feat is None:
            return torch.tensor(0.0, device=device)

        lane_points = map_feat.to(device)
        B, T, _ = pred_traj.shape

        idx = torch.linspace(0, T - 1, steps=min(self.scene_sample_points, T)).long().to(device)
        traj_points = pred_traj[:, idx, :2]           # (B, S, 2)

        # tp: (B, S, 1, 1, 2)  lp: (B, 1, L, P, 2)
        tp = traj_points.unsqueeze(2).unsqueeze(2)
        lp = lane_points[..., :2].unsqueeze(1)
        dists = torch.norm(tp - lp, dim=-1)           # (B, S, L, P)
        min_dist = dists.min(dim=-1)[0].min(dim=-1)[0]  # (B, S)

        lane_w = getattr(cfg, 'LANE_WIDTH', 3.5)
        loss_lane = F.relu(min_dist - lane_w / 2.0).mean()
        return loss_lane

    # ------------------------------------------------------------------
    # Combined forward (Eq. 14)
    # ------------------------------------------------------------------
    def forward(self, pred_traj, gt_traj, action_ids, map_feat=None):
        if pred_traj is None or gt_traj is None:
            raise ValueError("pred_traj and gt_traj must be provided")

        loss_reg    = self._regression_loss(pred_traj, gt_traj)
        loss_intent = self._intent_consistency_loss(pred_traj, action_ids)
        loss_phy    = self._physics_feasibility_loss(pred_traj)
        loss_scene  = (self._scene_compliance_loss(pred_traj, map_feat)
                       if map_feat is not None
                       else torch.tensor(0.0, device=pred_traj.device))

        w = self.weights
        total_loss = (w.get('reg',    1.0) * loss_reg
                    + w.get('intent', 0.5) * loss_intent
                    + w.get('phy',    0.1) * loss_phy
                    + w.get('scene',  0.1) * loss_scene)

        loss_dict = {
            'loss':       total_loss.item(),
            'total_loss': total_loss.item(),
            'reg':        loss_reg.item(),
            'intent':     loss_intent.item(),
            'phy':        loss_phy.item(),
            'scene':      loss_scene.item(),
        }
        return total_loss, loss_dict
