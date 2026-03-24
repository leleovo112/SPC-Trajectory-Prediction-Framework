# src/models/predictor.py
"""
HeterogeneousPredictor: the full neural execution module.

Architecture (Section 4):
  1. AgentEncoder        – LSTM over historical kinematics
  2. MapEncoder          – VectorNet point MLP + max-pool
  3. InteractionModule   – Dual-Stacked Transformer with distance masking
  4. MetaActionEmbedding – Learnable codebook (Eq. 15), dim = INTENT_DIM
  5. IntentionalDecoder  – Query-based Transformer decoder conditioned on
                           intent embedding (Eq. 16), outputs PRED_LEN × 4
"""
import torch
import torch.nn as nn
from config import cfg
from src.models.encoder import AgentEncoder, MapEncoder, InteractionModule


class MetaActionEmbedding(nn.Module):
    """
    Learnable meta-action codebook E_action (Eq. 15).
    Maps discrete action label k → continuous intent vector z_intent ∈ R^INTENT_DIM.
    Orthogonal initialisation encourages well-separated meta-action representations.
    """

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(cfg.NUM_ACTIONS, cfg.INTENT_DIM)
        nn.init.orthogonal_(self.embed.weight)

    def forward(self, action_id: torch.Tensor) -> torch.Tensor:
        return self.embed(action_id.long())    # (B, INTENT_DIM)


class IntentionalDecoder(nn.Module):
    """
    Query-based Transformer decoder conditioned on the intent embedding.

    The intent vector z_intent is projected to HIDDEN_DIM and appended to the
    key/value context [H_hist, H_map, z_intent], ensuring the probability
    distribution of generated trajectories is restricted to the semantic mode
    specified by the LLM (Eq. 16):

        H_plan = Attention(Q_plan, K=[H_hist, H_map, z_intent], V=[...])
    """

    def __init__(self):
        super().__init__()
        H = cfg.HIDDEN_DIM
        self.pred_len = cfg.PRED_LEN

        # Project INTENT_DIM → HIDDEN_DIM to join the key/value context
        self.intent_proj = nn.Sequential(
            nn.Linear(cfg.INTENT_DIM, H),
            nn.LayerNorm(H),
        )

        # Learnable planning queries (one per future step)
        self.plan_queries = nn.Parameter(torch.randn(1, self.pred_len, H) * 0.02)

        # Cross-attention: Q=planning queries, K/V=multi-modal context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=H, num_heads=4, batch_first=True, dropout=0.1
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H * 2),
            nn.GELU(),
            nn.Linear(H * 2, H),
        )
        self.norm = nn.LayerNorm(H)

        # Output head: hidden → 2D velocity increment [Δvx, Δvy]
        # Only velocity increments are needed; positions are integrated from start_state.
        self.head = nn.Linear(H, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.dt = cfg.DT

    def forward(
        self,
        context:     torch.Tensor,      # (B, H)   scene context from InteractionModule
        intent_feat: torch.Tensor,      # (B, INTENT_DIM)
        start_state: torch.Tensor,      # (B, ≥4)  last observed state [x, y, vx, vy, ...]
    ) -> torch.Tensor:
        """Returns: (B, PRED_LEN, 4) in [x, y, vx, vy]."""
        B = context.shape[0]
        device = context.device

        # Project intent to same dimension as context
        intent_h = self.intent_proj(intent_feat)    # (B, H)

        # Build key/value context: [H_hist(=context), z_intent]
        kv = torch.stack([context, intent_h], dim=1)   # (B, 2, H)

        # Expand planning queries to batch
        queries = self.plan_queries.expand(B, -1, -1)  # (B, PRED_LEN, H)

        # Cross-attention decoding
        attn_out, _ = self.cross_attn(query=queries, key=kv, value=kv)
        plan_feat   = self.norm(attn_out + queries)
        plan_feat   = plan_feat + self.ffn(plan_feat)  # (B, PRED_LEN, H)

        # Predict per-step velocity increments [Δvx, Δvy] (residual-physics scheme)
        dv = self.head(plan_feat)                      # (B, PRED_LEN, 2)

        # Physics integration from the last observed state
        if start_state is None:
            v0 = torch.zeros((B, 2), device=device)
            p0 = torch.zeros((B, 2), device=device)
        else:
            ss = start_state.to(device)
            p0 = ss[:, 0:2]
            v0 = ss[:, 2:4]

        v_seq = v0.unsqueeze(1) + torch.cumsum(dv * self.dt, dim=1)    # (B, T, 2)
        p_seq = p0.unsqueeze(1) + torch.cumsum(v_seq * self.dt, dim=1) # (B, T, 2)

        pred = torch.cat([p_seq, v_seq], dim=-1)   # (B, PRED_LEN, 4)
        return pred


class HeterogeneousPredictor(nn.Module):
    """Full neuro-symbolic execution module (Section 4.5)."""

    def __init__(self):
        super().__init__()
        self.agent_enc   = AgentEncoder(input_dim=cfg.INPUT_DIM, hidden_dim=cfg.HIDDEN_DIM)
        self.map_enc     = MapEncoder(map_dim=cfg.MAP_DIM, hidden_dim=cfg.HIDDEN_DIM)
        self.interaction = InteractionModule()
        self.action_embed = MetaActionEmbedding()
        self.decoder      = IntentionalDecoder()

    def forward(
        self,
        hist_norm:       torch.Tensor,         # (B, T_obs, INPUT_DIM)
        map_feat:        torch.Tensor | None,  # (B, L, P, MAP_DIM)  or None
        action_id:       torch.Tensor,         # (B,) long
        raw_start_state: torch.Tensor | None,  # (B, ≥4) last raw state
    ) -> torch.Tensor:
        device = hist_norm.device

        agent_feat = self.agent_enc(hist_norm)   # (B, H)

        if map_feat is None or (isinstance(map_feat, torch.Tensor) and map_feat.numel() == 0):
            map_feat_enc = None
            map_pos      = None
        else:
            map_feat_enc = self.map_enc(map_feat)                 # (B, L, H)
            map_pos      = map_feat[..., :2].mean(dim=2) / 50.0  # (B, L, 2) normalised

        B = hist_norm.shape[0]
        # Use the actual last-observed ego position (normalized by 50 m, same scale as
        # map_pos) so that the distance mask in InteractionModule is meaningful.
        # For ego-centric data the agent is near origin; for world-coord data this
        # provides the correct reference point for lane masking.
        if raw_start_state is not None:
            agent_pos = raw_start_state.to(device)[:, 0:2] / 50.0  # (B, 2)
        else:
            agent_pos = torch.zeros((B, 2), device=device)

        context     = self.interaction(agent_feat, map_feat_enc, agent_pos, map_pos)  # (B, H)
        intent_feat = self.action_embed(action_id)   # (B, INTENT_DIM)

        if raw_start_state is not None:
            raw_start_state = raw_start_state.to(device)

        pred_traj = self.decoder(context, intent_feat, raw_start_state)  # (B, PRED_LEN, 4)
        return pred_traj

    @torch.no_grad()
    def predict(
        self,
        hist_norm:       torch.Tensor,
        map_feat:        torch.Tensor | None,
        action_id:       torch.Tensor,
        raw_start_state: torch.Tensor | None,
        to_numpy:        bool = True,
    ):
        self.eval()
        pred = self.forward(hist_norm, map_feat, action_id, raw_start_state)
        return pred.cpu().numpy() if to_numpy else pred
