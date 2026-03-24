# src/models/encoder.py
"""
Heterogeneous feature encoding:
  - AgentEncoder:      LSTM over historical kinematics (Eq. 4)
  - MapEncoder:        VectorNet-style point MLP + max-pool (Eq. 5)
  - InteractionModule: Dual-Stacked Transformer with
      * Distance-masked direct interaction (Eq. 10) — masks neighbours > 50 m
      * Global chain-propagation (Eq. 11)
      * Safety-weighted agent-map cross-attention (Eq. 12-13)
"""
import torch
import torch.nn as nn
from config import cfg

# Distance threshold for Layer-1 (direct interaction) masking
_DIST_MASK_THRESHOLD = 50.0  # metres


class FourierEmbedding(nn.Module):
    """Positional embedding via sinusoidal Fourier features."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = None, num_freqs: int = 8):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = cfg.HIDDEN_DIM
        self.register_buffer(
            "freq_bands",
            2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        )
        self.out  = nn.Linear(input_dim * num_freqs * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_exp = x.unsqueeze(-1) * self.freq_bands.to(x.device)   # (..., D, F)
        x_exp = x_exp.view(*x.shape[:-1], -1)                     # (..., D*F)
        pe = torch.cat([torch.sin(x_exp), torch.cos(x_exp)], dim=-1)  # (..., 2*D*F)
        return self.norm(self.out(pe))


class SemanticAttributeEmbedding(nn.Module):
    """
    Embeds discrete semantic attributes (lane type, traffic light status)
    into a continuous feature of size hidden_dim.  Used to implement
    Eq. 5 (map semantic alignment) and Eq. 8 (relative position encoding).
    """

    # light: 3 states (0=red, 1=yellow, 2=green)
    # lane_type: 5 types (0=unknown, 1=normal, 2=bike, 3=sidewalk, 4=crosswalk)
    LIGHT_VOCAB     = 3
    LANE_TYPE_VOCAB = 5

    def __init__(self, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = cfg.HIDDEN_DIM
        self.emb_light     = nn.Embedding(self.LIGHT_VOCAB,     hidden_dim // 4)
        self.emb_lane_type = nn.Embedding(self.LANE_TYPE_VOCAB, hidden_dim // 4)
        self.proj = nn.Linear(hidden_dim // 2, hidden_dim)

    def forward(self, light: torch.Tensor, lane_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            light     : (B,) or (B, L) long tensor of traffic-light codes
            lane_type : same shape, lane-type codes
        Returns:
            (B, [L,] hidden_dim) semantic embedding
        """
        e_l  = self.emb_light(light.clamp(0, self.LIGHT_VOCAB - 1))
        e_lt = self.emb_lane_type(lane_type.clamp(0, self.LANE_TYPE_VOCAB - 1))
        return self.proj(torch.cat([e_l, e_lt], dim=-1))


class AgentEncoder(nn.Module):
    """LSTM encoder over T_obs historical state frames (Eq. 4)."""

    def __init__(self, input_dim: int = None, hidden_dim: int = None,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        input_dim  = input_dim  or cfg.INPUT_DIM
        hidden_dim = hidden_dim or cfg.HIDDEN_DIM
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T_obs, input_dim) → (B, hidden_dim)"""
        self.lstm.flatten_parameters()
        x_proj = self.proj(x)
        _, (h, _) = self.lstm(x_proj)
        return self.norm(h[-1])


class MapEncoder(nn.Module):
    """
    VectorNet-style subgraph encoder (Eq. 5).
    Optionally fuses semantic attributes (traffic light, lane type)
    when the map feature dimension is large enough.
    """

    # If map_feat dim ≥ SEMANTIC_DIM_THRESHOLD we attempt semantic attribute parsing
    SEMANTIC_DIM_THRESHOLD = 7

    def __init__(self, map_dim: int = None, hidden_dim: int = None):
        super().__init__()
        map_dim    = map_dim    or cfg.MAP_DIM
        hidden_dim = hidden_dim or cfg.HIDDEN_DIM

        # Geometric point MLP
        self.point_mlp = nn.Sequential(
            nn.Linear(map_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        # Optional semantic attribute embedding (used only when dim ≥ threshold)
        self.sem_embed = SemanticAttributeEmbedding(hidden_dim)
        self.fuse_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.map_dim   = map_dim
        self.hid       = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, P, D) → (B, L, hidden_dim)"""
        if x is None or x.numel() == 0:
            B = 0 if x is None else x.shape[0]
            return x.new_zeros((B, 0, self.hid))

        B, L, P, D = x.shape
        x_flat  = x.view(B * L, P, D)                    # (B*L, P, D)
        # Slice to map_dim before MLP to handle maps with more columns than expected
        x_geo   = x_flat[..., :self.map_dim]              # (B*L, P, map_dim)
        geo_feat = self.point_mlp(x_geo)                  # (B*L, P, hidden)
        geo_feat, _ = torch.max(geo_feat, dim=1)          # (B*L, hidden)
        geo_feat = geo_feat.view(B, L, self.hid)          # (B, L, hidden)

        # Semantic attribute fusion when available (requires D ≥ 7 in the original map)
        if D >= self.SEMANTIC_DIM_THRESHOLD:
            # Columns 5=lane_type, 6=traffic_light (convention from data_process)
            light_idx = x[..., 6].mean(dim=-1).long().clamp(0, 2)      # (B, L)
            lane_type = x[..., 5].mean(dim=-1).long().clamp(0, 4)      # (B, L)
            sem_feat  = self.sem_embed(light_idx, lane_type)            # (B, L, hidden)
            geo_feat  = self.fuse_proj(torch.cat([geo_feat, sem_feat], dim=-1))

        return geo_feat


class InteractionModule(nn.Module):
    """
    Dual-Stacked Transformer interaction module (Section 4.2):

    Layer 1 – Direct Interaction:
        Distance-masked self-attention (map nodes > 50 m are masked).
        Captures strong pairwise correlations within the immediate vicinity.

    Layer 2 – Chain Propagation:
        Global receptive-field Transformer, no masking.
        Captures implicit long-range interaction chains ("Butterfly Effect").

    Agent-Map Cross-Attention:
        Safety-priority weighted aggregation (Eq. 12-13).
    """

    def __init__(self):
        super().__init__()
        H = cfg.HIDDEN_DIM
        self.pe = FourierEmbedding(input_dim=2, hidden_dim=H)

        # Layer 1: distance-masked direct interaction
        self.direct_interaction = nn.TransformerEncoderLayer(
            d_model=H, nhead=4, dim_feedforward=1024, dropout=0.1,
            batch_first=True, norm_first=True
        )
        # Layer 2: global chain propagation
        self.chain_propagation = nn.TransformerEncoderLayer(
            d_model=H, nhead=4, dim_feedforward=1024, dropout=0.1,
            batch_first=True, norm_first=True
        )
        # Agent-Map cross-attention
        self.agent_map_cross_attn = nn.MultiheadAttention(
            embed_dim=H, num_heads=4, batch_first=True
        )
        # Safety priority weight scalar per map element (Eq. 13)
        self.safety_weight_mlp = nn.Sequential(nn.Linear(H, 1), nn.Sigmoid())
        self.norm = nn.LayerNorm(H)

    def _build_distance_padding_mask(
        self, agent_pos: torch.Tensor, map_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Build src_key_padding_mask (B, 1+L) for Layer-1 self-attention.
        The agent token (index 0) is never masked.
        Map tokens at distance > _DIST_MASK_THRESHOLD from the agent are masked.
        """
        B, L = map_pos.shape[:2]
        dists = torch.norm(
            map_pos - agent_pos.unsqueeze(1), dim=-1
        )  # (B, L)
        far = dists > _DIST_MASK_THRESHOLD  # (B, L)
        # Prepend False for the agent token
        agent_mask = torch.zeros(B, 1, dtype=torch.bool, device=far.device)
        return torch.cat([agent_mask, far], dim=1)   # (B, 1+L)

    def forward(
        self,
        agent_emb: torch.Tensor,    # (B, H)
        map_emb:   torch.Tensor,    # (B, L, H) or None
        agent_pos: torch.Tensor,    # (B, 2)
        map_pos:   torch.Tensor,    # (B, L, 2) or None
    ) -> torch.Tensor:
        """Returns: (B, H) scene context for the agent."""
        B = agent_emb.shape[0]
        agent_pe    = self.pe(agent_pos)                    # (B, H)
        agent_token = (agent_emb + agent_pe).unsqueeze(1)   # (B, 1, H)

        if map_emb is None or map_emb.numel() == 0:
            return self.norm(agent_token.squeeze(1))

        map_pe     = self.pe(map_pos)                       # (B, L, H)
        map_tokens = map_emb + map_pe                       # (B, L, H)

        # Concatenate agent + map tokens
        tokens = torch.cat([agent_token, map_tokens], dim=1)  # (B, 1+L, H)

        # --- Layer 1: distance-masked direct interaction ---
        dist_pad_mask = self._build_distance_padding_mask(agent_pos, map_pos)
        direct_out = self.direct_interaction(
            tokens, src_key_padding_mask=dist_pad_mask
        )  # (B, 1+L, H)

        # --- Layer 2: global chain propagation (no masking) ---
        chain_out = self.chain_propagation(direct_out)      # (B, 1+L, H)

        agent_context = chain_out[:, 0, :]                  # (B, H)

        # --- Agent-Map cross-attention with safety priority weighting ---
        map_attn_out, _ = self.agent_map_cross_attn(
            query=agent_context.unsqueeze(1),
            key=map_tokens,
            value=map_tokens,
        )  # (B, 1, H)

        safety_weight         = self.safety_weight_mlp(map_attn_out)   # (B, 1, 1)
        weighted_map_context  = map_attn_out * safety_weight            # (B, 1, H)

        final_context = self.norm(agent_context + weighted_map_context.squeeze(1))
        return final_context   # (B, H)
