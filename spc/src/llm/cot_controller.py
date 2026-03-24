# src/llm/cot_controller.py
"""
CoTController: orchestrates the LLM-driven Cognitive Expert pipeline.

RuleBasedFallback: deterministic rule engine used when the LLM is unavailable
or returns an invalid response.  Maps prompt keywords to the 12-action space.
"""
import os
import json
import time
import logging

from src.llm.llm_client import LLMClient
from src.llm.semantic_serializer import SemanticSerializer
from config import cfg

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# 12-action space reference (mirrors config.py ACTION_SPACE)
# Straight: 0=Keep 1=Acc 2=Dec 3=Stop
# Left:     4=Keep 5=Acc 6=Dec 7=Stop
# Right:    8=Keep 9=Acc 10=Dec 11=Stop
# -----------------------------------------------------------------------


class RuleBasedFallback:
    """
    Lightweight rule engine that derives a meta-action from the serialised
    prompt keywords whenever the LLM is unavailable or times out.
    Priority: Stop > Dec > Acc > Keep, combined with lateral intent.
    """

    @staticmethod
    def infer(prompt: str):
        pl = prompt.lower()

        # --- Scene flags ---
        red_light      = ("red" in pl and "traffic light" in pl) or "red light" in pl
        is_emergency   = "emergency braking" in pl or "immediate" in pl
        is_stationary  = "stationary" in pl
        is_decelerating = "decelerating" in pl or "decelerate" in pl or "decel" in pl
        is_accelerating = "accelerating" in pl or "accelerate" in pl
        is_high_risk   = "high risk" in pl or "critical range" in pl
        is_left        = "turning left" in pl or "left turn" in pl or "curves to the left" in pl or "left manoeuvre" in pl
        is_right       = "turning right" in pl or "right turn" in pl or "curves to the right" in pl or "right manoeuvre" in pl

        # --- Longitudinal determination ---
        if red_light or is_emergency:
            lon = "stop"
        elif is_stationary:
            lon = "stop"
        elif is_decelerating or is_high_risk:
            lon = "dec"
        elif is_accelerating:
            lon = "acc"
        else:
            lon = "keep"

        # --- Lateral determination ---
        if is_left:
            lat = "left"
        elif is_right:
            lat = "right"
        else:
            lat = "straight"

        # --- Map to 12-action ID ---
        lat_base   = {"straight": 0, "left": 4, "right": 8}[lat]
        lon_offset = {"keep": 0, "acc": 1, "dec": 2, "stop": 3}[lon]
        action_id  = lat_base + lon_offset

        reasoning = (
            f"Fallback rule engine: lat={lat}, lon={lon} → action_id={action_id}. "
            f"Flags: red_light={red_light}, emergency={is_emergency}, "
            f"decel={is_decelerating}, accel={is_accelerating}, "
            f"left={is_left}, right={is_right}."
        )
        return action_id, reasoning


class CoTController:
    """
    Manages the full semantic serialisation → LLM CoT → action embedding pipeline.
    Implements result caching to avoid redundant API calls.
    """

    def __init__(self, provider: str = "openai", cache_dir: str = None):
        self.llm        = LLMClient(provider=provider)
        self.serializer = SemanticSerializer()
        self.cache_dir  = cache_dir or cfg.LLM_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, sample_id) -> str:
        return os.path.join(self.cache_dir, f"{sample_id:08d}.json")

    def _load_cache(self, sample_id):
        path = self._cache_path(sample_id)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                return data.get('action_id'), data
            except Exception:
                pass
        return None, None

    def _save_cache(self, sample_id, action_id, parsed, prompt, response):
        path = self._cache_path(sample_id)
        payload = {
            'action_id': action_id,
            'parsed':    parsed,
            'prompt':    prompt,
            'response':  response,
            'timestamp': time.time(),
        }
        try:
            with open(path, 'w') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    def decide(self, sample: dict, sample_id=None) -> tuple:
        """
        Args:
            sample    : dict containing 'hist', 'map_polylines', 'surrounding', 'meta'
            sample_id : optional int for caching

        Returns:
            (action_id: int, parsed: dict)
        """
        # Cache lookup
        if sample_id is not None:
            cached_id, cached_data = self._load_cache(sample_id)
            if cached_id is not None:
                return cached_id, cached_data

        # Extract fields
        hist       = sample.get('hist') or sample.get('hist_states')
        map_feat   = sample.get('map_polylines', None)
        surrounding = sample.get('surrounding', sample.get('surrounding_agents', []))
        meta       = sample.get('meta', {})

        if hasattr(hist, 'cpu'):
            hist = hist.cpu().numpy()

        prompt = self.serializer.generate_cot_prompt(hist, map_feat, surrounding, meta)

        action_id, reasoning = self.llm.query(prompt)

        if action_id is None:
            action_id, reasoning = RuleBasedFallback.infer(prompt)
            parsed = {'action_id': action_id, 'reasoning': reasoning, 'fallback': True}
        else:
            parsed = {'action_id': action_id, 'reasoning': reasoning, 'fallback': False}

        if sample_id is not None:
            self._save_cache(sample_id, action_id, parsed, prompt, reasoning)

        return action_id, parsed
