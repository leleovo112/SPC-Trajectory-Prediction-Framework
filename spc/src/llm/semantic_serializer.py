# src/llm/semantic_serializer.py
"""
SemanticSerializer: transforms heterogeneous numerical perception data into
a structured symbolic context for the Cognitive Expert Reasoning Module.

Implements the four-layer Knowledge Base (Section 4.3):
  R_law  – Traffic regulations and right-of-way logic
  R_phy  – Physical consistency boundaries (kinematics, TTC, geometry)
  R_map  – Semantic mapping of spatial data to scene commonsense
  R_soc  – Social / defensive driving priors via LLM knowledge

Generates a four-stage CoT prompt (Algorithm 1):
  Stage 1: Risk Perception     – identify conflicts via R_phy and R_soc
  Stage 2: Rule Association    – invoke R_law for right-of-way analysis
  Stage 3: Evaluation          – verify candidate actions against R_phy limits
  Stage 4: Locking             – select final discrete intention k ∈ A
"""
import numpy as np
import torch
from typing import Any, Dict, List

from config import cfg


# -----------------------------------------------------------------------
# Knowledge Base (R_law, R_phy, R_map, R_soc)
# -----------------------------------------------------------------------
_R_LAW = """\
[R_law – Traffic Regulations & Right-of-Way]
• Red light / stop sign: vehicle MUST stop before the stop line.
• Yellow light: prepare to stop unless already committed to the intersection.
• Unprotected left turn: must yield to oncoming straight traffic.
• Lane change: must yield to vehicles already occupying the target lane.
• Merging on-ramp: highway vehicles have right of way; merging vehicle must yield.
• Emergency vehicle approaching (siren / lights): pull over and stop.
• Priority hierarchy: Physical Safety > Traffic Regulations > Driving Efficiency."""

_R_PHY = """\
[R_phy – Physical Consistency Boundaries]
• Maximum comfortable deceleration: 5 m/s²; emergency deceleration ≤ 8 m/s².
• Maximum comfortable acceleration: 3 m/s²; peak ≤ 5 m/s².
• Maximum angular velocity: 0.6 rad/s.
• Minimum safe headway Time-to-Collision (TTC) threshold: 3 s.
• Vehicle geometry (approx.): length ≈ 4.5 m, width ≈ 2.0 m.
• Safe lateral clearance during lane change: ≥ 0.5 m from lane markings."""

_R_MAP = """\
[R_map – Semantic Scene Commonsense]
• Distance < 5 m to a lead vehicle: Critical Range → immediate brake required.
• Distance 5–15 m: Warning Range → prepare to decelerate.
• Distance 15–50 m: Monitoring Range → stay alert.
• Distance > 50 m: Free-flow Range → normal driving.
• Intersection (multiple branching lanes detected): heightened vigilance, reduce speed.
• Lane curves left / right: adjust lateral position proactively."""

_R_SOC = """\
[R_soc – Social & Defensive Driving Priors]
• Defensive posture: when in doubt, decelerate rather than accelerate.
• Cooperative merging: if a merge can be safely facilitated without risk, do so.
• Red-light runner anticipation: even on green, scan for late-running cross-traffic.
• Vulnerable road users (pedestrians, cyclists): always grant right of way.
• Adversarial cut-in: maintain a 2-second gap buffer after a cut-in event.
• Social-norm yielding: mild proactive deceleration to enable cooperative manoeuvres
  optimises global traffic stability even when the ego has right of way."""

_KNOWLEDGE_BASE = f"""
{_R_LAW}

{_R_PHY}

{_R_MAP}

{_R_SOC}
""".strip()


# -----------------------------------------------------------------------
# SemanticSerializer
# -----------------------------------------------------------------------
class SemanticSerializer:
    """Converts raw numerical perception data into CoT prompts."""

    # ------------------------------------------------------------------
    # Private: analyse ego kinematics
    # ------------------------------------------------------------------
    @staticmethod
    def _analyze_ego_kinematics(hist_traj: np.ndarray) -> Dict[str, Any]:
        if hist_traj is None or len(hist_traj) == 0:
            return {"full_desc": "Unknown ego state.",
                    "v_kph": 0.0, "acc_avg": 0.0, "yaw_rate": 0.0,
                    "acc_desc": "unknown", "turn_desc": "unknown", "v_mps": 0.0}

        curr = hist_traj[-1]
        v_mps = float(np.linalg.norm(curr[2:4])) if len(curr) >= 4 else 0.0
        v_kph = v_mps * 3.6

        window = max(1, min(10, len(hist_traj) - 1))
        v_prev = float(np.linalg.norm(hist_traj[-window, 2:4])) if len(curr) >= 4 else 0.0
        acc_avg = (v_mps - v_prev) / (cfg.DT * window) if cfg.DT * window > 0 else 0.0

        yaw_curr = float(curr[6]) if len(curr) > 6 else 0.0
        yaw_prev = float(hist_traj[-window, 6]) if len(curr) > 6 else 0.0
        yaw_diff = (yaw_curr - yaw_prev + np.pi) % (2 * np.pi) - np.pi
        yaw_rate = yaw_diff / (cfg.DT * window) if cfg.DT * window > 0 else 0.0

        # Speed description
        if v_kph < 1.0:
            speed_desc = "Stationary (< 1 km/h)"
        elif v_kph < 20:
            speed_desc = f"Low-speed ({v_kph:.1f} km/h)"
        elif v_kph < 60:
            speed_desc = f"Urban normal ({v_kph:.1f} km/h)"
        else:
            speed_desc = f"High-speed ({v_kph:.1f} km/h)"

        # Longitudinal description
        if acc_avg > 0.8:
            acc_desc = "accelerating"
        elif acc_avg < -3.5:
            acc_desc = "emergency braking"
        elif acc_avg < -0.8:
            acc_desc = "decelerating"
        else:
            acc_desc = "maintaining constant speed"

        # Lateral description
        if yaw_rate > 0.05:
            turn_desc = "turning left"
        elif yaw_rate < -0.05:
            turn_desc = "turning right"
        else:
            turn_desc = "going straight"

        return {
            "full_desc": (f"Ego vehicle: {speed_desc}, {acc_desc}, {turn_desc}. "
                          f"Current speed = {v_kph:.1f} km/h, "
                          f"acc = {acc_avg:.2f} m/s², yaw_rate = {yaw_rate:.3f} rad/s."),
            "v_kph":    v_kph,
            "v_mps":    v_mps,
            "acc_avg":  acc_avg,
            "yaw_rate": yaw_rate,
            "acc_desc": acc_desc,
            "turn_desc": turn_desc,
        }

    # ------------------------------------------------------------------
    # Private: analyse surrounding agents
    # ------------------------------------------------------------------
    @staticmethod
    def _analyze_surrounding_agents(surrounding_agents: List[Dict]) -> str:
        if not surrounding_agents:
            return "No surrounding vehicles detected within the sensor range."

        descs = []
        for ag in surrounding_agents:
            dist   = ag.get('rel_dist', 0.0)
            direc  = ag.get('rel_dir', 'ahead')
            v_kph  = ag.get('v_kph',   0.0)
            action = ag.get('action',  'unknown')

            # Map distance to semantic range label (R_map)
            if dist < 5:
                range_lbl = "CRITICAL range"
            elif dist < 15:
                range_lbl = "warning range"
            elif dist < 50:
                range_lbl = "monitoring range"
            else:
                range_lbl = "free-flow range"

            ttc_str = ""
            if v_kph > 1.0 and direc == "ahead":
                ego_vkph = 0.0  # placeholder; refined in generate_cot_prompt
                rel_speed = max(ego_vkph - v_kph, 0.0) / 3.6
                if rel_speed > 0.1:
                    ttc = dist / rel_speed
                    ttc_str = f", TTC ≈ {ttc:.1f} s"

            descs.append(
                f"  • Vehicle at {dist:.1f} m {direc} [{range_lbl}], "
                f"{v_kph:.1f} km/h, {action}{ttc_str}."
            )
        return "Surrounding vehicles:\n" + "\n".join(descs)

    # ------------------------------------------------------------------
    # Private: analyse map / scene
    # ------------------------------------------------------------------
    @staticmethod
    def _analyze_map_scene(map_feat: np.ndarray) -> Dict[str, Any]:
        empty_result = {
            "full_desc": "Map information unavailable.",
            "has_intersection": False, "curve_dir": "straight",
            "light_status": -1, "light_desc": "Traffic light status unknown."
        }
        if map_feat is None:
            return empty_result
        if isinstance(map_feat, torch.Tensor):
            map_feat = map_feat.cpu().numpy()
        if map_feat.size == 0:
            return {**empty_result, "full_desc": "Open road ahead, no map constraints."}

        line_centers = map_feat[..., 0:2].mean(axis=1)
        valid = ((line_centers[:, 0] > 5) & (line_centers[:, 0] < 50)
                 & (np.abs(line_centers[:, 1]) < 15))
        if not np.any(valid):
            return {**empty_result, "full_desc": "Open road, no nearby lane constraints."}

        valid_lines = map_feat[valid]
        line_vecs = valid_lines[:, -1, 0:2] - valid_lines[:, 0, 0:2]
        angles = np.arctan2(line_vecs[:, 1], line_vecs[:, 0])

        has_intersection = np.any(angles > 0.3) and np.any(angles < -0.3)
        avg_angle = float(np.mean(angles))

        if avg_angle > 0.15:
            curve_dir  = "left"
            road_desc  = "The lane curves to the LEFT ahead."
        elif avg_angle < -0.15:
            curve_dir  = "right"
            road_desc  = "The lane curves to the RIGHT ahead."
        else:
            curve_dir  = "straight"
            road_desc  = "Straight road ahead."

        if has_intersection:
            road_desc = "Approaching an INTERSECTION with multiple branching lanes."

        # Traffic light (column index 6 by convention)
        if valid_lines.shape[-1] > 6:
            light_val = float(valid_lines[..., 6].mean())
            if light_val < 0.5:
                light_status = 0
                light_desc   = "Traffic light is RED – must stop before the stop line."
            elif light_val < 1.5:
                light_status = 1
                light_desc   = "Traffic light is YELLOW – prepare to stop."
            else:
                light_status = 2
                light_desc   = "Traffic light is GREEN – may proceed with caution."
        else:
            light_status = -1
            light_desc   = "Traffic light status not available in map data."

        return {
            "full_desc":        f"{road_desc} {light_desc}",
            "has_intersection": has_intersection,
            "curve_dir":        curve_dir,
            "light_status":     light_status,
            "light_desc":       light_desc,
        }

    # ------------------------------------------------------------------
    # Private: risk level description
    # ------------------------------------------------------------------
    @staticmethod
    def _analyze_risk_level(meta: Dict, kine: Dict, map_info: Dict) -> str:
        is_accident   = meta.get('is_accident', False)
        is_emergency  = "emergency braking" in kine.get("acc_desc", "")
        red_light     = map_info.get("light_status", -1) == 0
        intersection  = map_info.get("has_intersection", False)

        if is_emergency or red_light:
            return ("🔴 HIGH RISK: "
                    + ("Emergency braking detected. " if is_emergency else "")
                    + ("Red light ahead. " if red_light else "")
                    + "Immediate collision avoidance action required.")
        if is_accident or intersection:
            return ("🟡 MEDIUM RISK: "
                    + ("Accident-prone scenario detected. " if is_accident else "")
                    + ("Approaching intersection. " if intersection else "")
                    + "Defensive driving strategy recommended.")
        return "🟢 LOW RISK: Normal driving scenario. Maintain safe driving practices."

    # ------------------------------------------------------------------
    # Public: generate 4-stage CoT prompt
    # ------------------------------------------------------------------
    @staticmethod
    def generate_cot_prompt(
        raw_hist:          np.ndarray,
        map_feat:          np.ndarray,
        surrounding_agents: List[Dict],
        meta:              Dict,
    ) -> str:
        kine_data  = SemanticSerializer._analyze_ego_kinematics(raw_hist)
        map_data   = SemanticSerializer._analyze_map_scene(map_feat)
        agent_desc = SemanticSerializer._analyze_surrounding_agents(surrounding_agents)
        risk_desc  = SemanticSerializer._analyze_risk_level(meta, kine_data, map_data)

        # Inject ego speed into surrounding agent TTC strings
        for ag in surrounding_agents:
            ag['_ego_vkph'] = kine_data.get('v_kph', 0.0)

        # Build action space description from config
        if hasattr(cfg, 'ACTION_SPACE'):
            action_lines = [f"  ID {k}: {v}" for k, v in cfg.ACTION_SPACE.items()]
        else:
            action_lines = [f"  ID {i}: ACTION_{i}" for i in range(cfg.NUM_ACTIONS)]
        action_str = "\n".join(action_lines)

        prompt = f"""[SYSTEM ROLE]
You are the Cognitive Expert – the safety decision-making core of an autonomous vehicle.
Your task is to perform structured Chain-of-Thought (CoT) reasoning through FOUR mandatory
stages and output the final meta-action ID.  You must consult the Knowledge Base at every
stage and strictly honour the safety hierarchy:
    Physical Safety  >  Traffic Regulations  >  Driving Efficiency

════════════════════════════════════════════════════════════════
 KNOWLEDGE BASE (must be consulted at each reasoning stage)
════════════════════════════════════════════════════════════════
{_KNOWLEDGE_BASE}

════════════════════════════════════════════════════════════════
 PERCEPTION INPUTS  (serialised scene state Φ(sᵢ))
════════════════════════════════════════════════════════════════
1. Ego Vehicle Kinematics:
   {kine_data['full_desc']}

2. Map & Traffic Signal State:
   {map_data['full_desc']}

3. Surrounding Agent Interactions:
{agent_desc}

4. Scene Risk Assessment:
   {risk_desc}

════════════════════════════════════════════════════════════════
 ACTION SPACE  (12 decoupled meta-actions, select exactly one)
════════════════════════════════════════════════════════════════
{action_str}

════════════════════════════════════════════════════════════════
 FOUR-STAGE COT REASONING PROTOCOL  (must follow in order)
════════════════════════════════════════════════════════════════
Stage 1 – Risk Perception  [R_phy + R_soc]:
  Identify ALL risk sources: kinematic conflicts, TTC violations, traffic light
  infringements, vulnerable road users, and social-norm violations.

Stage 2 – Rule Association  [R_law + R_map]:
  For each identified risk, explicitly cite the applicable traffic regulation or
  right-of-way rule.  Determine which agent has priority and which must yield.

Stage 3 – Evaluation  [R_phy limits verification]:
  For each candidate action, verify: (a) it satisfies R_phy kinematic limits,
  (b) it resolves the identified risks, (c) it is consistent with R_law priority.
  Rank candidates and eliminate infeasible ones.

Stage 4 – Locking  [Final intent decision]:
  Select the single best action from the feasible set and justify why it
  optimally balances safety, compliance, and efficiency.

════════════════════════════════════════════════════════════════
 OUTPUT REQUIREMENTS  (strict JSON only – no other text)
════════════════════════════════════════════════════════════════
{{
    "stage1_risk_perception":  "<detailed description of all risk sources identified>",
    "stage2_rule_association": "<traffic rules / right-of-way logic invoked>",
    "stage3_evaluation":       "<candidate actions evaluated against R_phy limits>",
    "stage4_locking":          "<brief justification of the final action selection>",
    "action_id": <integer ID of the selected meta-action, 0–{cfg.NUM_ACTIONS - 1}>
}}"""
        return prompt.strip()
