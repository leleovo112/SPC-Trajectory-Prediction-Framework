# config.py
import torch
import os

# Data paths
DATA_ROOT = os.environ.get("DEEPACCIDENT_ROOT", "/root/autodl-tmp/deepaccident/data")
# Template path for split-specific processed data; {split} is replaced by 'train'/'val'/'test'
PROCESSED_DATA_TEMPLATE = os.environ.get(
    "PROCESSED_DATA_TEMPLATE",
    "/home/carla_user/processed/{split}_samples.pkl"
)
# Legacy single-file path (fallback when split-specific file is missing)
PROCESSED_DATA_PATH = os.environ.get("PROCESSED_DATA_PATH", "/home/carla_user/processed/all_samples.pkl")
SCALER_PATH = os.environ.get("SCALER_PATH", "/home/carla_user/processed/scalers.pkl")

# Model / data dims
INPUT_DIM = 7        # x, y, vx, vy, ax, ay, yaw
HIDDEN_DIM = 256
INTENT_DIM = 128     # meta-action embedding dimension
MAP_DIM = 2
PRED_LEN = 30
OBS_LEN = 20
DT = 0.1

# ---------------------------------------------------------------
# 12-action decoupled meta-action space (3 lateral × 4 longitudinal)
# Lateral:     Straight(0-3), Left(4-7), Right(8-11)
# Longitudinal: Keep(+0), Acc(+1), Dec(+2), Stop(+3)
# ---------------------------------------------------------------
NUM_ACTIONS = 12

ACTION_SPACE = {
    0:  "Straight_Keep  : Maintain current lane and speed",
    1:  "Straight_Acc   : Accelerate along current lane",
    2:  "Straight_Dec   : Decelerate along current lane",
    3:  "Straight_Stop  : Emergency stop / hard brake",
    4:  "Left_Keep      : Change to left lane / left turn, maintain speed",
    5:  "Left_Acc       : Change to left lane / left turn, accelerate",
    6:  "Left_Dec       : Change to left lane / left turn, decelerate",
    7:  "Left_Stop      : Yield / stop while executing left maneuver",
    8:  "Right_Keep     : Change to right lane / right turn, maintain speed",
    9:  "Right_Acc      : Change to right lane / right turn, accelerate",
    10: "Right_Dec      : Change to right lane / right turn, decelerate",
    11: "Right_Stop     : Yield / stop while executing right maneuver",
}

# Action group masks (used by safety loss and CoT fallback)
STOP_ACTIONS  = {3, 7, 11}
DEC_ACTIONS   = {2, 6, 10}
ACC_ACTIONS   = {1, 5, 9}
LEFT_ACTIONS  = {4, 5, 6, 7}
RIGHT_ACTIONS = {8, 9, 10, 11}

# Map padding defaults
MAP_MAX_LINES = 50
MAP_POINTS_PER_LINE = 20

# Training defaults
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EPOCHS = 20
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Safety / loss defaults
LOSS_WEIGHTS = {
    'reg':    1.0,
    'intent': 0.5,
    'phy':    0.1,
    'scene':  0.1,
}
MAX_ACC = 8.0
MAX_JERK = 20.0
MAX_ANG_VEL = 0.6
LANE_WIDTH = 3.5
SCENE_SAMPLE_POINTS = 8

# LLM defaults
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY",   "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
QWEN_API_KEY     = os.environ.get("QWEN_API_KEY",     "")
LLM_TEMPERATURE  = 0.0
LLM_MAX_TOKENS   = 1024
LLM_PROVIDER     = os.environ.get("LLM_PROVIDER", "openai")  # openai | deepseek | qwen | local

# Logging / output
LLM_CACHE_DIR    = os.environ.get("LLM_CACHE_DIR",   "/home/carla_user/llm_cache")
CHECKPOINT_DIR   = os.environ.get("CHECKPOINT_DIR",  "/home/carla_user/checkpoints")
OUTPUT_DIR       = os.environ.get("OUTPUT_DIR",       "/home/carla_user/results")
USE_TENSORBOARD  = False
TENSORBOARD_DIR  = "/home/carla_user/runs"

# Misc
SEED = 42


class Config:
    """Encapsulates all module-level constants as object attributes."""
    def __init__(self):
        for key, value in globals().items():
            if not key.startswith("__") and key != "Config":
                setattr(self, key, value)


cfg = Config()
