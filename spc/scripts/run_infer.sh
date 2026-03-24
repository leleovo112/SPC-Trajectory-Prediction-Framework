#!/bin/bash
CFG=${1:-configs/infer.yaml}
CKPT=${2:-checkpoints/best.pth}
python spc/infer.py $CFG $CKPT
