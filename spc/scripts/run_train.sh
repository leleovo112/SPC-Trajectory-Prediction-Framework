#!/bin/bash
CFG=${1:-configs/train.yaml}
python spc/train.py $CFG
