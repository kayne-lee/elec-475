#!/bin/zsh
cd "$(dirname "$0")"
PY=../.venv/bin/python
$PY train.py --kd none -e 100 > logs/train_none.log 2>&1
$PY train.py --kd response -e 100 > logs/train_response.log 2>&1
$PY train.py --kd feature -e 100 > logs/train_feature.log 2>&1
$PY train.py --model custom --kd both --aug -e 100 > logs/train_custom_both.log 2>&1
$PY train.py --model custom --kd none --aug -e 100 > logs/train_custom_none.log 2>&1
echo ALL_DONE
