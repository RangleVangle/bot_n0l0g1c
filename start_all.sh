#!/bin/bash
source venv/bin/activate
mkdir -p logs pids   # добавили pids

python worker_technical.py > logs/technical.log 2>&1 &
echo $! > pids/technical.pid

python worker_sentiment.py > logs/sentiment.log 2>&1 &
echo $! > pids/sentiment.pid

python worker_risk.py > logs/risk.log 2>&1 &
echo $! > pids/risk.pid

python worker_triangle.py > logs/triangle.log 2>&1 &
echo $! > pids/triangle.pid

python worker_retest.py > logs/retest.log 2>&1 &
echo $! > pids/retest.pid

# 👇 ВАЖНО: добавляем TMPDIR
TMPDIR=/mnt/ssd_combined/ray_tmp python worker_rl.py > logs/rl.log 2>&1 &
echo $! > pids/rl.pid

sleep 5
python master_bot.py 2>&1 | tee logs/master.log