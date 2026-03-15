#!/bin/bash
for pid in pids/*.pid; do
    if [ -f "$pid" ]; then
        kill $(cat "$pid") 2>/dev/null
        rm "$pid"
    fi
done
pkill -f master_bot.py  # если нужно убить мастер-бота