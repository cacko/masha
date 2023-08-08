#!/bin/zsh
#source ~/.zshrc
# direnv exec . mamba run --live-stream -n masha gunicorn 'masha.app:create_app()'
sudo direnv exec .  mamba run --live-stream -n masha uvicorn 'masha.app:create_app' --host 0.0.0.0 --port 23726 --workers 4  --log-level info --log-config log.ini --use-colors --factory --loop uvloop --access-log
