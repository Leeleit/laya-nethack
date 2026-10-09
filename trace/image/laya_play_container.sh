#!/usr/bin/env bash
# A persistent container of the laya-frozen image WITH a network (the arena's sandbox has none), so the games of
# laya_play.py reach laya_serve.py on the host: host.docker.internal -> its 127.0.0.1. Where the game container and
# the model server share a directory, a Unix socket there does instead (bc_vocab.address).
#   bash laya_play_container.sh <the working dir with trace/>
export MSYS_NO_PATHCONV=1
docker run -d --name laya-play --restart unless-stopped -v "$1":/work -w /work --entrypoint sleep laya-frozen infinity
