"""laya as a service for the games of laya_play.py: a request is a list of (state, question), the reply the raw
option logits of each, in order. Requests of all connected games are batched together on the GPU (by a token
budget, as predict.py), tokenized exactly as in training (build_training_item).

  python laya_serve.py <checkpoint dir> [--listen 127.0.0.1:6000 | --listen /data/laya/nethack/run/laya.sock]
(the training env: torch, laya; addresses: bc_vocab.address)
"""
import argparse
import os
import queue
import sys
import threading
import time
from multiprocessing.connection import Listener

import torch

from bc_vocab import address
from finetune_fast import load_model, logits_of
from finetune_single_device import build_training_item, collate

TOKENS = 32768
NO_GOLD = {"probabilities": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--listen", default="127.0.0.1:6000")
    ap.add_argument("--tokens", type=int, default=TOKENS, help="padded tokens per forward pass")
    a = ap.parse_args()
    device = torch.device("cuda", 0)
    cfg, tok, model = load_model(a.ckpt, device)
    if cfg.get("multi_question"):  # trained on all questions of a state per sequence; served one per sequence here
        sys.exit(f"{a.ckpt}: multi-question checkpoint, laya_serve asks one question per pass")
    model.eval()
    requests = queue.Queue()

    def client(conn):
        try:
            while True:
                requests.put((conn, conn.recv()))
        except (EOFError, ConnectionError, OSError):
            conn.close()

    if "/" in a.listen and os.path.exists(a.listen):
        os.unlink(a.listen)  # a socket file left by an earlier server
    # the default backlog of 1 refuses simultaneous connects on Windows (the games start together)
    listener = Listener(address(a.listen), backlog=64, authkey=b"laya")
    threading.Thread(target=lambda: [threading.Thread(target=client, args=(listener.accept(),), daemon=True).start()
                                     for _ in iter(int, 1)], daemon=True).start()
    print(f"serving {a.ckpt} (max_len {cfg['max_len']}) on {a.listen}", flush=True)
    served, t_log = 0, time.time()
    with torch.no_grad():
        while True:
            pending = [requests.get()]
            while True:  # whatever else is waiting goes into the same passes
                try:
                    pending.append(requests.get_nowait())
                except queue.Empty:
                    break
            items, owner = [], []
            for r, (conn, req) in enumerate(pending):
                for state, q in req:
                    items.append(build_training_item(tok, cfg, state, q, NO_GOLD))
                    owner.append(r)
            logits = [None] * len(items)
            order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
            b = []
            for i in order + [None]:
                if b and (i is None or (len(b) + 1) * len(items[i]["ids"]) > a.tokens):
                    z = logits_of(model, collate([items[j] for j in b], tok.pad_token_id), device).cpu()
                    for row, j in enumerate(b):
                        logits[j] = z[row, :len(items[j]["markers"])].tolist()
                    b = []
                if i is not None:
                    b.append(i)
            out = [[] for _ in pending]
            for j, r in enumerate(owner):
                out[r].append(logits[j])
            for (conn, _), res in zip(pending, out):
                try:
                    conn.send(res)
                except (ConnectionError, OSError):
                    pass
            served += len(items)
            if time.time() - t_log > 60:
                print(f"{served / (time.time() - t_log):.1f} questions/s", flush=True)
                served, t_log = 0, time.time()


if __name__ == "__main__":
    main()
