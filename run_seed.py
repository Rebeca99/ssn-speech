#!/usr/bin/env python3
"""
Pure-SSN Multi-Seed Experiment Runner
=====================================

Usage:
    python run_seed.py --seed 0 --outdir results/seed_0
    python run_seed.py --seed 1 --outdir results/seed_1
    ...

Or via SLURM array job (see submit_seeds.sh).

This script runs the FULL pipeline for one seed:
  1. Train baseline transformer + pure-SSN model
  2. All perturbation evaluations
  3. SOM nonlinearity sweep
  4. SOM trace analysis (decodability probes)
  5. Per-task evaluation
  6. Sequence length analysis

All results saved to --outdir as .pth and .json files.
No plotting — aggregation and plotting is done by aggregate_seeds.py.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ── parse args ─────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--epochs_baseline", type=int, default=12)
parser.add_argument("--epochs_ssn", type=int, default=15)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_val", type=int, default=8000)
parser.add_argument("--n_eval", type=int, default=3000,
                    help="samples per eval condition")
parser.add_argument("--n_trace", type=int, default=2000,
                    help="samples for SOM trace analysis")
args = parser.parse_args()

if args.outdir is None:
    args.outdir = f"results/seed_{args.seed}"
os.makedirs(args.outdir, exist_ok=True)

if args.device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = args.device

# ── set seeds ──────────────────────────────────────────────────

def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(args.seed)
print(f"Seed: {args.seed}  Device: {device}  Output: {args.outdir}")


# ════════════════════════════════════════════════════════════════
# 1.  DATASET
# ════════════════════════════════════════════════════════════════

@dataclass
class TaskConfig:
    N_CUE: int = 8
    FILLER_VOCAB: int = 10
    MIN_FILL: int = 6
    MAX_FILL: int = 18
    AMBIG_CUE: int = 3
    B2_CUE_SET: tuple = (2, 3, 4, 5)
    B1_CUE_SET: tuple = (2, 3, 4, 5)
    BOUNDARY: int = 3
    P_CUE_JITTER: float = 0.15
    P_TOKEN_DROPOUT: float = 0.05
    B1_CUE_WEIGHT: float = 0.8
    B1_LABEL_NOISE: float = 0.05
    B2_CUE_WEIGHT: float = 0.3
    B2_MODE_WEIGHT: float = 2.5
    B2_LABEL_NOISE: float = 0.03
    A_N_FLANKERS: int = 4
    A_CUE_POSITION: str = "random"
    A_BOUNDARY_SOFT: float = 0.8
    B_N_EXTRA_DISTRACTORS: int = 2
    B_DISTRACTOR_SAME_PROB: float = 0.5

CFG = TaskConfig()

def build_vocab(cfg):
    PAD, MASK, MODE_A, MODE_B, VOWEL = 0, 1, 2, 3, 4
    CUE0 = 5; FILL0 = CUE0 + cfg.N_CUE
    return dict(
        PAD=PAD, MASK=MASK, MODE_A=MODE_A, MODE_B=MODE_B, VOWEL=VOWEL,
        CUE0=CUE0, FILL0=FILL0,
        cue_tokens=list(range(CUE0, CUE0 + cfg.N_CUE)),
        filler_tokens=list(range(FILL0, FILL0 + cfg.FILLER_VOCAB)),
        vocab_size=FILL0 + cfg.FILLER_VOCAB,
    )

VOC = build_vocab(CFG)

def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def _maybe_jitter(k, cfg):
    if random.random() < cfg.P_CUE_JITTER:
        k = max(0, min(cfg.N_CUE - 1, k + random.choice([-1, 1])))
    return k
def _maybe_dropout(tok, cfg):
    if cfg.P_TOKEN_DROPOUT > 0 and random.random() < cfg.P_TOKEN_DROPOUT:
        return VOC["MASK"]
    return tok
def _sample_b1_label(cue_idx, cfg):
    logit = cfg.B1_CUE_WEIGHT * (cue_idx - cfg.BOUNDARY)
    p = _sigmoid(logit)
    if cfg.B1_LABEL_NOISE > 0:
        p = (1 - cfg.B1_LABEL_NOISE) * p + cfg.B1_LABEL_NOISE * 0.5
    return (1 if random.random() < p else 0), p
def _sample_b2_label(cue_idx, mode, cfg):
    bias = -cfg.B2_MODE_WEIGHT if mode == VOC["MODE_A"] else cfg.B2_MODE_WEIGHT
    logit = cfg.B2_CUE_WEIGHT * (cue_idx - cfg.BOUNDARY) + bias
    p = _sigmoid(logit)
    if cfg.B2_LABEL_NOISE > 0:
        p = (1 - cfg.B2_LABEL_NOISE) * p + cfg.B2_LABEL_NOISE * 0.5
    return (1 if random.random() < p else 0), p
def _sample_fillers(cfg):
    L = random.randint(cfg.MIN_FILL, cfg.MAX_FILL)
    return [random.choice(VOC["filler_tokens"]) for _ in range(L)]

def make_taskA(cfg):
    k = random.randint(0, cfg.N_CUE - 1)
    k = _maybe_jitter(k, cfg)
    cue_tok = _maybe_dropout(VOC["CUE0"] + k, cfg)
    n_fl = cfg.A_N_FLANKERS
    flankers = [random.choice(VOC["filler_tokens"]) for _ in range(n_fl)]
    pos = random.randint(0, n_fl)
    seq = flankers[:pos] + [cue_tok] + flankers[pos:] + [VOC["VOWEL"]]
    logit = cfg.A_BOUNDARY_SOFT * (k - cfg.BOUNDARY)
    p_pa = _sigmoid(logit)
    y = 1 if random.random() < p_pa else 0
    return seq, y, k

def make_taskB(cfg, context_needed, distractor_lag=None):
    mode = VOC["MODE_A"] if random.random() < 0.5 else VOC["MODE_B"]
    opposite = VOC["MODE_B"] if mode == VOC["MODE_A"] else VOC["MODE_A"]
    fillers = _sample_fillers(cfg)
    for _ in range(cfg.B_N_EXTRA_DISTRACTORS):
        if len(fillers) < 2: break
        dist_tok = mode if random.random() < cfg.B_DISTRACTOR_SAME_PROB else opposite
        p = random.randint(0, len(fillers) - 1)
        fillers = fillers[:p] + [dist_tok] + fillers[p:]
    if distractor_lag is not None:
        p = max(0, len(fillers) - distractor_lag)
        fillers = fillers[:p] + [opposite] + fillers[p:]
    cue_set = cfg.B2_CUE_SET if context_needed else cfg.B1_CUE_SET
    k = random.choice(cue_set)
    k = _maybe_jitter(k, cfg)
    k = max(0, min(cfg.N_CUE - 1, k))
    tok = _maybe_dropout(VOC["CUE0"] + k, cfg)
    x = [mode] + fillers + [tok, VOC["VOWEL"]]
    if context_needed:
        y, _ = _sample_b2_label(k, mode, cfg)
    else:
        y, _ = _sample_b1_label(k, cfg)
    return x, y, k

def pad_batch(seqs, pad_id=0):
    ml = max(len(s) for s in seqs)
    x = torch.full((len(seqs), ml), pad_id, dtype=torch.long)
    attn = torch.zeros(len(seqs), ml, dtype=torch.bool)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
        attn[i, :len(s)] = True
    return x, attn

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, n, task_mix=("A", "B1", "B2")):
        self.n, self.mix = n, task_mix
    def __len__(self): return self.n
    def __getitem__(self, _):
        t = random.choice(self.mix)
        if t == "A":    x, y, k = make_taskA(CFG)
        elif t == "B1": x, y, k = make_taskB(CFG, False)
        else:           x, y, k = make_taskB(CFG, True)
        return x, y, {"task": t, "cue_k": k}

def collate_fn(batch):
    xs, ys, ms = zip(*batch)
    x, a = pad_batch(xs)
    return x, a, torch.tensor(ys, dtype=torch.long), ms


# ════════════════════════════════════════════════════════════════
# 2.  MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════

class DaleLinear(nn.Module):
    def __init__(self, in_f, out_f, sign=+1):
        super().__init__()
        assert sign in (+1, -1)
        self.sign = sign
        self.raw_weight = nn.Parameter(torch.randn(out_f, in_f) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_f))
    def forward(self, x):
        return F.linear(x, self.sign * F.softplus(self.raw_weight), self.bias)
    @property
    def effective_weight(self):
        with torch.no_grad():
            return self.sign * F.softplus(self.raw_weight)

@dataclass
class SSNConfig:
    frac_E: float = 0.60; frac_PV: float = 0.18
    frac_SOM: float = 0.14; frac_VIP: float = 0.08
    n_E: float = 2.0; n_PV: float = 2.0
    n_SOM: float = 2.0; n_VIP: float = 2.0
    k_E: float = 0.04; k_PV: float = 0.04
    k_SOM: float = 0.04; k_VIP: float = 0.04
    tau_E: float = 1.0; tau_PV: float = 0.5
    tau_SOM: float = 3.0; tau_VIP: float = 0.8
    K_steps: int = 6; dt: float = 0.3
    som_carry_alpha: float = 0.85

class PureSSNLayer(nn.Module):
    def __init__(self, d_model, ssn_cfg=None):
        super().__init__()
        cfg = ssn_cfg or SSNConfig()
        self.cfg = cfg; self.d_model = d_model
        self.d_E = int(d_model * cfg.frac_E)
        self.d_PV = int(d_model * cfg.frac_PV)
        self.d_SOM = int(d_model * cfg.frac_SOM)
        self.d_VIP = d_model - self.d_E - self.d_PV - self.d_SOM
        assert self.d_VIP > 0
        self.inp_E = nn.Linear(d_model, self.d_E)
        self.inp_PV = nn.Linear(d_model, self.d_PV)
        self.inp_SOM = nn.Linear(d_model, self.d_SOM)
        self.inp_VIP = nn.Linear(d_model, self.d_VIP)
        self.W_EE = DaleLinear(self.d_E, self.d_E, +1)
        self.W_PVE = DaleLinear(self.d_E, self.d_PV, +1)
        self.W_SE = DaleLinear(self.d_E, self.d_SOM, +1)
        self.W_VE = DaleLinear(self.d_E, self.d_VIP, +1)
        self.W_EPV = DaleLinear(self.d_PV, self.d_E, -1)
        self.W_PVPV = DaleLinear(self.d_PV, self.d_PV, -1)
        self.W_SPV = DaleLinear(self.d_PV, self.d_SOM, -1)
        self.W_ES = DaleLinear(self.d_SOM, self.d_E, -1)
        self.W_PVS = DaleLinear(self.d_SOM, self.d_PV, -1)
        self.W_VS = DaleLinear(self.d_SOM, self.d_VIP, -1)
        self.W_SV = DaleLinear(self.d_VIP, self.d_SOM, -1)
        d_total = self.d_E + self.d_PV + self.d_SOM + self.d_VIP
        self.output_proj = nn.Linear(d_total, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _act(x, k, n):
        return k * F.relu(x).pow(n)

    def _euler_step(self, state, p, cfg, perturb):
        k_E = perturb.get("k_E", cfg.k_E)
        k_PV = perturb.get("k_PV", cfg.k_PV)
        k_SOM = perturb.get("k_SOM", cfg.k_SOM)
        k_VIP = perturb.get("k_VIP", cfg.k_VIP)
        n_SOM = perturb.get("n_SOM", cfg.n_SOM)
        n_VIP = perturb.get("n_VIP", cfg.n_VIP)
        y_E = self._act(state["x_E"], k_E, cfg.n_E)
        y_PV = self._act(state["x_PV"], k_PV, cfg.n_PV)
        y_SOM = self._act(state["x_SOM"], k_SOM, n_SOM)
        y_VIP = self._act(state["x_VIP"], k_VIP, n_VIP)
        inp_E = self.W_EE(y_E) + self.W_EPV(y_PV) + self.W_ES(y_SOM) + p["E"]
        inp_PV = self.W_PVE(y_E) + self.W_PVPV(y_PV) + self.W_PVS(y_SOM) + p["PV"]
        inp_SOM = self.W_SE(y_E) + self.W_SPV(y_PV) + self.W_SV(y_VIP) + p["SOM"]
        inp_VIP = self.W_VE(y_E) + self.W_VS(y_SOM) + p["VIP"]
        dt = cfg.dt
        state["x_E"] += dt / cfg.tau_E * (-state["x_E"] + inp_E)
        state["x_PV"] += dt / cfg.tau_PV * (-state["x_PV"] + inp_PV)
        state["x_SOM"] += dt / cfg.tau_SOM * (-state["x_SOM"] + inp_SOM)
        state["x_VIP"] += dt / cfg.tau_VIP * (-state["x_VIP"] + inp_VIP)
        return state

    def forward(self, h, attn_mask, perturb=None):
        perturb = perturb or {}
        cfg = self.cfg; B, T, D = h.shape
        alpha = cfg.som_carry_alpha
        p_E = self.inp_E(h); p_PV = self.inp_PV(h)
        p_SOM = self.inp_SOM(h); p_VIP = self.inp_VIP(h)
        som_carry = torch.zeros(B, self.d_SOM, device=h.device)
        outputs = []
        for t in range(T):
            mask_t = attn_mask[:, t].float().unsqueeze(-1)
            p_t = {"E": p_E[:, t], "PV": p_PV[:, t],
                   "SOM": p_SOM[:, t], "VIP": p_VIP[:, t]}
            state = {"x_E": p_t["E"].clone(), "x_PV": p_t["PV"].clone(),
                     "x_SOM": alpha * som_carry + (1 - alpha) * p_t["SOM"],
                     "x_VIP": p_t["VIP"].clone()}
            for _ in range(cfg.K_steps):
                state = self._euler_step(state, p_t, cfg, perturb)
            k_som = perturb.get("k_SOM", cfg.k_SOM)
            n_som = perturb.get("n_SOM", cfg.n_SOM)
            k_vip = perturb.get("k_VIP", cfg.k_VIP)
            n_vip = perturb.get("n_VIP", cfg.n_VIP)
            y_E = self._act(state["x_E"], cfg.k_E, cfg.n_E)
            y_PV = self._act(state["x_PV"], cfg.k_PV, cfg.n_PV)
            y_SOM = self._act(state["x_SOM"], k_som, n_som)
            y_VIP = self._act(state["x_VIP"], k_vip, n_vip)
            outputs.append(torch.cat([y_E, y_PV, y_SOM, y_VIP], dim=-1))
            som_carry = (alpha * som_carry + (1 - alpha) * state["x_SOM"]) * mask_t \
                        + som_carry * (1 - mask_t)
        h_ssn = torch.stack(outputs, dim=1)
        return self.layer_norm(h + self.output_proj(h_ssn))

class PureSSNClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2,
                 n_classes=2, ssn_cfg=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        pe = torch.zeros(512, d_model)
        pos = torch.arange(512).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        ssn_cfg = ssn_cfg or SSNConfig()
        self.layers = nn.ModuleList([
            PureSSNLayer(d_model, ssn_cfg) for _ in range(n_layers)])
        self.cls = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))
    def forward(self, x, attn_mask, perturb=None):
        h = self.emb(x) + self.pe[:, :x.size(1)]
        for layer in self.layers:
            h = layer(h, attn_mask, perturb=perturb)
        mask = attn_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.cls(pooled)

class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4,
                 n_layers=2, dropout=0.1, n_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        pe = torch.zeros(512, d_model)
        pos = torch.arange(512).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=4*d_model, dropout=dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.cls = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))
    def forward(self, x, attn_mask):
        h = self.emb(x) + self.pe[:, :x.size(1)]
        h = self.enc(h, src_key_padding_mask=~attn_mask)
        mask = attn_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.cls(pooled)


# ════════════════════════════════════════════════════════════════
# 3.  TRAINING
# ════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, epochs, lr=3e-4):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    for ep in range(1, epochs + 1):
        model.train(); losses = []
        for x, attn, y, _ in tqdm(train_loader, desc=f"ep {ep}", leave=False):
            x, attn, y = x.to(device), attn.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x, attn)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(loss.item())
        sched.step()
        val_acc = eval_acc(model, val_loader)
        ep_loss = float(np.mean(losses))
        history.append({"epoch": ep, "loss": ep_loss, "val_acc": val_acc})
        print(f"  ep {ep:2d}  loss={ep_loss:.4f}  val_acc={val_acc:.3f}")
    return history


# ════════════════════════════════════════════════════════════════
# 4.  EVALUATION FUNCTIONS
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_acc(model, loader, perturb=None):
    model.eval(); correct = total = 0
    for x, attn, y, _ in loader:
        x, attn = x.to(device), attn.to(device)
        logits = model(x, attn, perturb=perturb) if perturb else model(x, attn)
        correct += (logits.argmax(-1).cpu() == y).sum().item()
        total += len(y)
    return correct / total

@torch.no_grad()
def psychometric_curve(model, n_per_k=500, perturb=None):
    model.eval(); probs = []
    for k in range(CFG.N_CUE):
        xs = [[VOC["CUE0"] + k, VOC["VOWEL"]] for _ in range(n_per_k)]
        xp, at = pad_batch(xs); xp, at = xp.to(device), at.to(device)
        logits = model(xp, at, perturb=perturb) if perturb else model(xp, at)
        probs.append(F.softmax(logits, -1)[:, 1].mean().item())
    return probs

@torch.no_grad()
def context_length_curve(model, n=1500, Ls=(2,4,6,8,12,16,24,32), perturb=None):
    model.eval(); accs = []
    for L in Ls:
        xs, ys = [], []
        for _ in range(n):
            x, y, _ = make_taskB(CFG, True)
            xs.append(x[-L:] if len(x) > L else x); ys.append(y)
        xp, at = pad_batch(xs); xp, at = xp.to(device), at.to(device)
        logits = model(xp, at, perturb=perturb) if perturb else model(xp, at)
        accs.append(float((logits.argmax(-1).cpu().numpy() == np.array(ys)).mean()))
    return list(Ls), accs

@torch.no_grad()
def intrusion_curve(model, n=1500, lags=(0,2,4,6,8,12,16), perturb=None):
    model.eval(); accs = []
    for lag in lags:
        xs, ys = [], []
        for _ in range(n):
            x, y, _ = make_taskB(CFG, True, distractor_lag=lag)
            xs.append(x); ys.append(y)
        xp, at = pad_batch(xs); xp, at = xp.to(device), at.to(device)
        logits = model(xp, at, perturb=perturb) if perturb else model(xp, at)
        accs.append(float((logits.argmax(-1).cpu().numpy() == np.array(ys)).mean()))
    return list(lags), accs

@torch.no_grad()
def eval_B1_B2(model, n=3000, perturb=None):
    model.eval()
    def _acc(ctx):
        xs, ys = [], []
        for _ in range(n):
            x, y, _ = make_taskB(CFG, ctx)
            xs.append(x); ys.append(y)
        xp, at = pad_batch(xs); xp, at = xp.to(device), at.to(device)
        logits = model(xp, at, perturb=perturb) if perturb else model(xp, at)
        return float((logits.argmax(-1).cpu().numpy() == np.array(ys)).mean())
    b1, b2 = _acc(False), _acc(True)
    return b1, b2, b2 - b1

@torch.no_grad()
def eval_per_task(model, n=3000, perturb=None):
    model.eval(); r = {}
    for task_name, gen_fn in [("A", lambda: make_taskA(CFG)),
                               ("B1", lambda: make_taskB(CFG, False)),
                               ("B2", lambda: make_taskB(CFG, True))]:
        xs, ys = [], []
        for _ in range(n):
            x, y, _ = gen_fn()
            xs.append(x); ys.append(y)
        xp, at = pad_batch(xs); xp, at = xp.to(device), at.to(device)
        logits = model(xp, at, perturb=perturb) if perturb else model(xp, at)
        r[task_name] = float((logits.argmax(-1).cpu().numpy() == np.array(ys)).mean())
    return r

@torch.no_grad()
def eval_b2_by_length(model, n=5000, perturb=None):
    model.eval()
    xs, ys, lens = [], [], []
    for _ in range(n):
        x, y, _ = make_taskB(CFG, True)
        xs.append(x); ys.append(y); lens.append(len(x))
    lens = np.array(lens); ys = np.array(ys)
    xp, at = pad_batch(xs); xp, at = xp.to(device), at.to(device)
    logits = model(xp, at, perturb=perturb) if perturb else model(xp, at)
    preds = logits.argmax(-1).cpu().numpy()
    correct = (preds == ys)
    bins = [("short_8_12", 8, 12), ("medium_13_17", 13, 17),
            ("long_18_22", 18, 22), ("vlong_23p", 23, 100)]
    r = {}
    for label, lo, hi in bins:
        mask = (lens >= lo) & (lens <= hi)
        r[label] = {"acc": float(correct[mask].mean()) if mask.sum() >= 20 else None,
                     "n": int(mask.sum())}
    return r


# ════════════════════════════════════════════════════════════════
# 5.  SOM TRACE ANALYSIS
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_and_probe_som(model, n=2000, perturb=None, max_pos=25):
    """Extract SOM traces and run linear probes. Returns decodability curve."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    model.eval()
    xs, mode_labels = [], []
    for _ in range(n):
        mode = VOC["MODE_A"] if random.random() < 0.5 else VOC["MODE_B"]
        opposite = VOC["MODE_B"] if mode == VOC["MODE_A"] else VOC["MODE_A"]
        fillers = _sample_fillers(CFG)
        for _ in range(CFG.B_N_EXTRA_DISTRACTORS):
            if len(fillers) < 2: break
            dt = mode if random.random() < CFG.B_DISTRACTOR_SAME_PROB else opposite
            p = random.randint(0, len(fillers) - 1)
            fillers = fillers[:p] + [dt] + fillers[p:]
        k = random.choice(CFG.B2_CUE_SET)
        k = _maybe_jitter(k, CFG); k = max(0, min(CFG.N_CUE - 1, k))
        tok = _maybe_dropout(VOC["CUE0"] + k, CFG)
        xs.append([mode] + fillers + [tok, VOC["VOWEL"]])
        mode_labels.append(0 if mode == VOC["MODE_A"] else 1)

    seq_lens = np.array([len(s) for s in xs])
    modes = np.array(mode_labels)
    xpad, attn = pad_batch(xs)
    xpad, attn = xpad.to(device), attn.to(device)
    B, T = xpad.shape

    # manual forward collecting SOM traces from last layer
    h = model.emb(xpad) + model.pe[:, :T]
    for layer_idx, layer in enumerate(model.layers):
        cfg = layer.cfg; alpha = cfg.som_carry_alpha
        perturb_dict = perturb or {}
        p_E = layer.inp_E(h); p_PV = layer.inp_PV(h)
        p_SOM = layer.inp_SOM(h); p_VIP = layer.inp_VIP(h)
        som_carry = torch.zeros(B, layer.d_SOM, device=device)
        outputs, som_traces = [], []
        for t in range(T):
            mask_t = attn[:, t].float().unsqueeze(-1)
            p_t = {"E": p_E[:, t], "PV": p_PV[:, t],
                   "SOM": p_SOM[:, t], "VIP": p_VIP[:, t]}
            state = {"x_E": p_t["E"].clone(), "x_PV": p_t["PV"].clone(),
                     "x_SOM": alpha * som_carry + (1 - alpha) * p_t["SOM"],
                     "x_VIP": p_t["VIP"].clone()}
            for _ in range(cfg.K_steps):
                state = layer._euler_step(state, p_t, cfg, perturb_dict)
            k_som = perturb_dict.get("k_SOM", cfg.k_SOM)
            n_som = perturb_dict.get("n_SOM", cfg.n_SOM)
            k_vip = perturb_dict.get("k_VIP", cfg.k_VIP)
            n_vip = perturb_dict.get("n_VIP", cfg.n_VIP)
            y_E = layer._act(state["x_E"], cfg.k_E, cfg.n_E)
            y_PV = layer._act(state["x_PV"], cfg.k_PV, cfg.n_PV)
            y_SOM = layer._act(state["x_SOM"], k_som, n_som)
            y_VIP = layer._act(state["x_VIP"], k_vip, n_vip)
            outputs.append(torch.cat([y_E, y_PV, y_SOM, y_VIP], dim=-1))
            som_carry = (alpha * som_carry + (1 - alpha) * state["x_SOM"]) * mask_t \
                        + som_carry * (1 - mask_t)
            som_traces.append(som_carry.cpu().numpy().copy())
        h_ssn = torch.stack(outputs, dim=1)
        h = layer.layer_norm(h + layer.output_proj(h_ssn))

    traces = np.stack(som_traces, axis=0).transpose(1, 0, 2)  # (B, T, d_SOM)

    # linear probes
    positions, accs = [], []
    for t in range(min(max_pos, T)):
        mask = seq_lens > t
        if mask.sum() < 50: accs.append(0.5); positions.append(t); continue
        X = traces[mask, t, :]; y = modes[mask]
        if len(np.unique(y)) < 2: accs.append(0.5); positions.append(t); continue
        n_use = len(y); n_tr = int(0.8 * n_use)
        idx = np.random.permutation(n_use)
        X_tr, X_te = X[idx[:n_tr]], X[idx[n_tr:]]
        y_tr, y_te = y[idx[:n_tr]], y[idx[n_tr:]]
        if len(np.unique(y_te)) < 2: accs.append(0.5); positions.append(t); continue
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X_tr, y_tr)
        accs.append(float(accuracy_score(y_te, clf.predict(X_te))))
        positions.append(t)

    return {"positions": positions, "decodability": accs}


# ════════════════════════════════════════════════════════════════
# 6.  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════

def main():
    all_results = {"seed": args.seed, "config": asdict(CFG)}

    # ── data ───────────────────────────────────────────────────
    train_ds = ToyDataset(args.n_train)
    val_ds = ToyDataset(args.n_val)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # ── train baseline ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  Training BASELINE (seed={args.seed})")
    print("=" * 50)
    baseline = BaselineTransformer(VOC["vocab_size"]).to(device)
    hist_base = train_model(baseline, train_loader, val_loader, args.epochs_baseline)
    torch.save(baseline.state_dict(), os.path.join(args.outdir, "baseline.pth"))
    all_results["baseline_history"] = hist_base

    # ── train pure-SSN ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  Training PURE-SSN (seed={args.seed})")
    print("=" * 50)
    ssn_cfg = SSNConfig(K_steps=6)
    ssn_model = PureSSNClassifier(VOC["vocab_size"], ssn_cfg=ssn_cfg).to(device)
    hist_ssn = train_model(ssn_model, train_loader, val_loader, args.epochs_ssn)
    torch.save(ssn_model.state_dict(), os.path.join(args.outdir, "ssn_model.pth"))
    all_results["ssn_history"] = hist_ssn

    # ── perturbation conditions ────────────────────────────────
    conditions = {
        "Intact":         {},
        "SOM_suppressed": {"k_SOM": 0.0},
        "VIP_suppressed": {"k_VIP": 0.0},
        "Linear_SOM":     {"n_SOM": 1.0},
        "PV_weakened":    {"k_PV": 0.01},
        "SOM_VIP_off":    {"k_SOM": 0.0, "k_VIP": 0.0},
    }

    n_eval = args.n_eval

    # ── SSN perturbation evaluations ───────────────────────────
    print("\n" + "=" * 50)
    print("  Perturbation evaluations (SSN)")
    print("=" * 50)
    perturbation_results = {}
    for cname, cpert in conditions.items():
        print(f"  {cname}...")
        p = cpert if cpert else None
        psych = psychometric_curve(ssn_model, perturb=p)
        Ls, accL = context_length_curve(ssn_model, n=n_eval, perturb=p)
        lags, acclag = intrusion_curve(ssn_model, n=n_eval, perturb=p)
        b1, b2, benefit = eval_B1_B2(ssn_model, n=n_eval, perturb=p)
        intr_idx = float(acclag[0] - acclag[-1])
        perturbation_results[cname] = {
            "psych": psych, "Ls": Ls, "accL": accL,
            "lags": lags, "acclag": acclag,
            "b1": b1, "b2": b2, "benefit": benefit,
            "intrusion_idx": intr_idx,
        }
        print(f"    B1={b1:.3f} B2={b2:.3f} benefit={benefit:+.3f} "
              f"intrusion={intr_idx:+.3f}")

    # ── baseline evaluation ────────────────────────────────────
    print("  Baseline...")
    psych_b = psychometric_curve(baseline)
    Ls_b, accL_b = context_length_curve(baseline, n=n_eval)
    lags_b, acclag_b = intrusion_curve(baseline, n=n_eval)
    b1_b, b2_b, ben_b = eval_B1_B2(baseline, n=n_eval)
    perturbation_results["Baseline"] = {
        "psych": psych_b, "Ls": Ls_b, "accL": accL_b,
        "lags": lags_b, "acclag": acclag_b,
        "b1": b1_b, "b2": b2_b, "benefit": ben_b,
        "intrusion_idx": float(acclag_b[0] - acclag_b[-1]),
    }
    all_results["perturbation"] = perturbation_results

    # ── SOM nonlinearity sweep ─────────────────────────────────
    print("\n  SOM nonlinearity sweep...")
    som_sweep = {}
    for nv in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
        p = {"n_SOM": nv}
        _, _, ben = eval_B1_B2(ssn_model, n=2000, perturb=p)
        _, acl = intrusion_curve(ssn_model, n=1000, lags=(0, 4, 8, 16), perturb=p)
        som_sweep[str(nv)] = {"benefit": ben,
                               "intrusion": float(acl[0] - acl[-1])}
        print(f"    n_SOM={nv:.2f}  benefit={ben:+.3f}  "
              f"intrusion={float(acl[0]-acl[-1]):+.3f}")
    all_results["som_sweep"] = som_sweep

    # ── VIP nonlinearity sweep ─────────────────────────────────
    print("\n  VIP nonlinearity sweep...")
    vip_sweep = {}
    for nv in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
        p = {"n_VIP": nv}
        _, _, ben = eval_B1_B2(ssn_model, n=2000, perturb=p)
        _, acl = intrusion_curve(ssn_model, n=1000, lags=(0, 4, 8, 16), perturb=p)
        vip_sweep[str(nv)] = {"benefit": ben,
                               "intrusion": float(acl[0] - acl[-1])}
        print(f"    n_VIP={nv:.2f}  benefit={ben:+.3f}  "
              f"intrusion={float(acl[0]-acl[-1]):+.3f}")
    all_results["vip_sweep"] = vip_sweep

    # ── graded SOM gain sweep ──────────────────────────────────
    print("\n  Graded SOM gain sweep...")
    som_gain_sweep = {}
    for kv in [0.0, 0.005, 0.01, 0.02, 0.03, 0.04]:
        p = {"k_SOM": kv}
        _, _, ben = eval_B1_B2(ssn_model, n=2000, perturb=p)
        _, acl = intrusion_curve(ssn_model, n=1000, lags=(0, 4, 8, 16), perturb=p)
        som_gain_sweep[str(kv)] = {"benefit": ben,
                                    "intrusion": float(acl[0] - acl[-1])}
        print(f"    k_SOM={kv:.3f}  benefit={ben:+.3f}  "
              f"intrusion={float(acl[0]-acl[-1]):+.3f}")
    all_results["som_gain_sweep"] = som_gain_sweep

    # ── graded VIP gain sweep ──────────────────────────────────
    print("\n  Graded VIP gain sweep...")
    vip_gain_sweep = {}
    for kv in [0.0, 0.005, 0.01, 0.02, 0.03, 0.04]:
        p = {"k_VIP": kv}
        _, _, ben = eval_B1_B2(ssn_model, n=2000, perturb=p)
        _, acl = intrusion_curve(ssn_model, n=1000, lags=(0, 4, 8, 16), perturb=p)
        vip_gain_sweep[str(kv)] = {"benefit": ben,
                                    "intrusion": float(acl[0] - acl[-1])}
        print(f"    k_VIP={kv:.3f}  benefit={ben:+.3f}  "
              f"intrusion={float(acl[0]-acl[-1]):+.3f}")
    all_results["vip_gain_sweep"] = vip_gain_sweep

    # ── per-task evaluation ────────────────────────────────────
    print("\n  Per-task evaluation...")
    pertask = {}
    for cname, cpert in list(conditions.items()) + [("Baseline", None)]:
        if cname == "Baseline":
            pertask[cname] = eval_per_task(baseline, n=n_eval)
        else:
            p = cpert if cpert else None
            pertask[cname] = eval_per_task(ssn_model, n=n_eval, perturb=p)
        print(f"    {cname}: {pertask[cname]}")
    all_results["per_task"] = pertask

    # ── sequence length analysis ───────────────────────────────
    print("\n  Sequence length analysis...")
    seqlen = {}
    for cname, cpert in list(conditions.items()) + [("Baseline", None)]:
        if cname == "Baseline":
            seqlen[cname] = eval_b2_by_length(baseline, n=5000)
        else:
            p = cpert if cpert else None
            seqlen[cname] = eval_b2_by_length(ssn_model, n=5000, perturb=p)
        print(f"    {cname}: {seqlen[cname]}")
    all_results["seq_length"] = seqlen

    # ── SOM trace analysis ─────────────────────────────────────
    print("\n  SOM trace analysis...")
    trace_results = {}
    for cname, cpert in [("Intact", {}), ("SOM_suppressed", {"k_SOM": 0.0}),
                          ("VIP_suppressed", {"k_VIP": 0.0}),
                          ("Linear_SOM", {"n_SOM": 1.0})]:
        p = cpert if cpert else None
        trace_results[cname] = extract_and_probe_som(
            ssn_model, n=args.n_trace, perturb=p)
        accs = trace_results[cname]["decodability"]
        print(f"    {cname}: early={np.mean(accs[1:6]):.3f} "
              f"mid={np.mean(accs[6:13]):.3f} late={np.mean(accs[13:]):.3f}")
    all_results["som_trace"] = trace_results

    # ── save everything ────────────────────────────────────────
    out_path = os.path.join(args.outdir, "all_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  All results saved to {out_path}")
    print(f"  Model checkpoints in {args.outdir}/")
    print("  DONE.")


if __name__ == "__main__":
    main()
