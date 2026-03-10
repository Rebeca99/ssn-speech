# SSN-Speech: Cortical Microcircuit Model for Speech-Like Computation

A pure-SSN (Stabilised Supralinear Network) model that embeds a biologically
constrained cortical microcircuit (PYR/PV/SOM/VIP) in a sequence-processing
architecture to study how interneuron-specific perturbations produce
dissociable deficits in speech-relevant computations.

## Architecture

```
Embedding → Positional Encoding → [PureSSNLayer × 2] → Mean Pool → Classifier
```

- **No self-attention.** Temporal context propagates solely through the SOM
  carry-over state (slow causal inhibitory trace).
- **Dale's law enforced** on all recurrent connectivity via sign-constrained
  weight parameterisation.
- **4 populations:** PYR (excitatory), PV (fast inhibition), SOM (slow
  inhibition), VIP (disinhibition of SOM).
- **SSN dynamics:** `τ dx/dt = −x + [W·y + p]^n_+` iterated via Euler steps
  at each sequence position.

## Tasks

- **Task A (categorical sharpness):** Classify an ambiguous cue token embedded
  among noisy flankers. Tests phoneme boundary perception.
- **Task B1 (local):** Classify based on cue only; MODE token is irrelevant.
- **Task B2 (context-needed):** Cue is ambiguous; MODE token at position 0
  determines the correct label. Multiple distractor modes in the filler
  region. Tests temporal context integration.

## Perturbation Conditions

| Condition       | Perturbation       | Predicted deficit            |
|-----------------|--------------------|------------------------------|
| SOM suppressed  | k_SOM = 0          | Distractor vulnerability     |
| VIP suppressed  | k_VIP = 0          | Reduced context engagement   |
| Linear SOM      | n_SOM = 1          | Failure at long sequences    |
| PV weakened     | k_PV = 0.01        | Control (general)            |

## Repository Structure

```
ssn-speech/
├── README.md                  ← this file
├── pyproject.toml             ← project metadata & dependencies
├── setup.sh                   ← one-time environment setup
├── run_seed.py                ← main experiment script (1 seed)
├── submit_seeds.sh            ← SLURM array job (5 seeds parallel)
├── run_all_sequential.sh      ← run all seeds sequentially (no SLURM)
├── aggregate_seeds.py         ← aggregate results & generate figures
├── results/                   ← created at runtime
│   ├── seed_0/
│   │   ├── all_results.json
│   │   ├── baseline.pth
│   │   └── ssn_model.pth
│   ├── seed_1/
│   └── ...
├── figures/                   ← created by aggregate_seeds.py
│   ├── fig1_main_results.pdf
│   ├── fig2_som_trace.pdf
│   ├── fig3_per_task.pdf
│   ├── fig4_seq_length.pdf
│   └── fig5_gain_sweeps.pdf
└── logs/                      ← training logs per seed
```

## Quick Start

### 1. Clone and setup

```bash
git clone <your-repo-url>
cd ssn-speech
bash setup.sh
```

This installs `uv` (if needed), creates a virtual environment, and installs
PyTorch with CUDA 12.1 plus all other dependencies.

### 2. Run experiments

**Option A: SLURM cluster (parallel, ~30 min on A100)**

Edit `submit_seeds.sh` if your GPU partition has a different name (check with
`sinfo`), then:

```bash
sbatch submit_seeds.sh
```

This launches 5 jobs (seeds 0–4) in parallel, each on a separate GPU.
Monitor with `squeue -u $USER` or `tail -f logs/seed_0.out`.

**Option B: Single GPU machine (sequential, ~2.5 hrs on A100)**

```bash
bash run_all_sequential.sh
```

Runs all 5 seeds one after another on whatever GPU is available.

**Option C: Single seed (for testing)**

```bash
source .venv/bin/activate
python run_seed.py --seed 0 --outdir results/seed_0
```

### 3. Aggregate results and generate figures

After all seeds have finished:

```bash
source .venv/bin/activate
python aggregate_seeds.py --results_dir results --n_seeds 5 --outdir figures
```

This produces 5 publication-quality figures (PNG + PDF) with mean ± SEM
error bars across seeds, plus a summary table formatted for the paper.

## What each script does

### `run_seed.py`

Complete pipeline for one seed:

1. **Train** baseline transformer (12 epochs) and pure-SSN model (15 epochs)
2. **Perturbation evaluations:** Intact, SOM suppressed, VIP suppressed,
   Linear SOM, PV weakened, SOM+VIP off — measuring psychometric curves,
   context length curves, intrusion curves, B1/B2 accuracy
3. **SOM nonlinearity sweep:** n_SOM = 1.0 to 2.5
4. **VIP nonlinearity sweep:** n_VIP = 1.0 to 2.5
5. **Graded SOM gain sweep:** k_SOM = 0.0 to 0.04 (dose-response)
6. **Graded VIP gain sweep:** k_VIP = 0.0 to 0.04 (dose-response)
7. **Per-task evaluation:** separate accuracy for Task A, B1, B2
8. **Sequence length analysis:** B2 accuracy binned by sequence length
9. **SOM trace analysis:** linear probes decoding MODE identity from SOM
   carry-over state at each sequence position

Outputs: `all_results.json` (all metrics), `baseline.pth`, `ssn_model.pth`

### `aggregate_seeds.py`

Loads `all_results.json` from each seed directory and produces:

- **Fig 1:** 6-panel main results (psychometric, context length, intrusion,
  dissociation scatter, SOM nonlinearity sweep, B1 vs B2 bars)
- **Fig 2:** SOM trace decodability across sequence positions
- **Fig 3:** Per-task accuracy grouped bar chart
- **Fig 4:** B2 accuracy vs sequence length
- **Fig 5:** Graded gain dose-response curves (SOM and VIP)
- **Summary table** with mean ± SEM for all key metrics

## Estimated runtimes

| Hardware | Per seed | All 5 seeds (parallel) | All 5 seeds (sequential) |
|----------|----------|------------------------|--------------------------|
| A100     | ~30 min  | ~30 min                | ~2.5 hrs                 |
| V100     | ~1 hr    | ~1 hr                  | ~5 hrs                   |
| T4       | ~3 hrs   | ~3 hrs                 | ~15 hrs                  |

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 with CUDA
- NumPy, Matplotlib, scikit-learn, tqdm
