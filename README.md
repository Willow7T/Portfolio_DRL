# Deep Reinforcement Learning for Portfolio Allocation

This project explores whether deep reinforcement learning (DRL) can make useful portfolio allocation decisions from historical market data. It builds daily asset returns and market features, trains agents to choose portfolio weights, and compares their out-of-sample performance with conventional portfolio strategies.

The research code includes:

- Proximal Policy Optimization (PPO) and Deep Q-Network (DQN) agents, built with Stable-Baselines3 and a custom Gymnasium portfolio environment.
- A multi-asset LSTM analysis in a notebook, alongside DQN/PPO comparisons and an expanded multi-regime experiment.
- Data preparation, chronological train/validation/test splits, transaction-cost-aware evaluation, portfolio constraints, benchmark strategies, and performance reports.
- Three experiment configurations: a small five-stock baseline, a diversified ETF universe, and an expanded sector-balanced stock and defensive-asset universe with regime features.

The results are for research and educational use. Historical backtests do not guarantee future performance. In particular, the expanded stock universe uses Yahoo Finance data and does not have point-in-time index membership or complete delisting returns, so treat its results as exploratory.

## Requirements

- Python 3.11 or later.
- The packages and versions pinned in [`requirements.txt`](requirements.txt).
- An NVIDIA GPU and a driver compatible with the CUDA 13.2 PyTorch wheels used by the requirements file for the supplied training configurations. PyTorch policy training is configured to use CUDA by default (the expanded experiment uses CPU for PPO and CUDA for DQN). A separate CUDA toolkit installation is not specified by this project.
- Internet access for downloading historical price data from Yahoo Finance with `yfinance`.
- Enough disk space for downloaded data, trained model checkpoints, TensorBoard logs, and reports. These generated files are stored under `data/` and `results/`.

The default configurations request hundreds of thousands of training steps per algorithm and can take substantial time and GPU memory. The expanded experiment trains multiple seeds. Use its `--smoke` option for a short pipeline run.

## Setup

Run commands from the repository root so the relative `data/` and `results/` paths resolve correctly.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS, create and activate the virtual environment with the platform's usual `python3 -m venv .venv` and `source .venv/bin/activate` commands. The requirements pin CUDA-enabled PyTorch wheels; CPU-only setups need a matching PyTorch installation and the configurations in `src/utils/config.py` changed to use `device="cpu"` (and explicit per-algorithm device settings where applicable).

## Run an experiment

### Five-stock baseline

Prepare adjusted-price data, daily returns, and features, then train and evaluate PPO and DQN:

```powershell
python -m src.main
python -m src.experiment
```

The baseline tickers, date range, training settings, and data paths are in [`src/utils/config.py`](src/utils/config.py). To evaluate already-trained models without training again:

```powershell
python -m src.experiment --skip-training
```

### Diversified ETF experiment

Download the ETF universe and prepare returns and features:

```powershell
python -m src.robust_experiment --prepare-only
```

Then train and evaluate the agents:

```powershell
python -m src.robust_experiment
```

Alternatively, `python -m src.robust_experiment --prepare-data` prepares the data and continues directly into training and evaluation. Add `--ppo-only` to omit DQN, or `--skip-training` to evaluate existing checkpoints.

### Expanded multi-regime experiment

Prepare data for the expanded stock and defensive-asset universe:

```powershell
python -m src.expanded_experiment --prepare-only
```

Train the agents and run the development workflow (including walk-forward baseline reports):

```powershell
python -m src.expanded_experiment
```

For a shorter run, use `python -m src.expanded_experiment --prepare-data --smoke`. The final 2022–2025 test evaluation is intentionally disabled by default; once the protocol and checkpoints are frozen, run again with `--evaluate-final` to unlock it. `--ppo-only` omits DQN, and `--skip-training` skips training and prepares the development reports from existing checkpoints.

## Notebooks

The Jupyter notebooks in `notebooks/` provide interactive analysis:

1. `01_lstm_analysis.ipynb` explores the five-stock data, trains a multi-asset LSTM, and compares its predictions and portfolio strategy with classical baselines.
2. `02_dqn_ppo_analysis.ipynb` trains or loads PPO and DQN agents and compares them with the LSTM and conventional strategies.
3. `03_expanded_multi_regime_training.ipynb` prepares the expanded universe, inspects regime coverage and baselines, trains agents, and reviews evaluation outputs.

Start Jupyter from the repository root after installing the requirements:

```powershell
python -m pip install jupyterlab
python -m jupyter lab
```

## Project layout

```text
src/                 Data, environment, agent training, evaluation, and experiment code
notebooks/           Interactive LSTM, DRL, and expanded-universe analyses
data/raw/            Downloaded adjusted prices (generated)
data/processed/      Returns, features, and regime labels (generated)
results/models/      Trained agents and fitted transformers (generated)
results/reports/     Metrics, returns, validation histories, and TensorBoard logs
results/figures/     Generated plots
```

Generated data and experiment outputs are ignored by Git. Commands create the required output directories as needed.

## Outputs and configuration

Evaluation writes out-of-sample returns and performance metrics to `results/reports/`. The expanded experiment also writes regime, walk-forward, paired-bootstrap, and transaction-cost sensitivity reports. Trained models and feature scalers are saved under `results/models/`; plots are written under `results/figures/`.

Change ticker universes, dates, split boundaries, algorithm settings, device selection, and output paths in `src/utils/config.py`. The expanded universe and sector labels are defined in `src/data/universe.py`.
