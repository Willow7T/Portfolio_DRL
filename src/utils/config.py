"""Central experiment configuration."""

from dataclasses import dataclass
from pathlib import Path

from ..data.universe import EXPANDED_SECTORS, EXPANDED_TICKERS

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"] # Only using five tickers for testing purposes. In practice, we will use a larger set of tickers.
START_DATE = "2018-01-01"
END_DATE = "2026-01-01"
WINDOW_SIZE = 30
RANDOM_SEED = 42

# Liquid, long-history asset-class proxies. A fixed ETF universe avoids the
# severe survivorship bias of selecting today's winning stocks retrospectively.
ROBUST_TICKERS = [
    "SPY",  # US large-cap equities
    "QQQ",  # US growth/technology equities
    "IWM",  # US small-cap equities
    "EFA",  # developed markets ex-US
    "EEM",  # emerging-market equities
    "VNQ",  # real estate
    "TLT",  # long-duration US Treasuries
    "IEF",  # intermediate US Treasuries
    "GLD",  # gold
    "DBC",  # broad commodities
]
ROBUST_START_DATE = "2007-01-01"


@dataclass(frozen=True)
class ExperimentConfig:
    """Settings shared by training, evaluation, and backtesting."""

    returns_path: Path = Path("data/processed/returns.csv")
    model_directory: Path = Path("results/models")
    report_directory: Path = Path("results/reports")
    figure_directory: Path = Path("results/figures")
    train_end: str = "2022-12-31"
    validation_end: str = "2023-12-31"
    lookback_window: int = WINDOW_SIZE
    initial_balance: float = 1_000_000.0
    transaction_cost: float = 0.001
    drawdown_limit: float = 0.20
    drawdown_penalty: float = 2.0
    sharpe_reward_weight: float = 0.05
    slippage_cost: float = 0.0
    turnover_penalty: float = 0.0
    max_weight: float = 1.0
    # Deployment-candidate training profile. Each algorithm is trained once per
    # seed; the strongest validation checkpoint becomes the canonical model.
    ppo_timesteps: int = 500_000
    dqn_timesteps: int = 500_000
    training_seeds: tuple[int, ...] = (42, 73, 101)
    validation_frequency: int = 10_000
    validation_patience: int = 15
    minimum_training_timesteps: int = 150_000
    selection_start_timestep: int = 0
    selection_metric: str = "sharpe_ratio"
    validation_aggregation: str = "pooled"
    validation_dispersion_penalty: float = 0.25
    validation_drawdown_penalty: float = 0.50
    validation_turnover_penalty: float = 0.10
    validation_worst_period_weight: float = 0.0
    policy_hidden_layers: tuple[int, ...] = (64, 64)
    policy_weight_decay: float = 0.0
    rebalance_frequency: int = 1
    sector_labels: tuple[str, ...] = ()
    max_sector_weight: float = 1.0
    training_episode_length: int | None = None
    balance_regime_sampling: bool = False
    use_shared_asset_extractor: bool = False
    per_asset_feature_count: int = 0
    global_feature_count: int = 0
    asset_embedding_dim: int = 32
    context_embedding_dim: int = 64
    action_temperature: float = 1.0
    action_smoothing: float = 1.0
    minimum_rebalance_turnover: float = 0.0
    maximum_rebalance_turnover: float = 2.0
    continuous_action_mode: str = "scores"

    ppo_learning_rate: float = 3e-4
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 256
    ppo_n_epochs: int = 10
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_range: float = 0.20
    ppo_ent_coef: float = 0.005

    dqn_learning_rate: float = 1e-4
    dqn_buffer_size: int = 200_000
    dqn_learning_starts: int = 20_000
    dqn_batch_size: int = 256
    dqn_gamma: float = 0.99
    dqn_train_frequency: int = 4
    dqn_gradient_steps: int = 1
    dqn_target_update_interval: int = 10_000
    dqn_exploration_fraction: float = 0.25
    dqn_exploration_final_epsilon: float = 0.05
    dqn_optimize_memory_usage: bool = False

    # Force Stable-Baselines3 policy/value-network updates onto the CUDA GPU.
    # Environment stepping remains on CPU.
    device: str = "cuda"
    ppo_device: str | None = None
    dqn_device: str | None = None
    seed: int = RANDOM_SEED


DEFAULT_CONFIG = ExperimentConfig()


ROBUST_CONFIG = ExperimentConfig(
    returns_path=Path("data/processed/robust_returns.csv"),
    model_directory=Path("results/models/robust"),
    report_directory=Path("results/reports/robust"),
    figure_directory=Path("results/figures/robust"),
    train_end="2018-12-31",
    validation_end="2021-12-31",
    # Rolling features already encode horizons up to one year; a shorter raw
    # sequence limits parameter count and reduces memorisation risk.
    lookback_window=20,
    transaction_cost=0.0010,
    slippage_cost=0.0005,
    turnover_penalty=0.02,
    max_weight=0.35,
    drawdown_limit=0.20,
    drawdown_penalty=3.0,
    sharpe_reward_weight=0.02,
    ppo_timesteps=750_000,
    dqn_timesteps=750_000,
    training_seeds=(42, 73, 101), # Seeds
    validation_frequency=15_000,
    validation_patience=12,
    minimum_training_timesteps=250_000,
    selection_metric="robust_score",
    validation_aggregation="yearly_robust",
    validation_dispersion_penalty=0.25,
    validation_drawdown_penalty=0.50,
    validation_turnover_penalty=0.10,
    policy_hidden_layers=(256, 256),
    policy_weight_decay=1e-5,
    ppo_batch_size=512,
    ppo_ent_coef=0.01,
    dqn_buffer_size=500_000,
    dqn_learning_starts=50_000,
    dqn_batch_size=512,
    dqn_exploration_fraction=0.35,
    device="cuda",
)


EXPANDED_CONFIG = ExperimentConfig(
    returns_path=Path("data/processed/expanded_returns.csv"),
    model_directory=Path("results/models/expanded_v2"),
    report_directory=Path("results/reports/expanded_v2"),
    figure_directory=Path("results/figures/expanded_v2"),
    train_end="2018-12-31",
    validation_end="2021-12-31",
    lookback_window=20,
    transaction_cost=0.0010,
    slippage_cost=0.0005,
    turnover_penalty=0.10,
    max_weight=0.10,
    sector_labels=tuple(EXPANDED_SECTORS[ticker] for ticker in EXPANDED_TICKERS),
    max_sector_weight=0.25,
    rebalance_frequency=21,
    drawdown_limit=0.20,
    drawdown_penalty=3.0,
    sharpe_reward_weight=0.0,
    ppo_timesteps=750_000,
    dqn_timesteps=750_000,
    training_seeds=(42, 73, 101, 131, 173),
    validation_frequency=15_000,
    validation_patience=12,
    minimum_training_timesteps=250_000,
    selection_start_timestep=100_000,
    selection_metric="robust_score",
    validation_aggregation="yearly_robust",
    validation_dispersion_penalty=0.25,
    validation_drawdown_penalty=0.50,
    validation_turnover_penalty=0.50,
    validation_worst_period_weight=0.35,
    policy_hidden_layers=(128, 128),
    policy_weight_decay=1e-5,
    ppo_learning_rate=1e-4,
    ppo_batch_size=256,
    ppo_n_epochs=5,
    ppo_clip_range=0.15,
    ppo_ent_coef=0.001,
    dqn_buffer_size=50_000,
    dqn_learning_starts=10_000,
    dqn_optimize_memory_usage=True,
    training_episode_length=252,
    balance_regime_sampling=True,
    use_shared_asset_extractor=True,
    per_asset_feature_count=9,
    # Six existing market features, seven causal regime statistics, four
    # threshold states, and four train-fitted mixture probabilities.
    global_feature_count=21,
    asset_embedding_dim=32,
    context_embedding_dim=64,
    action_temperature=1.0,
    action_smoothing=0.25,
    minimum_rebalance_turnover=0.05,
    maximum_rebalance_turnover=0.25,
    continuous_action_mode="softmax",
    device="cuda",
    ppo_device="cpu",
    dqn_device="cuda",
)
