import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

# ============================================================
# 1. OU Process Simulation
# ============================================================

def simulate_ou_paths(
    n_paths: int = 3000,
    n_steps: int = 252,
    x0: float = 0.0,
    mu: float = 0.0,
    theta: float = 3.0,
    sigma: float = 1.0,
    dt: float = 1 / 252,
    seed: int = 42
) -> np.ndarray:
    """
    Exact discretization of Ornstein-Uhlenbeck process.

    dX_t = theta * (mu - X_t) dt + sigma dW_t
    """
    rng = np.random.default_rng(seed)

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = x0

    exp_term = np.exp(-theta * dt)
    variance = sigma**2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
    std_term = np.sqrt(variance)

    for t in range(n_steps):
        eps = rng.normal(size=n_paths)
        paths[:, t + 1] = (
            mu
            + (paths[:, t] - mu) * exp_term
            + std_term * eps
        )

    return paths


# ============================================================
# 2. Triple Barrier Evaluation
# ============================================================

def evaluate_triple_barrier_trade(
    path: np.ndarray,
    t0: int,
    direction: int,
    profit_take: float,
    stop_loss: float,
    vertical_barrier: int,
    transaction_cost: float = 0.0
) -> dict:
    """
    Triple barrier trade evaluation.

    direction:
        +1 = long
        -1 = short
    """
    x0 = path[t0]
    t_end = min(t0 + vertical_barrier, len(path) - 1)

    label = 0
    exit_time = t_end
    exit_reason = "vertical"

    for t in range(t0 + 1, t_end + 1):
        pnl = direction * (path[t] - x0)

        if pnl >= profit_take:
            label = 1
            exit_time = t
            exit_reason = "profit_take"
            break

        if pnl <= -stop_loss:
            label = -1
            exit_time = t
            exit_reason = "stop_loss"
            break

    gross_payoff = direction * (path[exit_time] - x0)
    net_payoff = gross_payoff - transaction_cost

    return {
        "entry_time": t0,
        "exit_time": exit_time,
        "holding_period": exit_time - t0,
        "direction": direction,
        "label": label,
        "exit_reason": exit_reason,
        "gross_payoff": gross_payoff,
        "net_payoff": net_payoff
    }


# ============================================================
# 3. Strategy Simulation
# ============================================================

def apply_triple_barrier_strategy(
    paths: np.ndarray,
    mu: float,
    long_run_std: float,
    entry_z: float,
    profit_take_z: float,
    stop_loss_z: float,
    vertical_barrier: int,
    transaction_cost: float,
    signal_cost: float,
    opportunity_z: float,
    allow_overlap: bool = False
) -> dict:
    """
    Mean-reversion entry + triple barrier exit.

    Entry rule:
        z <= -entry_z -> long
        z >=  entry_z -> short

    Recall opportunity:
        If |z| >= opportunity_z and oracle mean-reversion trade
        hits profit-taking first, it is treated as a missed opportunity.
    """
    profit_take = profit_take_z * long_run_std
    stop_loss = stop_loss_z * long_run_std

    path_payoffs = []
    path_trade_counts = []
    path_signal_counts = []
    path_drawdowns = []

    all_trades = []

    tp = 0
    fp = 0
    fn = 0

    total_signal_count = 0
    total_possible_times = 0

    for path_id, path in enumerate(paths):
        z = (path - mu) / long_run_std
        t = 0

        cumulative_payoff = 0.0
        equity_curve = [0.0]
        trade_count = 0
        signal_count = 0

        while t < len(path) - 1:
            total_possible_times += 1

            signal = False
            direction = 0

            if z[t] <= -entry_z:
                signal = True
                direction = 1
            elif z[t] >= entry_z:
                signal = True
                direction = -1

            if signal:
                signal_count += 1
                total_signal_count += 1

                trade = evaluate_triple_barrier_trade(
                    path=path,
                    t0=t,
                    direction=direction,
                    profit_take=profit_take,
                    stop_loss=stop_loss,
                    vertical_barrier=vertical_barrier,
                    transaction_cost=transaction_cost
                )

                trade_count += 1

                # signal cost is charged per signal/trade
                trade_net_payoff = trade["net_payoff"] - signal_cost
                cumulative_payoff += trade_net_payoff
                equity_curve.append(cumulative_payoff)

                trade["path_id"] = path_id
                trade["entry_z"] = entry_z
                trade["net_payoff_after_signal_cost"] = trade_net_payoff
                all_trades.append(trade)

                if trade["label"] == 1:
                    tp += 1
                else:
                    fp += 1

                if allow_overlap:
                    t += 1
                else:
                    t = trade["exit_time"] + 1

            else:
                # Oracle missed opportunity for recall
                if abs(z[t]) >= opportunity_z:
                    oracle_direction = int(-np.sign(z[t]))

                    oracle_trade = evaluate_triple_barrier_trade(
                        path=path,
                        t0=t,
                        direction=oracle_direction,
                        profit_take=profit_take,
                        stop_loss=stop_loss,
                        vertical_barrier=vertical_barrier,
                        transaction_cost=0.0
                    )

                    if oracle_trade["label"] == 1:
                        fn += 1

                t += 1

        equity_curve = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = equity_curve - running_max
        max_drawdown = drawdown.min()

        path_payoffs.append(cumulative_payoff)
        path_trade_counts.append(trade_count)
        path_signal_counts.append(signal_count)
        path_drawdowns.append(max_drawdown)

    trade_df = pd.DataFrame(all_trades)

    path_payoffs = np.array(path_payoffs)
    path_trade_counts = np.array(path_trade_counts)
    path_signal_counts = np.array(path_signal_counts)
    path_drawdowns = np.array(path_drawdowns)

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    signal_frequency = (
        total_signal_count / total_possible_times
        if total_possible_times > 0 else np.nan
    )

    avg_net_payoff = path_payoffs.mean()
    std_net_payoff = path_payoffs.std()
    loss_probability = np.mean(path_payoffs < 0)

    expected_drawdown = path_drawdowns.mean()
    expected_mdd = path_drawdowns.mean()
    worst_mdd = path_drawdowns.min()

    avg_trade_count = path_trade_counts.mean()
    avg_signal_count = path_signal_counts.mean()

    if len(trade_df) > 0:
        win_rate = (trade_df["label"] == 1).mean()
        avg_trade_payoff = trade_df["net_payoff_after_signal_cost"].mean()
        avg_holding_period = trade_df["holding_period"].mean()
        gross_payoff_sum = trade_df["gross_payoff"].sum()
        net_payoff_sum = trade_df["net_payoff_after_signal_cost"].sum()
    else:
        win_rate = np.nan
        avg_trade_payoff = np.nan
        avg_holding_period = np.nan
        gross_payoff_sum = 0.0
        net_payoff_sum = 0.0

    return {
        "entry_z": entry_z,
        "profit_take_z": profit_take_z,
        "stop_loss_z": stop_loss_z,
        "vertical_barrier": vertical_barrier,

        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,

        "avg_net_payoff": avg_net_payoff,
        "std_net_payoff": std_net_payoff,
        "loss_probability": loss_probability,
        "expected_mdd": expected_mdd,
        "worst_mdd": worst_mdd,

        "signal_frequency": signal_frequency,
        "avg_trade_count": avg_trade_count,
        "avg_signal_count": avg_signal_count,

        "win_rate": win_rate,
        "avg_trade_payoff": avg_trade_payoff,
        "avg_holding_period": avg_holding_period,
        "gross_payoff_sum": gross_payoff_sum,
        "net_payoff_sum": net_payoff_sum,

        "trade_df": trade_df,
        "path_payoffs": path_payoffs,
        "path_drawdowns": path_drawdowns
    }


# ============================================================
# 4. Metrics
# ============================================================

def fbeta_score(precision: float, recall: float, beta: float) -> float:
    if np.isnan(precision) or np.isnan(recall):
        return np.nan

    denom = beta**2 * precision + recall

    if denom == 0:
        return np.nan

    return (1 + beta**2) * precision * recall / denom


def add_implied_fbeta_columns(
    df: pd.DataFrame,
    beta_values: list[float] | np.ndarray
) -> pd.DataFrame:
    out = df.copy()

    for beta in beta_values:
        col = f"fbeta_{beta:.2f}"
        out[col] = [
            fbeta_score(p, r, beta)
            for p, r in zip(out["precision"], out["recall"])
        ]

    return out


# ============================================================
# 5. Efficient Frontier
# ============================================================

def get_efficient_frontier(
    df: pd.DataFrame,
    return_col: str = "avg_net_payoff",
    risk_col: str = "loss_probability"
) -> pd.DataFrame:
    """
    Efficient frontier:
    A rule is efficient if no other rule has both:
        higher or equal return
        lower or equal risk
    with at least one strict improvement.
    """
    data = df.copy().dropna(subset=[return_col, risk_col])
    efficient = []

    for i, row in data.iterrows():
        dominated = (
            (data[return_col] >= row[return_col]) &
            (data[risk_col] <= row[risk_col]) &
            (
                (data[return_col] > row[return_col]) |
                (data[risk_col] < row[risk_col])
            )
        ).any()

        if not dominated:
            efficient.append(i)

    frontier = data.loc[efficient].sort_values(risk_col)
    return frontier


def minmax_scale(series: pd.Series) -> pd.Series:
    s_min = series.min()
    s_max = series.max()

    if s_max == s_min:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (series - s_min) / (s_max - s_min)


def select_rule_by_mandate(
    df: pd.DataFrame,
    conservative_weight: float = 0.7,
    aggressive_weight: float = 0.3,
    return_col: str = "avg_net_payoff",
    risk_col: str = "loss_probability",
    cost_col: str = "signal_frequency",
    cost_weight: float = 0.2
) -> pd.Series:
    """
    Select optimal rule under mandate.

    conservative_weight:
        How much the mandate penalizes risk.

    aggressive_weight:
        How much the mandate rewards return.

    cost_weight:
        How much the mandate penalizes signal frequency / operational cost.

    Utility:
        aggressive_weight * scaled_return
        - conservative_weight * scaled_risk
        - cost_weight * scaled_cost
    """
    tmp = df.copy().dropna(subset=[return_col, risk_col, cost_col])

    tmp["return_score"] = minmax_scale(tmp[return_col])
    tmp["risk_score"] = minmax_scale(tmp[risk_col])
    tmp["cost_score"] = minmax_scale(tmp[cost_col])

    tmp["mandate_utility"] = (
        aggressive_weight * tmp["return_score"]
        - conservative_weight * tmp["risk_score"]
        - cost_weight * tmp["cost_score"]
    )

    best_idx = tmp["mandate_utility"].idxmax()
    return tmp.loc[best_idx]


# ============================================================
# 6. Full Grid Simulation
# ============================================================

def run_ou_triple_barrier_frontier_mc(
    n_paths: int = 3000,
    n_steps: int = 252,
    x0: float = 0.0,
    mu: float = 0.0,
    theta: float = 3.0,
    sigma: float = 1.0,
    dt: float = 1 / 252,

    entry_z_grid: np.ndarray | None = None,
    profit_take_z_grid: np.ndarray | None = None,
    stop_loss_z_grid: np.ndarray | None = None,
    vertical_barrier_grid: list[int] | None = None,

    transaction_cost: float = 0.0005,
    signal_cost: float = 0.0001,
    opportunity_z: float = 0.5,
    allow_overlap: bool = False,
    beta_values: list[float] | None = None,
    seed: int = 42
) -> dict:
    """
    Monte Carlo simulation over trading-rule grid.

    Rule grid:
        entry_z
        profit_take_z
        stop_loss_z
        vertical_barrier

    Output:
        strategy_df
        efficient frontier
        mandate-selected rules
    """
    if entry_z_grid is None:
        entry_z_grid = np.round(np.arange(0.5, 3.05, 0.25), 2)

    if profit_take_z_grid is None:
        profit_take_z_grid = np.round(np.arange(0.25, 1.55, 0.25), 2)

    if stop_loss_z_grid is None:
        stop_loss_z_grid = np.round(np.arange(0.25, 1.55, 0.25), 2)

    if vertical_barrier_grid is None:
        vertical_barrier_grid = [5, 10, 20, 40]

    if beta_values is None:
        beta_values = [0.5, 1.0, 2.0]

    paths = simulate_ou_paths(
        n_paths=n_paths,
        n_steps=n_steps,
        x0=x0,
        mu=mu,
        theta=theta,
        sigma=sigma,
        dt=dt,
        seed=seed
    )

    long_run_std = sigma / np.sqrt(2 * theta)

    rows = []
    trade_dfs = {}
    path_payoff_map = {}
    path_drawdown_map = {}

    rule_id = 0

    for entry_z in tqdm(entry_z_grid):
        for profit_take_z in profit_take_z_grid:
            for stop_loss_z in stop_loss_z_grid:
                for vertical_barrier in vertical_barrier_grid:

                    result = apply_triple_barrier_strategy(
                        paths=paths,
                        mu=mu,
                        long_run_std=long_run_std,
                        entry_z=entry_z,
                        profit_take_z=profit_take_z,
                        stop_loss_z=stop_loss_z,
                        vertical_barrier=vertical_barrier,
                        transaction_cost=transaction_cost,
                        signal_cost=signal_cost,
                        opportunity_z=opportunity_z,
                        allow_overlap=allow_overlap
                    )

                    trade_dfs[rule_id] = result.pop("trade_df")
                    path_payoff_map[rule_id] = result.pop("path_payoffs")
                    path_drawdown_map[rule_id] = result.pop("path_drawdowns")

                    result["rule_id"] = rule_id
                    rows.append(result)

                    rule_id += 1

    strategy_df = pd.DataFrame(rows)
    strategy_df = add_implied_fbeta_columns(strategy_df, beta_values)

    frontier_loss_prob = get_efficient_frontier(
        strategy_df,
        return_col="avg_net_payoff",
        risk_col="loss_probability"
    )

    frontier_mdd = get_efficient_frontier(
        strategy_df,
        return_col="avg_net_payoff",
        risk_col="expected_mdd_abs"
    ) if "expected_mdd_abs" in strategy_df.columns else None

    # expected_mdd is negative, so create positive risk value
    strategy_df["expected_mdd_abs"] = -strategy_df["expected_mdd"]
    strategy_df["worst_mdd_abs"] = -strategy_df["worst_mdd"]

    frontier_mdd = get_efficient_frontier(
        strategy_df,
        return_col="avg_net_payoff",
        risk_col="expected_mdd_abs"
    )

    return {
        "paths": paths,
        "strategy_df": strategy_df,
        "frontier_loss_probability": frontier_loss_prob,
        "frontier_mdd": frontier_mdd,
        "trade_dfs": trade_dfs,
        "path_payoff_map": path_payoff_map,
        "path_drawdown_map": path_drawdown_map,
        "long_run_std": long_run_std
    }


# ============================================================
# 7. Plotting
# ============================================================

def plot_efficient_frontier(
    strategy_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    return_col: str = "avg_net_payoff",
    risk_col: str = "loss_probability",
    color_col: str = "fbeta_1.00",
    size_col: str = "signal_frequency",
    title: str = "Mandate Efficient Frontier"
):
    """
    Scatter plot:
        x-axis = risk
        y-axis = return
        color = implied F-beta
        size = signal frequency
    """
    df = strategy_df.dropna(subset=[return_col, risk_col, color_col, size_col]).copy()

    sizes = 30 + 400 * minmax_scale(df[size_col])

    plt.figure(figsize=(12, 7))

    scatter = plt.scatter(
        df[risk_col],
        df[return_col],
        c=df[color_col],
        s=sizes,
        alpha=0.65
    )

    plt.plot(
        frontier_df[risk_col],
        frontier_df[return_col],
        linewidth=2.5,
        marker="o",
        label="Efficient frontier"
    )

    plt.colorbar(scatter, label=color_col)
    plt.xlabel(risk_col)
    plt.ylabel(return_col)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_mandate_heatmap(
    strategy_df: pd.DataFrame,
    conservative_weights: np.ndarray | None = None,
    cost_weights: np.ndarray | None = None,
    aggressive_weight_rule: str = "one_minus_conservative",
    return_col: str = "avg_net_payoff",
    risk_col: str = "loss_probability",
    cost_col: str = "signal_frequency"
) -> pd.DataFrame:
    """
    Heatmap over:
        x-axis = conservative weight
        y-axis = cost weight
        color = selected rule's utility

    aggressive weight is usually 1 - conservative weight.
    """
    if conservative_weights is None:
        conservative_weights = np.round(np.arange(0.0, 1.01, 0.05), 2)

    if cost_weights is None:
        cost_weights = np.round(np.arange(0.0, 1.01, 0.05), 2)

    heatmap_values = pd.DataFrame(
        index=cost_weights,
        columns=conservative_weights,
        dtype=float
    )

    selected_rules = []

    for cw in conservative_weights:
        if aggressive_weight_rule == "one_minus_conservative":
            aw = 1.0 - cw
        else:
            aw = 0.5

        for cost_w in cost_weights:
            best = select_rule_by_mandate(
                strategy_df,
                conservative_weight=cw,
                aggressive_weight=aw,
                return_col=return_col,
                risk_col=risk_col,
                cost_col=cost_col,
                cost_weight=cost_w
            )

            heatmap_values.loc[cost_w, cw] = best["mandate_utility"]

            selected_rules.append({
                "conservative_weight": cw,
                "aggressive_weight": aw,
                "cost_weight": cost_w,
                "selected_rule_id": best["rule_id"],
                "selected_entry_z": best["entry_z"],
                "selected_profit_take_z": best["profit_take_z"],
                "selected_stop_loss_z": best["stop_loss_z"],
                "selected_vertical_barrier": best["vertical_barrier"],
                "selected_avg_net_payoff": best["avg_net_payoff"],
                "selected_loss_probability": best["loss_probability"],
                "selected_expected_mdd_abs": best["expected_mdd_abs"],
                "selected_signal_frequency": best["signal_frequency"],
                "selected_precision": best["precision"],
                "selected_recall": best["recall"],
                "selected_fbeta_0.50": best.get("fbeta_0.50", np.nan),
                "selected_fbeta_1.00": best.get("fbeta_1.00", np.nan),
                "selected_fbeta_2.00": best.get("fbeta_2.00", np.nan),
                "selected_utility": best["mandate_utility"]
            })

    plt.figure(figsize=(12, 7))
    plt.imshow(
        heatmap_values.values,
        aspect="auto",
        origin="lower",
        extent=[
            conservative_weights.min(),
            conservative_weights.max(),
            cost_weights.min(),
            cost_weights.max()
        ]
    )
    plt.colorbar(label="Selected rule utility")
    plt.xlabel("Conservative weight")
    plt.ylabel("Cost weight")
    plt.title("Mandate Utility Heatmap")
    plt.show()

    return pd.DataFrame(selected_rules)


# ============================================================
# 8. Example Run
# ============================================================

if __name__ == "__main__":

    results = run_ou_triple_barrier_frontier_mc(
        n_paths=2000,
        n_steps=252,
        x0=0.0,
        mu=0.0,
        theta=3.0,
        sigma=1.0,
        dt=1 / 252,

        entry_z_grid=np.round(np.arange(0.5, 3.05, 0.25), 2),
        profit_take_z_grid=np.round(np.arange(0.25, 1.55, 0.25), 2),
        stop_loss_z_grid=np.round(np.arange(0.25, 1.55, 0.25), 2),
        vertical_barrier_grid=[5, 10, 20, 40],

        transaction_cost=0.0005,
        signal_cost=0.0001,
        opportunity_z=0.5,
        allow_overlap=False,
        beta_values=[0.5, 1.0, 2.0],
        seed=42
    )

    strategy_df = results["strategy_df"]
    frontier_loss = results["frontier_loss_probability"]
    frontier_mdd = results["frontier_mdd"]

    print("\n==============================")
    print("Top 10 rules by avg_net_payoff")
    print("==============================")
    print(
        strategy_df
        .sort_values("avg_net_payoff", ascending=False)
        .head(10)
        [[
            "rule_id",
            "entry_z",
            "profit_take_z",
            "stop_loss_z",
            "vertical_barrier",
            "avg_net_payoff",
            "loss_probability",
            "expected_mdd_abs",
            "signal_frequency",
            "precision",
            "recall",
            "fbeta_0.50",
            "fbeta_1.00",
            "fbeta_2.00"
        ]]
    )

    # Efficient frontier: return vs loss probability
    plot_efficient_frontier(
        strategy_df=strategy_df,
        frontier_df=frontier_loss,
        return_col="avg_net_payoff",
        risk_col="loss_probability",
        color_col="fbeta_1.00",
        size_col="signal_frequency",
        title="Efficient Frontier: Return vs Loss Probability"
    )

    # Efficient frontier: return vs expected MDD
    plot_efficient_frontier(
        strategy_df=strategy_df,
        frontier_df=frontier_mdd,
        return_col="avg_net_payoff",
        risk_col="expected_mdd_abs",
        color_col="fbeta_1.00",
        size_col="signal_frequency",
        title="Efficient Frontier: Return vs Expected MDD"
    )

    # Example mandate: 70% conservative, 30% aggressive
    selected_70_30 = select_rule_by_mandate(
        strategy_df,
        conservative_weight=0.7,
        aggressive_weight=0.3,
        return_col="avg_net_payoff",
        risk_col="loss_probability",
        cost_col="signal_frequency",
        cost_weight=0.2
    )

    print("\n==============================")
    print("Selected Rule: 70% Conservative / 30% Aggressive")
    print("==============================")
    print(
        selected_70_30[
            [
                "rule_id",
                "entry_z",
                "profit_take_z",
                "stop_loss_z",
                "vertical_barrier",
                "avg_net_payoff",
                "loss_probability",
                "expected_mdd_abs",
                "signal_frequency",
                "precision",
                "recall",
                "fbeta_0.50",
                "fbeta_1.00",
                "fbeta_2.00",
                "mandate_utility"
            ]
        ]
    )

    # Mandate heatmap
    selected_rules_by_mandate = plot_mandate_heatmap(
        strategy_df,
        conservative_weights=np.round(np.arange(0.0, 1.01, 0.05), 2),
        cost_weights=np.round(np.arange(0.0, 1.01, 0.05), 2),
        return_col="avg_net_payoff",
        risk_col="loss_probability",
        cost_col="signal_frequency"
    )

    print("\n==============================")
    print("Mandate-selected rules sample")
    print("==============================")
    print(selected_rules_by_mandate.head())