#!/usr/bin/env python3
"""
USD 100% Capped ProNote 的定价示例脚本。

本脚本展示如何使用 Black–Scholes 闭式公式与蒙特卡洛模拟对一款挂钩标普 500 的“100% Capped ProNote with Participation”进行定价，并利用蒙特卡洛的微小扰动重估法计算 Delta 与 Gamma。

使用的第三方依赖仅包含 numpy 与 scipy.stats。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.stats import norm


# === 全局参数区 ===
# 初始标的指数水平
S0: float = 3873.33
# 无风险利率（年化，连续复利）
r: float = 0.03
# 年化波动率（可替换为更合理的估计值）
sigma: float = 0.20
# 产品到期时间（年）
T: float = 3.0
# 本金（票面价值）
nominal: float = 1000.0
# 到期收益封顶倍数（132%）
cap_mult: float = 1.32


@dataclass
class PriceResult:
    """用于存储蒙特卡洛估值与希腊值的结果容器。"""

    price: float
    delta: float
    gamma: float


def payoff(ST: np.ndarray | float, S0: float, cap_mult: float, nominal: float) -> np.ndarray | float:
    """计算到期支付。

    参数:
    ST: 终端价格（可为标量或 ndarray）。
    S0: 初始价格。
    cap_mult: 封顶倍数（如 1.32 表示 132%）。
    nominal: 本金。

    返回:
    到期收益 = nominal * min(ST / S0, cap_mult)。
    """

    return nominal * np.minimum(np.asarray(ST) / S0, cap_mult)


def bs_call_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black–Scholes 欧式看涨期权定价公式。"""

    if T <= 0 or sigma <= 0:
        raise ValueError("T 和 sigma 需要为正数")

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call


def price_capped_pronote_bs(
    S0: float, r: float, sigma: float, T: float, nominal: float, cap_mult: float
) -> float:
    """基于 Black–Scholes 的封顶结构定价。

    将产品拆解为：
    1) 一张到期偿付 nominal 的零息债券；
    2) 一组执行价为 S0 与 cap_mult * S0 的看涨价差，按 nominal / S0 的比例放大。
    """

    discount_bond = nominal * math.exp(-r * T)
    K1 = S0
    K2 = cap_mult * S0
    call_spread = (nominal / S0) * (bs_call_price(S0, K1, T, r, sigma) - bs_call_price(S0, K2, T, r, sigma))
    return discount_bond + call_spread


def simulate_terminal_prices(
    S0: float, r: float, sigma: float, T: float, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """在风险中性假设下模拟终端价格。"""

    z = rng.standard_normal(n_paths)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * math.sqrt(T) * z
    ST = S0 * np.exp(drift + diffusion)
    return ST


def price_capped_pronote_mc(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    nominal: float,
    cap_mult: float,
    n_paths: int,
    rng: np.random.Generator,
) -> float:
    """使用蒙特卡洛模拟定价封顶结构。"""

    ST = simulate_terminal_prices(S0, r, sigma, T, n_paths, rng)
    payoffs = payoff(ST, S0, cap_mult, nominal)
    discounted = math.exp(-r * T) * np.mean(payoffs)
    return float(discounted)


def greeks_capped_pronote_mc(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    nominal: float,
    cap_mult: float,
    n_paths: int,
    eps: float = 0.01,
) -> PriceResult:
    """通过微小扰动重估计算蒙特卡洛价格、Delta 与 Gamma。"""

    rng = np.random.default_rng(2024)
    z = rng.standard_normal(n_paths)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * math.sqrt(T) * z

    def priced_with_S(base_S: float) -> float:
        ST = base_S * np.exp(drift + diffusion)
        payoff_vals = payoff(ST, S0=base_S, cap_mult=cap_mult, nominal=nominal)
        return math.exp(-r * T) * float(np.mean(payoff_vals))

    V0 = priced_with_S(S0)
    V_plus = priced_with_S(S0 * (1 + eps))
    V_minus = priced_with_S(S0 * (1 - eps))

    delta = (V_plus - V_minus) / (2 * S0 * eps)
    gamma = (V_plus - 2 * V0 + V_minus) / (S0 * eps) ** 2

    return PriceResult(price=V0, delta=delta, gamma=gamma)


def main() -> None:
    # 计算 Black–Scholes 价格
    bs_price = price_capped_pronote_bs(S0, r, sigma, T, nominal, cap_mult)

    # 蒙特卡洛定价与希腊值
    rng = np.random.default_rng(42)
    mc_price = price_capped_pronote_mc(S0, r, sigma, T, nominal, cap_mult, n_paths=200_000, rng=rng)
    mc_result = greeks_capped_pronote_mc(
        S0, r, sigma, T, nominal, cap_mult, n_paths=200_000, eps=0.01
    )

    print("Black–Scholes 定价 (含零息债 + 看涨价差): {:.4f}".format(bs_price))
    print("蒙特卡洛定价: {:.4f}".format(mc_price))
    print("蒙特卡洛 Delta: {:.6f}".format(mc_result.delta))
    print("蒙特卡洛 Gamma: {:.6f}".format(mc_result.gamma))


if __name__ == "__main__":
    main()
