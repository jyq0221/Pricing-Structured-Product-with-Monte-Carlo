# 定价示例：USD 100% Capped ProNote with Participation

本仓库提供一个独立的 Python 脚本 `pronote_pricing.py`，用 Black–Scholes 闭式公式与蒙特卡洛模拟对标普 500 相关的“USD 100% Capped ProNote with Participation”进行定价，并计算 Delta 与 Gamma。所有注释均为中文，方便快速理解代码含义。

## 如何运行

1. 确保已安装依赖：
   ```bash
   pip install numpy scipy
   ```
2. 直接运行脚本即可得到 Black–Scholes 价格、蒙特卡洛价格以及对应的 Delta 和 Gamma：
   ```bash
   python pronote_pricing.py
   ```

## 核心参数

脚本顶部定义了可调参数：
- `S0`：初始标普 500 水平，示例为 3873.33。
- `r`：无风险利率（连续复利），示例为 3%。
- `sigma`：年化波动率占位值，默认 20%。
- `T`：到期年限，默认 3 年。
- `nominal`：产品本金，默认 1000 美元。
- `cap_mult`：封顶倍数，默认 1.32（132%）。

## 功能说明

- **payoff 计算**：到期支付为 `nominal * min(ST / S0, cap_mult)`。
- **Black–Scholes 定价**：将产品拆解为零息债券与 S0、`cap_mult * S0` 之间的看涨价差。
- **蒙特卡洛定价**：在风险中性假设下模拟 GBM 终端价格并折现平均收益。
- **希腊值计算**：通过固定随机种子的微小扰动重估得到 Delta 与 Gamma，降低方差。

运行后，终端会输出两套价格与对应的敏感度，便于交叉验证定价结果。
