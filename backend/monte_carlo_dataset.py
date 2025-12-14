# monte_carlo_dataset_large.py
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from math import log, sqrt, exp
from scipy.stats import norm

# ------- Configurable settings -------
N_SAMPLES = 1_000_000       # total rows
PATHS_PER_SAMPLE = 500       # Monte Carlo paths per option (adjust for speed/memory)
BATCH_SIZE = 2000            # rows processed per batch
SEED = 42
ANTITHETIC = True            # use antithetic variates
OUTPUT_CSV = "options_mc_dataset_large.csv"
# -------------------------------------

rng = np.random.default_rng(SEED)

# ----------------------
# 1. Sample random parameters
# ----------------------
def sample_parameters(n):
    S = rng.uniform(10, 200, size=n)             # spot price
    rel_k = rng.uniform(0.5, 1.5, size=n)
    K = S * rel_k                                # strike relative to S
    sigma = rng.uniform(0.05, 0.6, size=n)       # volatility
    r = rng.uniform(0.0, 0.05, size=n)           # risk-free rate
    T = rng.uniform(0.01, 2.0, size=n)           # time to maturity
    opt_type = rng.integers(0, 2, size=n)        # 1=call, 0=put
    return pd.DataFrame({"S": S, "K": K, "sigma": sigma, "r": r, "T": T, "type": opt_type})

# ----------------------
# 2. Monte Carlo pricing
# ----------------------
def mc_price_for_params(S_arr, K_arr, sigma_arr, r_arr, T_arr, opt_type_arr, paths_per_sample, antithetic=True):
    batch_n = S_arr.shape[0]
    M = paths_per_sample
    if antithetic:
        if M % 2 != 0:
            raise ValueError("paths_per_sample must be even for antithetic variates")
        half = M // 2
        Z = rng.standard_normal(size=(half, batch_n))
        Z_full = np.vstack([Z, -Z])  # shape (M, batch_n)
    else:
        Z_full = rng.standard_normal(size=(M, batch_n))

    drift = (r_arr - 0.5 * sigma_arr**2) * T_arr
    vol_sqrtT = sigma_arr * np.sqrt(T_arr)

    exponent = drift[np.newaxis, :] + vol_sqrtT[np.newaxis, :] * Z_full
    S_T = S_arr[np.newaxis, :] * np.exp(exponent)

    calls = np.maximum(S_T - K_arr[np.newaxis, :], 0.0)
    puts  = np.maximum(K_arr[np.newaxis, :] - S_T, 0.0)
    payoffs = np.where(opt_type_arr[np.newaxis, :] == 1, calls, puts)

    avg_payoff = payoffs.mean(axis=0)
    prices = np.exp(-r_arr * T_arr) * avg_payoff
    return prices

# ----------------------
# 3. Black-Scholes formula
# ----------------------
def black_scholes_price(S, K, r, sigma, T, option_type):
    eps = 1e-12
    T = np.maximum(T, eps)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(option_type==1, call, put)

# ----------------------
# 4. Incremental batch processing and saving
# ----------------------
# Remove existing file if exists
if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)

total_batches = (N_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in tqdm(range(total_batches), desc="Generating MC dataset"):
    start_row = batch_idx * BATCH_SIZE
    end_row = min(start_row + BATCH_SIZE, N_SAMPLES)
    n_rows = end_row - start_row

    # Sample parameters
    df_batch = sample_parameters(n_rows)

    # Compute Monte Carlo prices
    batch_prices = mc_price_for_params(
        S_arr=df_batch["S"].values,
        K_arr=df_batch["K"].values,
        sigma_arr=df_batch["sigma"].values,
        r_arr=df_batch["r"].values,
        T_arr=df_batch["T"].values,
        opt_type_arr=df_batch["type"].values,
        paths_per_sample=PATHS_PER_SAMPLE,
        antithetic=ANTITHETIC
    )
    df_batch["mc_price"] = batch_prices

    # Compute Black-Scholes prices
    df_batch["bs_price"] = black_scholes_price(
        df_batch["S"].values, df_batch["K"].values,
        df_batch["r"].values, df_batch["sigma"].values,
        df_batch["T"].values, df_batch["type"].values
    )

    # Append to CSV
    df_batch.to_csv(OUTPUT_CSV, mode='a', header=(batch_idx==0), index=False)

print(f"Saved dataset to {OUTPUT_CSV}")
