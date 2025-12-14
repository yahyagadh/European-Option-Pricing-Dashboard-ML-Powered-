from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import xgboost as xgb
from math import log, sqrt, exp
from scipy.stats import norm

# ----------------------
# 1. Initialize FastAPI
# ----------------------
app = FastAPI(
    title="European Option Pricing API",
    description="Fast pricing of European options using a GBDT surrogate model",
    version="1.0"
)

# ----------------------
# 2. Enable CORS for frontend
# ----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# 3. Load trained XGBoost model
# ----------------------
model = xgb.Booster()
model.load_model("models/gbdt_model_large.json")  # Ensure correct path

# ----------------------
# 4. Define input schema
# ----------------------
class OptionInput(BaseModel):
    S: float       # Spot price
    K: float       # Strike price
    T: float       # Time to maturity
    sigma: float   # Volatility
    r: float       # Risk-free rate
    type: int      # 1 = Call, 0 = Put

# ----------------------
# 5. Feature engineering
# ----------------------
def prepare_features(option: OptionInput):
    moneyness = option.S / option.K
    vol_sqrtT = option.sigma * np.sqrt(option.T)
    log_moneyness = np.log(moneyness + 1e-8)

    data = np.array([[ 
        option.S,
        option.K,
        option.T,
        option.sigma,
        option.r,
        option.type,
        moneyness,
        vol_sqrtT,
        log_moneyness
    ]])

    feature_names = ["S", "K", "T", "sigma", "r", "type", "moneyness", "vol_sqrtT", "log_moneyness"]
    return xgb.DMatrix(data, feature_names=feature_names)

# ----------------------
# 6. Black-Scholes formula
# ----------------------
def black_scholes_price(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return max(0.0, S-K) if option_type==1 else max(0.0, K-S)
    
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type == 1:  # Call
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:  # Put
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ----------------------
# 7. Monte Carlo Simulation
# ----------------------
def monte_carlo_price(S, K, T, r, sigma, option_type, num_simulations=100000):
    # Simulate end-of-period stock prices
    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * sqrt(T) * Z)
    
    if option_type == 1:  # Call
        payoffs = np.maximum(ST - K, 0)
    else:  # Put
        payoffs = np.maximum(K - ST, 0)
    
    mc_price = np.exp(-r * T) * np.mean(payoffs)
    return mc_price

# ----------------------
# 8. Prediction endpoint
# ----------------------
@app.post("/predict")
def predict_option_price(option: OptionInput):
    try:
        # GBDT Prediction
        dmatrix = prepare_features(option)
        gbdt_price = float(model.predict(dmatrix)[0])

        # Black-Scholes Prediction
        bs_price = black_scholes_price(option.S, option.K, option.T, option.r, option.sigma, option.type)

        # Monte Carlo Prediction
        mc_price = monte_carlo_price(option.S, option.K, option.T, option.r, option.sigma, option.type)

        return {
            "option_type": "Call" if option.type == 1 else "Put",
            "gbdt_price": round(gbdt_price, 6),
            "bs_price": round(bs_price, 6),
            "mc_price": round(mc_price, 6)
        }
    except Exception as e:
        return {"error": str(e)}

# ----------------------
# 9. Health check endpoint
# ----------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}
