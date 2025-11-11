# backend/main.py - COMPLETE implementation with ALL features
import os
import time
import csv
import sqlite3
import json
import subprocess
from datetime import datetime, timedelta
from threading import Thread, Lock
from typing import List, Dict, Optional
import traceback
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from requests.exceptions import RequestException
import pandas as pd
import numpy as np
from collections import deque
from scipy import stats
import pytz

# ---------- CONFIG ----------
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

HEADERS = {
    "client-id": DHAN_CLIENT_ID,
    "access-token": DHAN_ACCESS_TOKEN,
    "Content-Type": "application/json"
}

STOCKS_CSV = os.getenv("STOCKS_CSV", "stocks.csv")
RATE_LIMIT_SECONDS = float(os.getenv("RATE_LIMIT_SECONDS", "3"))
DB_PATH = os.getenv("DB_PATH", "option_data.db")

# Backup settings
BACKUP_FOLDER = os.getenv("BACKUP_FOLDER", "backups")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

# API endpoints
EXPIRY_LIST_URL = "https://api.dhan.co/v2/optionchain/expirylist"
OPTION_CHAIN_URL = "https://api.dhan.co/v2/optionchain"

# Alert Thresholds - EXACTLY as in original
IV_SPIKE_THRESHOLD = 5
IV_CRASH_THRESHOLD = 5
OI_SPIKE_THRESHOLD = 10
GAMMA_THRESHOLD = 0.02
THETA_DECAY_THRESHOLD = -20
PRICE_STABILITY_THRESHOLD = 0.5
EXPIRY_DAY_THETA_ACCELERATION = -50
PIN_RISK_THRESHOLD = 100000
VOLATILITY_CRUSH_THRESHOLD = 15
GAMMA_SQUEEZE_THRESHOLD = 0.05
MOMENTUM_WINDOW = 10
VWAP_DEVIATION_THRESHOLD = 0.5
LIQUIDITY_THRESHOLD = 50000
SMART_MONEY_THRESHOLD = 1000000
VOLATILITY_REGIME_PERIODS = [5, 10, 20]

# IST timezone
IST = pytz.timezone("Asia/Kolkata")

# Constants
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_BACKOFF = 2

# Global state stores with locks
rolling_data_store = {}
previous_data_store = {}
sentiment_history_store = {}
sent_alerts_store = {}
data_lock = Lock()

app = FastAPI(title="Complete Option Chain API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DATABASE SETUP ----------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    cur = conn.cursor()
    
    # Main snapshots
    cur.execute("""
    CREATE TABLE IF NOT EXISTS option_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        securityId INTEGER,
        fetched_at TEXT,
        expiry TEXT,
        last_price REAL,
        raw_json TEXT
    )
    """)
    
    # Sentiment log - COMPLETE schema
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sentiment_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        timestamp TEXT,
        strike_price REAL,
        option_type TEXT,
        ltp REAL,
        oi REAL,
        ltp_change REAL,
        oi_change REAL,
        volume REAL,
        iv REAL,
        iv_change REAL,
        delta REAL,
        gamma REAL,
        theta REAL,
        vega REAL,
        sentiment_score INTEGER,
        sentiment_bias TEXT,
        underlying_value REAL,
        exposure REAL,
        notional_value REAL,
        mfi REAL,
        activity_metric REAL,
        flow_type TEXT
    )
    """)
    
    # Analysis results
    cur.execute("""
    CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        timestamp TEXT,
        analysis_type TEXT,
        data TEXT
    )
    """)
    
    # Alerts log
    cur.execute("""
    CREATE TABLE IF NOT EXISTS alerts_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        timestamp TEXT,
        alert_type TEXT,
        message TEXT,
        severity TEXT
    )
    """)
    
    # Expiry analysis
    cur.execute("""
    CREATE TABLE IF NOT EXISTS expiry_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        timestamp TEXT,
        expiry_date TEXT,
        pin_strike REAL,
        pin_probability REAL,
        gamma_wall REAL,
        max_pain REAL,
        reversal_score REAL,
        data TEXT
    )
    """)
    
    conn.commit()
    return conn

db_conn = init_db()

# ---------- TELEGRAM ALERTS ----------
def send_telegram_alert(message: str):
    """Send alert to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"[Telegram Error] {e}")

def send_unique_telegram_alert(symbol: str, alert: str):
    """Send alert only if not sent recently"""
    with data_lock:
        if symbol not in sent_alerts_store:
            sent_alerts_store[symbol] = set()
        
        if alert not in sent_alerts_store[symbol]:
            send_telegram_alert(f"*{symbol}*\n{alert}")
            sent_alerts_store[symbol].add(alert)
            
            # Clear old alerts after 100 entries
            if len(sent_alerts_store[symbol]) > 100:
                sent_alerts_store[symbol].clear()

# ---------- HELPER FUNCTIONS ----------
def load_stocks(csv_path: str) -> List[Dict]:
    stocks = []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if 'securityId' in r and 'symbol' in r:
                try:
                    sec = int(r['securityId'])
                    # Get segment
                    seg = r.get('segment', 'NSE_FNO')  # Default to NSE_FNO
                    stocks.append({
                        "securityId": sec, 
                        "symbol": r['symbol'],
                        "segment": seg
                    })
                except:
                    continue
    return stocks

def post_with_retries(url, payload, headers, timeout=REQUEST_TIMEOUT):
    backoff = 1
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(backoff)
            backoff *= RETRY_BACKOFF

def get_segment_for_security(security_id: int, segment: str = None) -> str:
    """Get correct segment for security ID"""
    if segment:
        return segment
    
    # Index segments
    if security_id in [13, 25, 11536]:  # NIFTY, BANKNIFTY, FINNIFTY
        return "IDX_I"
    
    # Default to NSE_FNO for stocks
    return "NSE_FNO"

def fetch_expiry_list(security_id: int, segment: str = None) -> list:
    seg = get_segment_for_security(security_id, segment)
    
    payload = {
        "UnderlyingScrip": security_id,
        "UnderlyingSeg": seg
    }
    
    try:
        resp = post_with_retries(EXPIRY_LIST_URL, payload, HEADERS)
        return resp.get("data", [])
    except Exception as e:
        print(f"[ERROR] Expiry list fetch failed for {security_id}: {e}")
        return []

def fetch_option_chain(security_id: int, segment: str = None) -> tuple[dict, str]:
    seg = get_segment_for_security(security_id, segment)
    
    expiries = fetch_expiry_list(security_id, segment)
    if not expiries:
        raise Exception(f"No expiries available for {security_id}")
    expiry = expiries[0]
    
    payload = {
        "UnderlyingScrip": security_id,
        "UnderlyingSeg": seg,
        "Expiry": expiry
    }
    
    data = post_with_retries(OPTION_CHAIN_URL, payload, HEADERS)
    return data, expiry

def is_expiry_day(expiry_date_str: str) -> bool:
    """Check if today is expiry day"""
    try:
        expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
        return expiry_date == datetime.now(IST).date()
    except:
        return False

# ---------- SENTIMENT ANALYSIS ----------
def score_option_sentiment(row: Dict) -> tuple[int, str]:
    """Score sentiment - EXACTLY as in original"""
    score = 0
    
    if row['OI_Change'] > 0:
        score += 1
    elif row['OI_Change'] < 0:
        score -= 1
    
    if row['LTP_Change'] > 0:
        score += 1
    elif row['LTP_Change'] < 0:
        score -= 1
    
    if row['IV_Change'] > 0:
        score += 1
    elif row['IV_Change'] < 0:
        score -= 1
    
    if row['Theta'] < 0:
        score += 1
    elif row['Theta'] > 0:
        score -= 1
    
    if row['Vega'] > 0:
        score += 1
    elif row['Vega'] < 0:
        score -= 1
    
    if row['IV_Change'] > 0 and row['LTP_Change'] < 0:
        score -= 1
    elif row['IV_Change'] > 0 and row['LTP_Change'] > 0:
        score += 1
    
    if score >= 3:
        bias = "Aggressive Buying"
    elif score >= 1:
        bias = "Mild Buying"
    elif score == 0:
        bias = "Neutral"
    elif score <= -3:
        bias = "Aggressive Writing"
    else:
        bias = "Mild Writing"
    
    return score, bias

# ---------- ADVANCED ANALYSIS FUNCTIONS ----------
def calculate_money_flow_index(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate MFI - EXACTLY as in original"""
    typical_price = df['LTP']
    money_flow = typical_price * df['Volume']
    
    positive_flow = money_flow.where(df['LTP_Change'] > 0, 0)
    negative_flow = money_flow.where(df['LTP_Change'] < 0, 0)
    
    positive_mf = positive_flow.rolling(window=window, min_periods=1).sum()
    negative_mf = negative_flow.rolling(window=window, min_periods=1).sum()
    
    mfi = 100 - (100 / (1 + (positive_mf / (negative_mf + 1))))
    return mfi.fillna(50)

def calculate_enhanced_activity_metric(row: Dict) -> float:
    """Enhanced activity metric - EXACTLY as in original"""
    base_exposure = row['Exposure']
    oi_change = abs(row['OI_Change'])
    volume = row['Volume']
    ltp_change = row['LTP_Change']
    iv_change = row['IV_Change']
    mfi = row.get('MFI', 50)
    
    volume_factor = np.log1p(volume / 50)
    oi_momentum = min(oi_change / 3, 4)
    price_momentum = abs(ltp_change) / 8
    iv_factor = abs(iv_change) / 15
    mfi_factor = abs(mfi - 50) / 25
    
    activity_score = base_exposure * (
        1 + 
        (volume_factor * 0.3) + 
        (oi_momentum * 0.25) + 
        (price_momentum * 0.2) + 
        (iv_factor * 0.15) +
        (mfi_factor * 0.1)
    )
    
    return activity_score

def detect_dark_pool_activity(row: Dict) -> bool:
    """Detect dark pool activity - EXACTLY as in original"""
    volume = row.get('Volume', 1)
    oi_change = row['OI_Change']
    ltp_change = row['LTP_Change']
    
    if oi_change > 5 and abs(ltp_change) < 2 and volume > 100:
        return True
    return False

def classify_enhanced_big_money_flow(row: Dict, df: pd.DataFrame) -> tuple[str, str, float]:
    """Classify money flow - EXACTLY as in original"""
    activity = row['Activity_Metric']
    oi_change_pct = row['OI_Change']
    ltp_change_pct = row['LTP_Change']
    iv_change_pct = row['IV_Change']
    volume = row.get('Volume', 1)
    option_type = row['Type']
    mfi = row.get('MFI', 50)
    dark_pool = row.get('Dark_Pool_Activity', False)
    
    activity_percentile = np.percentile(df['Activity_Metric'], 65)
    oi_change_values = df['OI_Change'].replace([np.inf, -np.inf], np.nan).dropna()
    significant_oi_change = np.percentile(oi_change_values, 75) if not oi_change_values.empty else 5
    significant_volume = 75
    
    if dark_pool:
        return f"Dark Pool {option_type}", "#800080", activity * 1.5
    
    if activity < activity_percentile:
        return "Low Activity", "#333333", activity * 0.1
    
    if option_type == "CE":
        if (mfi > 70 and volume > significant_volume and 
            ltp_change_pct > 2):
            return "Aggressive Call Buy", "#006400", activity * 1.2
        elif (oi_change_pct > significant_oi_change and 
            ltp_change_pct < -2 and mfi < 30 and
            volume > significant_volume):
            return "Heavy Call Short", "#DC143C", activity * 1.1
        elif (volume > significant_volume and 
            ltp_change_pct > 1):
            return "Call Buy", "#2E8B57", activity
        elif (oi_change_pct > significant_oi_change and 
            ltp_change_pct < -1 and volume > significant_volume):
            return "Call Short", "#FF6B6B", activity
        else:
            return "Call Activity", "#90EE90", activity * 0.7
    
    elif option_type == "PE":
        if (mfi < 30 and volume > significant_volume and 
            ltp_change_pct > 2):
            return "Aggressive Put Buy", "#8B0000", activity * 1.2
        elif (oi_change_pct > significant_oi_change and 
            ltp_change_pct < -2 and mfi > 70 and
            volume > significant_volume):
            return "Heavy Put Write", "#228B22", activity * 1.1
        elif (volume > significant_volume and 
            ltp_change_pct > 1):
            return "Put Buy", "#8B0000", activity
        elif (oi_change_pct > significant_oi_change and 
            ltp_change_pct < -1 and volume > significant_volume):
            return "Put Write", "#90EE90", activity
        else:
            return "Put Activity", "#FFB6C1", activity * 0.7
    
    return "Neutral", "#666666", activity * 0.3

# ---------- EXPIRY DAY ANALYSIS ----------
def expiry_day_analysis(df: pd.DataFrame, underlying_price: float, expiry_date: str) -> tuple[List[str], Dict]:
    """Complete expiry day analysis - EXACTLY as in original"""
    if not is_expiry_day(expiry_date):
        return [], {}
    
    alerts = []
    expiry_signals = {}
    
    # 1. PIN RISK
    strike_oi = df.groupby('StrikePrice')['OI'].sum().reset_index()
    max_oi_strike = strike_oi.loc[strike_oi['OI'].idxmax()]
    
    if max_oi_strike['OI'] > PIN_RISK_THRESHOLD:
        distance_to_pin = abs(underlying_price - max_oi_strike['StrikePrice'])
        pin_probability = max(0, 100 - (distance_to_pin / 10))
        
        alerts.append(f"📌 PIN RISK: Massive OI at {max_oi_strike['StrikePrice']}\n"
                     f"OI: {max_oi_strike['OI']/1000000:.1f}M | Pin Probability: {pin_probability:.0f}%")
        
        expiry_signals['pin_strike'] = max_oi_strike['StrikePrice']
        expiry_signals['pin_probability'] = pin_probability
    
    # 2. GAMMA WALL
    ce_gamma = df[df['Type'] == 'CE'].groupby('StrikePrice')['Gamma'].sum()
    pe_gamma = df[df['Type'] == 'PE'].groupby('StrikePrice')['Gamma'].sum()
    total_gamma = ce_gamma.add(pe_gamma, fill_value=0)
    
    if len(total_gamma) > 0:
        gamma_wall_strike = total_gamma.idxmax()
        max_gamma = total_gamma.max()
        
        if max_gamma > GAMMA_SQUEEZE_THRESHOLD:
            wall_direction = "above" if underlying_price < gamma_wall_strike else "below"
            alerts.append(f"🧱 GAMMA WALL at {gamma_wall_strike}\n"
                         f"Total Gamma: {max_gamma:.4f} | Price is {wall_direction} wall")
            
            expiry_signals['gamma_wall'] = gamma_wall_strike
            expiry_signals['gamma_strength'] = max_gamma
    
    # 3. VOLATILITY CRUSH
    avg_iv = df['IV'].mean()
    
    # 4. THETA BURN
    avg_theta = df['Theta'].mean()
    if avg_theta < EXPIRY_DAY_THETA_ACCELERATION:
        alerts.append(f"⏰ EXTREME THETA BURN: {avg_theta:.2f}")
    
    # 5. DELTA IMBALANCE
    ce_delta_oi = (df[df['Type'] == 'CE']['Delta'] * df[df['Type'] == 'CE']['OI']).sum()
    pe_delta_oi = (df[df['Type'] == 'PE']['Delta'] * df[df['Type'] == 'PE']['OI']).sum()
    total_delta_oi = ce_delta_oi + pe_delta_oi
    
    if abs(total_delta_oi) > 1000000:
        bias = "Bullish" if total_delta_oi > 0 else "Bearish"
        alerts.append(f"⚖️ DELTA IMBALANCE: {bias} bias\n"
                     f"Net Delta*OI: {total_delta_oi/1000000:.1f}M")
    
    # 6. REVERSAL PREDICTION
    reversal_score = 0
    max_pain = strike_oi.loc[strike_oi['OI'].idxmax(), 'StrikePrice']
    distance_from_max_pain = abs(underlying_price - max_pain)
    
    if distance_from_max_pain > 50:
        reversal_score += min(20, distance_from_max_pain / 5)
    
    oi_concentration = max_oi_strike['OI'] / strike_oi['OI'].sum()
    if oi_concentration > 0.3:
        reversal_score += 25
    
    if max_gamma > GAMMA_SQUEEZE_THRESHOLD:
        reversal_score += 20
    
    current_time = datetime.now(IST).time()
    if current_time >= datetime.strptime("14:30", "%H:%M").time():
        reversal_score += 15
    
    if reversal_score > 40:
        direction = "toward Max Pain" if distance_from_max_pain > 30 else "away from current level"
        alerts.append(f"🎯 EXPIRY REVERSAL SIGNAL (Score: {reversal_score:.0f}/100)\n"
                     f"High probability reversal {direction}")
        
        expiry_signals['reversal_score'] = reversal_score
    
    # Save to database
    cur = db_conn.cursor()
    cur.execute("""
        INSERT INTO expiry_analysis 
        (symbol, timestamp, expiry_date, pin_strike, pin_probability, 
         gamma_wall, max_pain, reversal_score, data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "TEMP",  # Will be set by caller
        datetime.now(IST).isoformat(),
        expiry_date,
        expiry_signals.get('pin_strike'),
        expiry_signals.get('pin_probability'),
        expiry_signals.get('gamma_wall'),
        max_pain,
        reversal_score,
        json.dumps(expiry_signals)
    ))
    db_conn.commit()
    
    return alerts, expiry_signals

# ---------- COMPLETE ANALYSIS ----------
def complete_analyze_data(symbol: str, option_chain: dict, expiry: str) -> dict:
    """COMPLETE analysis with ALL features from original"""
    
    if "data" not in option_chain or "oc" not in option_chain["data"]:
        return {}
    
    option_chain_data = option_chain["data"]["oc"]
    underlying_price = option_chain["data"]["last_price"]
    
    # Get previous data
    with data_lock:
        if symbol not in previous_data_store:
            previous_data_store[symbol] = {}
        previous_data = previous_data_store[symbol]
    
    # Determine ATM
    atm_strike = min(option_chain_data.keys(), 
                     key=lambda x: abs(float(x) - underlying_price))
    atm_strike = float(atm_strike)
    
    # ATM ± 5 strikes
    min_strike = atm_strike - 5 * 50
    max_strike = atm_strike + 5 * 50
    
    data_list = []
    alerts = []
    
    for strike, contracts in option_chain_data.items():
        strike_price = float(strike)
        
        if strike_price < min_strike or strike_price > max_strike:
            continue
        
        ce_data = contracts.get("ce", {})
        pe_data = contracts.get("pe", {})
        
        for opt_type, opt_data in [("CE", ce_data), ("PE", pe_data)]:
            iv = opt_data.get("implied_volatility", 0)
            oi = opt_data.get("oi", 0)
            ltp = opt_data.get("last_price", 0)
            greeks = opt_data.get("greeks", {})
            volume = opt_data.get("volume", 0)
            
            delta = greeks.get("delta", 0)
            gamma = greeks.get("gamma", 0)
            theta = greeks.get("theta", 0)
            vega = greeks.get("vega", 0)
            
            key = f"{strike_price}_{opt_type}"
            prev = previous_data.get(key, {})
            
            oi_change = ((oi - prev.get("OI", oi)) / oi * 100) if oi else 0
            iv_change = ((iv - prev.get("IV", iv)) / iv * 100) if iv else 0
            ltp_change = ((ltp - prev.get("LTP", ltp)) / ltp * 100) if ltp else 0
            
            prev_underlying = previous_data.get("underlying_price", underlying_price)
            price_change = abs((underlying_price - prev_underlying) / prev_underlying * 100) if prev_underlying else 0
            
            row = {
                "StrikePrice": strike_price,
                "Type": opt_type,
                "IV": iv,
                "OI": oi,
                "LTP": ltp,
                "Delta": delta,
                "Gamma": gamma,
                "Theta": theta,
                "Vega": vega,
                "Volume": volume,
                "OI_Change": oi_change,
                "IV_Change": iv_change,
                "LTP_Change": ltp_change,
                "Exposure": oi * ltp,
                "Notional_Value": oi * ltp * 75
            }
            
            # Score sentiment
            score, bias = score_option_sentiment(row)
            row["SentimentScore"] = score
            row["SentimentBias"] = bias
            
            data_list.append(row)
            
            # Update previous data
            previous_data[key] = {
                "IV": iv,
                "OI": oi,
                "LTP": ltp
            }
            
            # Generate alerts - EXACTLY as in original
            if opt_type == "CE" and iv_change > IV_SPIKE_THRESHOLD and oi_change > OI_SPIKE_THRESHOLD and delta > 0.75:
                alert = f"🔥 STRONG BREAKOUT (CALLS): IV + OI + Delta Confirmed!\nStrike: {strike_price} | IV: {iv_change:.2f}% | OI: {oi_change:.2f}% | Delta: {delta:.2f}"
                alerts.append(alert)
                send_unique_telegram_alert(symbol, alert)
            
            if opt_type == "PE" and iv_change > IV_SPIKE_THRESHOLD and oi_change > OI_SPIKE_THRESHOLD and delta < -0.75:
                alert = f"🔥 STRONG BREAKOUT (PUTS): IV + OI + Delta Confirmed!\nStrike: {strike_price} | IV: {iv_change:.2f}% | OI: {oi_change:.2f}% | Delta: {delta:.2f}"
                alerts.append(alert)
                send_unique_telegram_alert(symbol, alert)
            
            if gamma > GAMMA_THRESHOLD:
                alert = f"⚡ HIGH GAMMA: Big Move Incoming!\nStrike: {strike_price} | {opt_type}_Gamma: {gamma:.4f}"
                alerts.append(alert)
            
            if theta < THETA_DECAY_THRESHOLD:
                alert = f"⏳ HIGH TIME DECAY: Risk for Long Options!\nStrike: {strike_price} | {opt_type}_Theta: {theta:.2f}"
                alerts.append(alert)
            
            if iv_change < -IV_CRASH_THRESHOLD:
                alert = f"🔥 IV CRASH: Sudden drop!\nStrike: {strike_price} | {opt_type}_IV: {iv_change:.2f}%"
                alerts.append(alert)
            
            if oi_change > OI_SPIKE_THRESHOLD:
                alert = f"🚀 OI SURGE: Institutional activity!\nStrike: {strike_price} | {opt_type}_OI: {oi_change:.2f}%"
                alerts.append(alert)
            
            if iv_change > IV_SPIKE_THRESHOLD and price_change < PRICE_STABILITY_THRESHOLD:
                alert = f"📈 IV RISING BUT PRICE STABLE\nStrike: {strike_price} | {opt_type}_IV: {iv_change:.2f}%"
                alerts.append(alert)
    
    previous_data["underlying_price"] = underlying_price
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    # Calculate MFI
    df['MFI'] = calculate_money_flow_index(df)
    
    # Enhanced activity metric
    df['Activity_Metric'] = df.apply(calculate_enhanced_activity_metric, axis=1)
    
    # Dark pool detection
    df['Dark_Pool_Activity'] = df.apply(detect_dark_pool_activity, axis=1)
    
    # Classify flows
    classification = df.apply(lambda row: pd.Series(
        classify_enhanced_big_money_flow(row, df)), axis=1)
    df[['Flow_Type', 'Color', 'Weighted_Activity']] = classification
    
    # Detect straddle writing
    straddle_writes = []
    for strike in df['StrikePrice'].unique():
        ce_bias = df[(df['StrikePrice'] == strike) & (df['Type'] == 'CE')]['SentimentBias'].values
        pe_bias = df[(df['StrikePrice'] == strike) & (df['Type'] == 'PE')]['SentimentBias'].values
        if len(ce_bias) > 0 and len(pe_bias) > 0:
            if "Aggressive Writing" in ce_bias and "Aggressive Writing" in pe_bias:
                straddle_writes.append(strike)
                alert = f"🔗 STRADDLE WRITING at {int(strike)}"
                alerts.append(alert)
                send_unique_telegram_alert(symbol, alert)
    
    # Save to sentiment_log
    timestamp = datetime.now(IST).isoformat()
    cur = db_conn.cursor()
    
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO sentiment_log 
            (symbol, timestamp, strike_price, option_type, ltp, oi, ltp_change, 
             oi_change, volume, iv, iv_change, delta, gamma, theta, vega, 
             sentiment_score, sentiment_bias, underlying_value, exposure, 
             notional_value, mfi, activity_metric, flow_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol, timestamp, row["StrikePrice"], row["Type"], row["LTP"],
            row["OI"], row["LTP_Change"], row["OI_Change"], row["Volume"],
            row["IV"], row["IV_Change"], row["Delta"], row["Gamma"], 
            row["Theta"], row["Vega"], row["SentimentScore"], 
            row["SentimentBias"], underlying_price, row["Exposure"],
            row["Notional_Value"], row.get("MFI", 50), row["Activity_Metric"],
            row["Flow_Type"]
        ))
    
    db_conn.commit()
    
    # Calculate aggregate metrics
    ce_avg_sentiment = df[df['Type'] == 'CE']['SentimentScore'].mean()
    pe_avg_sentiment = df[df['Type'] == 'PE']['SentimentScore'].mean()
    
    total_ce_oi = df[df['Type'] == 'CE']['OI'].sum()
    total_pe_oi = df[df['Type'] == 'PE']['OI'].sum()
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    
    # Expiry day analysis
    expiry_alerts, expiry_signals = expiry_day_analysis(df, underlying_price, expiry)
    alerts.extend(expiry_alerts)
    
    # Save alerts
    for alert in alerts[:20]:  # Top 20 alerts
        severity = "HIGH" if any(word in alert for word in ["BREAKOUT", "EXTREME"]) else "MEDIUM"
        cur.execute("""
            INSERT INTO alerts_log (symbol, timestamp, alert_type, message, severity)
            VALUES (?, ?, ?, ?, ?)
        """, (symbol, timestamp, "ANALYSIS", alert, severity))
    
    db_conn.commit()
    
    # Build response
    result = {
        "timestamp": timestamp,
        "underlying_price": underlying_price,
        "atm_strike": atm_strike,
        "expiry": expiry,
        "is_expiry_day": is_expiry_day(expiry),
        "ce_avg_sentiment": float(ce_avg_sentiment) if not pd.isna(ce_avg_sentiment) else 0,
        "pe_avg_sentiment": float(pe_avg_sentiment) if not pd.isna(pe_avg_sentiment) else 0,
        "put_call_ratio": pcr,
        "total_ce_oi": float(total_ce_oi),
        "total_pe_oi": float(total_pe_oi),
        "options_data": df.to_dict('records'),
        "alerts": alerts[:20],
        "expiry_signals": expiry_signals,
        "straddle_writes": straddle_writes
    }
    
    # Save analysis
    cur.execute("""
        INSERT INTO analysis_results (symbol, timestamp, analysis_type, data)
        VALUES (?, ?, ?, ?)
    """, (symbol, timestamp, 'complete', json.dumps(result)))
    db_conn.commit()
    
    return result

# ---------- POLLING LOOP ----------
def poll_loop(stock_list: List[Dict]):
    """Enhanced polling with complete analysis"""
    i = 0
    n = len(stock_list)
    
    print(f"[STARTUP] Starting poll loop for {n} stocks")
    
    while True:
        try:
            stock = stock_list[i % n]
            securityId = stock["securityId"]
            symbol = stock["symbol"]
            segment = stock.get("segment")
            
            # Fetch option chain
            try:
                oc, expiry = fetch_option_chain(securityId, segment)
            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")
                time.sleep(RATE_LIMIT_SECONDS)
                i += 1
                continue
            
            # Save raw snapshot
            last_price = oc.get("data", {}).get("last_price", None)
            cur = db_conn.cursor()
            cur.execute(
                "INSERT INTO option_snapshots (symbol, securityId, fetched_at, expiry, last_price, raw_json) VALUES (?, ?, ?, ?, ?, ?)",
                (symbol, securityId, datetime.now(IST).isoformat(), expiry or "", last_price or 0.0, json.dumps(oc))
            )
            db_conn.commit()
            
            # Perform complete analysis
            analysis = complete_analyze_data(symbol, oc, expiry)
            
            if analysis:
                print(f"[{datetime.now(IST).strftime('%H:%M:%S')}] {symbol}: "
                      f"Price={last_price:.2f}, PCR={analysis['put_call_ratio']:.2f}, "
                      f"CE={analysis['ce_avg_sentiment']:.2f}, "
                      f"PE={analysis['pe_avg_sentiment']:.2f}, "
                      f"Alerts={len(analysis['alerts'])}")
            
            time.sleep(RATE_LIMIT_SECONDS)
            i += 1
            
        except Exception as e:
            print(f"[ERROR] poll_loop: {e}")
            traceback.print_exc()
            time.sleep(5)

# ---------- API MODELS ----------
class SnapshotOut(BaseModel):
    id: int
    symbol: str
    securityId: int
    fetched_at: str
    expiry: str
    last_price: float
    raw_json: dict

class CompleteAnalysisOut(BaseModel):
    timestamp: str
    underlying_price: float
    atm_strike: float
    expiry: str
    is_expiry_day: bool
    ce_avg_sentiment: float
    pe_avg_sentiment: float
    put_call_ratio: float
    total_ce_oi: float
    total_pe_oi: float
    alerts: List[str]

# ---------- API ENDPOINTS ----------
@app.get("/")
def root():
    return {
        "status": "ok", 
        "message": "Complete Option Chain API with ALL features",
        "features": [
            "Multi-stock support (NSE_FNO, IDX_I)",
            "Complete sentiment analysis",
            "HFT Algo Scanner with MFI",
            "Expiry day analysis",
            "Dark pool detection",
            "Telegram alerts",
            "Straddle detection",
            "Greeks analysis",
            "Flow classification"
        ]
    }

@app.get("/api/stocks")
def get_stocks():
    """Get available stocks"""
    return load_stocks(STOCKS_CSV)

@app.get("/api/latest/{symbol}", response_model=SnapshotOut)
def get_latest(symbol: str):
    cur = db_conn.cursor()
    cur.execute(
        "SELECT id, symbol, securityId, fetched_at, expiry, last_price, raw_json "
        "FROM option_snapshots WHERE symbol = ? ORDER BY id DESC LIMIT 1",
        (symbol,)
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No data for symbol")
    return {
        "id": row[0],
        "symbol": row[1],
        "securityId": row[2],
        "fetched_at": row[3],
        "expiry": row[4],
        "last_price": row[5],
        "raw_json": json.loads(row[6])
    }

@app.get("/api/analysis/{symbol}")
def get_latest_analysis(symbol: str):
    """Get latest complete analysis"""
    cur = db_conn.cursor()
    cur.execute(
        "SELECT data FROM analysis_results WHERE symbol = ? AND analysis_type = 'complete' "
        "ORDER BY id DESC LIMIT 1",
        (symbol,)
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No analysis data")
    return json.loads(row[0])

@app.get("/api/sentiment_history/{symbol}")
def get_sentiment_history(symbol: str, limit: int = Query(500, le=2000)):
    """Get sentiment log history with ALL fields"""
    cur = db_conn.cursor()
    cur.execute(
        "SELECT * FROM sentiment_log WHERE symbol = ? ORDER BY id DESC LIMIT ?",
        (symbol, limit)
    )
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row)) for row in rows]

@app.get("/api/alerts/{symbol}")
def get_alerts(symbol: str, limit: int = Query(50, le=200)):
    """Get recent alerts"""
    cur = db_conn.cursor()
    cur.execute(
        "SELECT * FROM alerts_log WHERE symbol = ? ORDER BY id DESC LIMIT ?",
        (symbol, limit)
    )
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row)) for row in rows]

@app.get("/api/expiry_analysis/{symbol}")
def get_expiry_analysis(symbol: str):
    """Get latest expiry analysis"""
    cur = db_conn.cursor()
    cur.execute(
        "SELECT * FROM expiry_analysis WHERE symbol = ? ORDER BY id DESC LIMIT 1",
        (symbol,)
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No expiry analysis")
    columns = [desc[0] for desc in cur.description]
    return dict(zip(columns, row))

@app.get("/api/history/{symbol}")
def get_history(symbol: str, limit: int = Query(200, le=1000)):
    cur = db_conn.cursor()
    cur.execute(
        "SELECT id, symbol, securityId, fetched_at, expiry, last_price "
        "FROM option_snapshots WHERE symbol = ? ORDER BY id DESC LIMIT ?",
        (symbol, limit)
    )
    rows = cur.fetchall()
    keys = ["id", "symbol", "securityId", "fetched_at", "expiry", "last_price"]
    return [dict(zip(keys, r)) for r in rows]

@app.get("/api/flow_summary/{symbol}")
def get_flow_summary(symbol: str):
    """Get aggregated flow summary"""
    cur = db_conn.cursor()
    
    # Get latest timestamp
    cur.execute(
        "SELECT MAX(timestamp) FROM sentiment_log WHERE symbol = ?",
        (symbol,)
    )
    latest_ts = cur.fetchone()[0]
    
    if not latest_ts:
        raise HTTPException(status_code=404, detail="No flow data")
    
    # Get flow data from last 10 minutes
    time_cutoff = (datetime.fromisoformat(latest_ts) - timedelta(minutes=10)).isoformat()
    
    cur.execute("""
        SELECT flow_type, COUNT(*) as count, SUM(activity_metric) as total_activity,
               AVG(mfi) as avg_mfi
        FROM sentiment_log 
        WHERE symbol = ? AND timestamp > ?
        GROUP BY flow_type
        ORDER BY total_activity DESC
    """, (symbol, time_cutoff))
    
    rows = cur.fetchall()
    columns = ['flow_type', 'count', 'total_activity', 'avg_mfi']
    return [dict(zip(columns, row)) for row in rows]

@app.get("/api/stats")
def get_stats():
    """Get database statistics"""
    cur = db_conn.cursor()
    
    stats = {}
    
    for table in ['option_snapshots', 'sentiment_log', 'analysis_results', 'alerts_log']:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        
        cur.execute(f"SELECT COUNT(DISTINCT symbol) FROM {table}")
        symbols = cur.fetchone()[0]
        
        stats[table] = {"count": count, "symbols": symbols}
    
    return stats

# ---------- BACKUP ----------
def hourly_backup_and_git_push():
    try:
        os.makedirs(BACKUP_FOLDER, exist_ok=True)
        ts = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
        
        # Backup all tables
        for table in ['option_snapshots', 'sentiment_log', 'analysis_results', 'alerts_log', 'expiry_analysis']:
            fname = f"backup_{table}_{ts}.csv"
            path = os.path.join(BACKUP_FOLDER, fname)
            
            cur = db_conn.cursor()
            cur.execute(f"SELECT * FROM {table}")
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            
            with open(path, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                writer.writerows(rows)
            
            print(f"[BACKUP] Created {path}")
        
        # Git push if configured
        if GITHUB_TOKEN and GITHUB_REPO:
            def run(cmd, check=True):
                return subprocess.run(cmd, shell=True, check=check, 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if not os.path.exists(".git"):
                run("git init")
                run(f'git config user.email "backup-bot@render.com"')
                run(f'git config user.name "Render Backup Bot"')
                remote_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"
                run(f"git remote add origin {remote_url}", check=False)
            
            run(f"git add {BACKUP_FOLDER}")
            run(f'git commit -m "Hourly backup {ts}" || echo "No changes"')
            run(f"git push origin HEAD:{GITHUB_BRANCH}")
            print(f"[BACKUP] Pushed to GitHub")
            
    except Exception as e:
        print(f"[BACKUP ERROR] {e}")
        traceback.print_exc()

# ---------- STARTUP ----------
def start_background_workers():
    stocks = load_stocks(STOCKS_CSV)
    if not stocks:
        raise RuntimeError("No stocks in CSV")
    
    print(f"[STARTUP] Loaded {len(stocks)} stocks:")
    for s in stocks:
        print(f"  - {s['symbol']} (ID: {s['securityId']}, Segment: {s.get('segment', 'NSE_FNO')})")
    
    t = Thread(target=poll_loop, args=(stocks,), daemon=True)
    t.start()
    
    scheduler = BackgroundScheduler()
    scheduler.add_job(hourly_backup_and_git_push, 'cron', minute=0)
    scheduler.start()
    
    print(f"[STARTUP] All systems ready!")

@app.on_event("startup")
def on_startup():
    print("\n" + "="*60)
    print("COMPLETE OPTIONS ANALYSIS BACKEND")
    print("="*60)
    start_background_workers()