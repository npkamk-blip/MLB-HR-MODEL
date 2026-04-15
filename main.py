from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import pandas as pd
import io
import math
import os
import threading
import asyncio
import uvicorn
from datetime import date, timedelta, datetime
from collections import defaultdict

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MLB_API = "https://statsapi.mlb.com/api/v1"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = "npkamk-blip/MLB-HR-MODEL"
GITHUB_API = "https://api.github.com"

# ── League Constants (update each April) ──
LEAGUE_CONSTANTS = {
    "lg_barrel_pct":   8.0,    # league avg barrel%
    "lg_hr9":          1.10,   # league avg HR/9
    "lg_hard_hit":     38.0,   # league avg hard hit%
    "lg_bullpen_hr9":  1.20,   # league avg bullpen HR/9
    "lg_hr_per_pa":    0.028,  # league avg HR/PA
    "lg_era":          4.20,   # league avg ERA
    "max_hr_per_pa":   0.12,   # ceiling on any batter base rate
    "hr_prob_cap":     28.0,   # hard cap on model output %
}

# ── Model Weights (learned from ML, updated every 45 days) ──
# All start at 1.0 — neutral. Recalibration moves them based on what predicted HRs.
# Exponent weights: effective_mult = raw_mult ** weight
# weight > 1.0 = stat matters MORE than assumed
# weight < 1.0 = stat matters LESS than assumed
DEFAULT_WEIGHTS = {
    # Batter contact quality
    "barrel_season_w":    1.0,
    "barrel_l8d_w":       1.0,
    "la_season_w":        1.0,
    "la_l8d_w":           1.0,
    "ev_season_w":        1.0,
    "ev_l8d_w":           1.0,
    "iso_season_w":       1.0,
    "iso_vs_hand_w":      1.0,
    "hard_hit_season_w":  1.0,
    "hard_hit_l8d_w":     1.0,
    # Pitcher vulnerability
    "pit_hr9_season_w":   1.0,
    "pit_hr9_vs_hand_w":  1.0,
    "pit_slg_season_w":   1.0,
    "pit_slg_vs_hand_w":  1.0,
    # Context
    "park_w":             1.0,
    "weather_w":          1.0,
    "bullpen_w":          1.0,
    "bat_platoon_w":      1.0,
    "pit_platoon_w":      1.0,
    # Pitch matchup
    "pitch_delta_w":      1.0,
    # K% penalty
    "k_pct_w":            1.0,
    # Active stats list (which 8 are in the model)
    "active_stats": [
        "barrel_season", "la_season", "pit_hr9_vs_hand",
        "iso_vs_hand", "park", "weather", "pitch_delta", "bat_platoon"
    ],
    # Metadata
    "last_calibrated":    None,
    "records_used":       0,
    "calibration_round":  0,
    "promoted_stats":     [],
    "dropped_stats":      [],
    "recent_changes":     [],
}
_model_weights = DEFAULT_WEIGHTS.copy()

def W(key):
    """Get a model weight, default 1.0"""
    return float(_model_weights.get(key, 1.0))

async def load_model_weights():
    """Load learned weights from GitHub on startup"""
    global _model_weights
    content, _ = await github_get_file("data/model_weights.json")
    if content:
        try:
            import json
            w = json.loads(content)
            _model_weights = {**DEFAULT_WEIGHTS, **w}
            print(f"Loaded model weights — round {w.get('calibration_round',0)}, {w.get('records_used',0)} records")
        except Exception as e:
            print(f"Weight load error: {e}")
    else:
        print("No model weights found — using defaults (1.0)")

async def save_model_weights(weights_dict, changes=None):
    """Save updated weights to GitHub"""
    import json
    existing, sha = await github_get_file("data/model_weights.json")
    content = json.dumps(weights_dict, indent=2)
    msg = f"weights: round {weights_dict.get('calibration_round',0)}, {weights_dict.get('records_used',0)} records"
    await github_put_file("data/model_weights.json", content, msg, sha)

async def save_model_log(weights_dict):
    """Save daily model log snapshot to GitHub"""
    import json
    today = date.today().isoformat()
    path = f"data/model_log/{today}.json"
    log = {
        "date": today,
        "rotation_round": get_rotation_round(),
        "rotation_day": get_rotation_day(),
        "weights": {k: v for k, v in weights_dict.items() if k.endswith("_w")},
        "active_stats": weights_dict.get("active_stats", DEFAULT_WEIGHTS["active_stats"]),
        "last_calibrated": weights_dict.get("last_calibrated"),
        "records_used": weights_dict.get("records_used", 0),
        "promoted_stats": weights_dict.get("promoted_stats", []),
        "dropped_stats": weights_dict.get("dropped_stats", []),
        "recent_changes": weights_dict.get("recent_changes", []),
    }
    existing, sha = await github_get_file(path)
    await github_put_file(path, json.dumps(log, indent=2), f"model log: {today}", sha)

async def recalibrate_model():
    """
    Run logistic regression on all history records.
    Update the 22 weights. Auto-promote candidates with correlation > 0.15.
    Enforce 8 stat hard cap — weakest gets replaced by strong candidate.
    Save updated weights + model log to GitHub.
    """
    global _model_weights
    import json, math
    # Pull all history files
    all_records = []
    try:
        keys_data, _ = await github_get_file("data/predictions")
        # list files via GitHub API
        if not GITHUB_TOKEN: return {"error": "No GitHub token"}
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(
                f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/data/predictions",
                headers={"Authorization": f"token {GITHUB_TOKEN}"}
            )
            files = r.json() if r.is_success else []
        for f in files:
            if not f.get("name","").endswith(".json"): continue
            content, _ = await github_get_file(f"data/predictions/{f['name']}")
            if content:
                try:
                    recs = json.loads(content)
                    all_records.extend(recs)
                except: pass
    except Exception as e:
        print(f"Recalibration data load error: {e}")
        return {"error": str(e)}

    # Filter to completed non-DNP records
    completed = [r for r in all_records if r.get("hit_hr") in [0, 1]]
    n = len(completed)
    if n < 50:
        return {"error": f"Not enough data — need 50+ records, have {n}"}

    print(f"Recalibrating on {n} completed records")

    # ── Point-biserial correlation for each stat ──
    def correlate(records, key):
        vals = [(r[key], r["hit_hr"]) for r in records if r.get(key) is not None and r.get(key) != 0]
        if len(vals) < 20: return None
        xs = [v[0] for v in vals]
        ys = [v[1] for v in vals]
        n2 = len(xs)
        mx = sum(xs)/n2; my = sum(ys)/n2
        num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        dx = math.sqrt(sum((x-mx)**2 for x in xs))
        dy = math.sqrt(sum((y-my)**2 for y in ys))
        if dx*dy == 0: return None
        return round(num/(dx*dy), 4)

    # All stats to evaluate
    all_stats = [
        # Active model stats
        "barrel_season_w",   "barrel_l8d_w",
        "la_season_w",       "la_l8d_w",
        "ev_season_w",       "ev_l8d_w",
        "iso_season_w",      "iso_vs_hand_w",
        "hard_hit_season_w", "hard_hit_l8d_w",
        "pit_hr9_season_w",  "pit_hr9_vs_hand_w",
        "pit_slg_season_w",  "pit_slg_vs_hand_w",
        "park_w",            "weather_w",
        "bullpen_w",         "bat_platoon_w",
        "pit_platoon_w",     "pitch_delta_w",
        "k_pct_w",
        # Round 2 candidates
        "fb_pct_season",     "pull_pct_season",
        "pit_fb_pct_allowed","hard_hit_l8d",
        "k_pct_l8d",
        # Round 3 candidates
        "pit_era_diff",
        # Round 4 candidates
        "pit_k9_season",
    ]

    # Map weight keys to stored field names
    stat_field_map = {
        "barrel_season_w": "barrel_pct_season",
        "barrel_l8d_w": "barrel_pct_l8d",
        "la_season_w": "la_season",
        "la_l8d_w": "la_l8d",
        "ev_season_w": "ev_season",
        "ev_l8d_w": "ev_l8d",
        "iso_season_w": "iso_season",
        "iso_vs_hand_w": "iso_vs_hand",
        "hard_hit_season_w": "hard_hit_season",
        "hard_hit_l8d_w": "hard_hit_l8d",
        "pit_hr9_season_w": "pit_hr9_season",
        "pit_hr9_vs_hand_w": "pit_hr9_vs_hand",
        "pit_slg_season_w": "pit_slg_season",
        "pit_slg_vs_hand_w": "pit_slg_vs_hand",
        "park_w": "park_factor",
        "weather_w": "weather_mult",
        "bullpen_w": "bullpen_vuln",
        "bat_platoon_w": "bat_platoon_mult",
        "pit_platoon_w": "pit_platoon_mult",
        "pitch_delta_w": "combined_pitch_delta",
        "k_pct_w": "k_pct_season",
        "fb_pct_season": "fb_pct_season",
        "pull_pct_season": "pull_pct_season",
        "pit_fb_pct_allowed": "pit_fb_pct_allowed",
        "k_pct_l8d": "k_pct_l8d",
        "pit_era_diff": "pit_era_diff",
        "pit_hr_fb_pct": "pit_hr_fb_pct",
        "pit_k9_season": "pit_k9_season",
    }

    correlations = {}
    for stat_key in all_stats:
        field = stat_field_map.get(stat_key, stat_key)
        corr = correlate(completed, field)
        if corr is not None:
            correlations[stat_key] = corr

    # ── Convert correlations to weights ──
    # New weight = 1.0 + (correlation * 2.0) clamped to 0.3 - 2.5
    new_weights = _model_weights.copy()
    changes = []
    for stat_key, corr in correlations.items():
        if not stat_key.endswith("_w"): continue
        old_w = W(stat_key)
        new_w = round(max(0.3, min(2.5, 1.0 + corr * 2.0)), 3)
        new_weights[stat_key] = new_w
        if abs(new_w - old_w) >= 0.05:
            direction = "up" if new_w > old_w else "down"
            changes.append(f"{stat_key}: {old_w:.2f} -> {new_w:.2f} ({direction})")

    # ── Determine active 8 stats ──
    # Rank all stats by correlation, keep top 8, enforce cap
    ranked = sorted(
        [(k, v) for k, v in correlations.items() if k.endswith("_w")],
        key=lambda x: abs(x[1]), reverse=True
    )
    new_active = [k.replace("_w","") for k,v in ranked[:8]]

    # ── Auto-promote candidates with corr > 0.15 ──
    promoted = []
    dropped = []
    candidates = {k: v for k, v in correlations.items() if not k.endswith("_w") and abs(v) >= 0.15}
    current_active = _model_weights.get("active_stats", DEFAULT_WEIGHTS["active_stats"])

    for cand_key, cand_corr in sorted(candidates.items(), key=lambda x: abs(x[1]), reverse=True):
        if len(new_active) >= 8:
            # Find weakest active stat to displace
            weakest = min(
                [(k, correlations.get(k+"_w", 0)) for k in new_active],
                key=lambda x: abs(x[1])
            )
            if abs(cand_corr) > abs(weakest[1]):
                dropped.append(weakest[0])
                new_active.remove(weakest[0])
                new_active.append(cand_key)
                promoted.append(cand_key)
                # Add weight for newly promoted stat
                new_weights[cand_key + "_w"] = round(max(0.3, min(2.5, 1.0 + cand_corr * 2.0)), 3)
        else:
            new_active.append(cand_key)
            promoted.append(cand_key)

    # ── Save updated weights ──
    new_weights["active_stats"] = new_active
    new_weights["last_calibrated"] = date.today().isoformat()
    new_weights["records_used"] = n
    new_weights["calibration_round"] = get_rotation_round()
    new_weights["promoted_stats"] = promoted
    new_weights["dropped_stats"] = dropped
    new_weights["recent_changes"] = changes[:20]

    _model_weights = new_weights

    await save_model_weights(new_weights)
    await save_model_log(new_weights)

    print(f"Recalibration complete — {len(changes)} weight changes, {len(promoted)} promotions, {len(dropped)} drops")
    return {
        "records_used": n,
        "weight_changes": len(changes),
        "promoted": promoted,
        "dropped": dropped,
        "top_predictors": [k for k,v in ranked[:5]],
        "changes": changes[:10],
    }


# Every 45 days we rotate candidate stats in/out to find what actually predicts HRs
# Round start date: April 13, 2026 (first day of data collection)
ROTATION_START = date(2026, 4, 13)
ROTATION_DAYS  = 45

def get_rotation_round():
    """Return current rotation round number (1-based)"""
    delta = (date.today() - ROTATION_START).days
    return max(1, delta // ROTATION_DAYS + 1)

def get_rotation_day():
    """Return days into current rotation round"""
    delta = (date.today() - ROTATION_START).days
    return delta % ROTATION_DAYS

# Stats being evaluated per round
ROTATION_SCHEDULE = {
    1: {
        "active": ["barrel_pct_season","barrel_pct_l8d","la_season","la_l8d",
                   "ev_season","iso_vs_hand","iso_season","l8d_hr",
                   "pit_hr9_vs_hand","pit_hr9_season","pit_slg_vs_hand",
                   "pit_hard_hit_season","pitch_matchup_score",
                   "bullpen_vuln","bat_platoon_mult","pit_platoon_mult",
                   "park_factor","weather_mult","hot_cold_mult","k_pct_season",
                   "avg_pa_per_game"],
        "candidates": [],  # nothing new yet, establishing baseline
        "note": "Baseline round — establishing correlations for all core stats"
    },
    2: {
        "active": [],  # filled after round 1 correlation analysis
        "candidates": ["fb_pct_season","pull_pct_season","pit_fb_pct_allowed",
                       "hard_hit_l8d","k_pct_l8d"],
        "note": "Testing flyball%, pull%, pitcher flyball% allowed"
    },
    3: {
        "active": [],
        "candidates": ["hard_hit_l8d", "k_pct_l8d", "pit_hr_fb_pct"],
        "note": "Testing Hard Hit% L8D, K% L8D, Pitcher HR/FB%"
    },
    4: {
        "active": [],
        "candidates": ["batter_vs_pit_career_ab","batter_vs_pit_career_hr",
                       "pit_days_rest","game_time_hour"],
        "note": "Testing H2H career history, days rest, game time"
    },
}


SAVANT_BASE = "https://baseballsavant.mlb.com"

def savant_batter_url(year=None, min_pa=10, extra=""):
    yr = year or current_season()
    return (f"{SAVANT_BASE}/leaderboard/custom?year={yr}&type=batter&filter=&sort=4"
            f"&sortDir=desc&min={min_pa}&selections=pa,ab,hit,home_run,strikeout,"
            f"k_percent,slg_percent,batting_avg,barrel_batted_rate,exit_velocity_avg,"
            f"launch_angle_avg,hard_hit_percent,pull_percent,n_fb_percent{extra}&csv=true")

def savant_pitcher_url(year=None, min_pa=5, extra=""):
    yr = year or current_season()
    return (f"{SAVANT_BASE}/leaderboard/custom?year={yr}&type=pitcher&filter=&sort=4"
            f"&sortDir=desc&min={min_pa}&selections=pa,home_run,barrel_batted_rate,"
            f"exit_velocity_avg,hard_hit_percent,k_percent,p_era,n_fb_percent{extra}&csv=true")

def savant_pitch_arsenal_url(ptype="pitcher", year=None, min_pa=1):
    yr = year or current_season()
    return (f"{SAVANT_BASE}/leaderboard/pitch-arsenal-stats?type={ptype}"
            f"&pitchType=&year={yr}&team=&min={min_pa}&csv=true")

def current_season():
    """Return current MLB season year"""
    today = date.today()
    # MLB season runs roughly March-November
    return today.year if today.month >= 3 else today.year - 1

def savant_8d_url():
    cutoff = (date.today() - timedelta(days=8)).isoformat()
    yr = current_season()
    return (f"{SAVANT_BASE}/leaderboard/custom?year={yr}&type=batter&filter=&sort=4"
            f"&sortDir=desc&min=3&selections=pa,ab,home_run,barrel_batted_rate,"
            f"exit_velocity_avg,launch_angle_avg,hard_hit_percent,pull_percent,"
            f"slg_percent,batting_avg,strikeout,k_percent&csv=true"
            f"&game_date_gt={cutoff}")

_cache = {
    "bat_2026":     pd.DataFrame(),
    "bat_2025":     pd.DataFrame(),
    "bat_8d":       pd.DataFrame(),
    "bat_l5g":      {},
    "bat_l8d_hr":   {},
    "bat_games":    {},
    "bat_vs_lhp":   pd.DataFrame(),
    "bat_vs_rhp":   pd.DataFrame(),
    "pit_2026":     pd.DataFrame(),
    "pit_2025":     pd.DataFrame(),
    "pit_vs_lhh":   pd.DataFrame(),
    "pit_vs_rhh":   pd.DataFrame(),
    "pit_arsenal":  pd.DataFrame(),
    "bat_arsenal":  pd.DataFrame(),
    "team_hitting":  {},
    "team_pitching": {},
    "team_bullpen":  {},
    "player_hands": {},
    "player_ip":    {},
    "ready":        False,
    "last_updated": None,
    "last_8d_update": None,
}

_games_cache = {}   # { date_str: { "data": ..., "ts": datetime } }
GAMES_CACHE_TTL = 900  # 15 minutes in seconds

PARK_HR_FACTORS = {
    "Colorado Rockies":      {"L":1.40,"R":1.40},
    "Cincinnati Reds":       {"L":1.30,"R":1.25},
    "Baltimore Orioles":     {"L":1.22,"R":1.18},
    "New York Yankees":      {"L":1.25,"R":1.12},
    "Philadelphia Phillies": {"L":1.18,"R":1.12},
    "Boston Red Sox":        {"L":1.08,"R":1.18},
    "Chicago Cubs":          {"L":1.12,"R":1.10},
    "Atlanta Braves":        {"L":1.10,"R":1.12},
    "Texas Rangers":         {"L":1.10,"R":1.08},
    "Milwaukee Brewers":     {"L":1.08,"R":1.08},
    "Arizona Diamondbacks":  {"L":1.08,"R":1.08},
    "Toronto Blue Jays":     {"L":1.05,"R":1.07},
    "Houston Astros":        {"L":1.04,"R":1.02},
    "Los Angeles Dodgers":   {"L":1.02,"R":1.04},
    "Minnesota Twins":       {"L":1.02,"R":1.02},
    "Kansas City Royals":    {"L":1.05,"R":1.05},
    "Chicago White Sox":     {"L":1.02,"R":1.02},
    "Cleveland Guardians":   {"L":0.98,"R":0.97},
    "Detroit Tigers":        {"L":0.97,"R":0.95},
    "St. Louis Cardinals":   {"L":0.97,"R":0.98},
    "Washington Nationals":  {"L":0.97,"R":0.96},
    "Pittsburgh Pirates":    {"L":0.96,"R":0.98},
    "New York Mets":         {"L":0.94,"R":0.96},
    "Los Angeles Angels":    {"L":0.96,"R":0.93},
    "Tampa Bay Rays":        {"L":0.94,"R":0.94},
    "Seattle Mariners":      {"L":0.91,"R":0.93},
    "Miami Marlins":         {"L":0.90,"R":0.92},
    "San Francisco Giants":  {"L":0.88,"R":0.86},
    "San Diego Padres":      {"L":0.87,"R":0.89},
    "Oakland Athletics":     {"L":0.88,"R":0.88},
}

STADIUMS = {
    "Arizona Diamondbacks":  {"lat":33.4453,"lon":-112.0667,"dome":True},
    "Atlanta Braves":        {"lat":33.8907,"lon":-84.4677,"dome":False,"hr_bearing":225,"open_factor":0.5},
    "Baltimore Orioles":     {"lat":39.2838,"lon":-76.6217,"dome":False,"hr_bearing":180,"open_factor":0.6},
    "Boston Red Sox":        {"lat":42.3467,"lon":-71.0972,"dome":False,"hr_bearing":270,"open_factor":0.7},
    "Chicago Cubs":          {"lat":41.9484,"lon":-87.6553,"dome":False,"hr_bearing":225,"open_factor":1.0},
    "Chicago White Sox":     {"lat":41.8299,"lon":-87.6338,"dome":False,"hr_bearing":315,"open_factor":0.5},
    "Cincinnati Reds":       {"lat":39.0979,"lon":-84.5082,"dome":False,"hr_bearing":270,"open_factor":0.6},
    "Cleveland Guardians":   {"lat":41.4954,"lon":-81.6854,"dome":False,"hr_bearing":225,"open_factor":0.6},
    "Colorado Rockies":      {"lat":39.7559,"lon":-104.9942,"dome":False,"hr_bearing":270,"open_factor":0.7},
    "Detroit Tigers":        {"lat":42.3390,"lon":-83.0485,"dome":False,"hr_bearing":135,"open_factor":0.5},
    "Houston Astros":        {"lat":29.7573,"lon":-95.3555,"dome":True},
    "Kansas City Royals":    {"lat":39.0517,"lon":-94.4803,"dome":False,"hr_bearing":180,"open_factor":0.6},
    "Los Angeles Angels":    {"lat":33.8003,"lon":-117.8827,"dome":False,"hr_bearing":270,"open_factor":0.5},
    "Los Angeles Dodgers":   {"lat":34.0739,"lon":-118.2400,"dome":False,"hr_bearing":315,"open_factor":0.5},
    "Miami Marlins":         {"lat":25.7781,"lon":-80.2197,"dome":True},
    "Milwaukee Brewers":     {"lat":43.0282,"lon":-87.9712,"dome":True},
    "Minnesota Twins":       {"lat":44.9817,"lon":-93.2778,"dome":False,"hr_bearing":225,"open_factor":0.6},
    "New York Mets":         {"lat":40.7571,"lon":-73.8458,"dome":False,"hr_bearing":270,"open_factor":0.5},
    "New York Yankees":      {"lat":40.8296,"lon":-73.9262,"dome":False,"hr_bearing":270,"open_factor":0.6},
    "Oakland Athletics":     {"lat":38.5726,"lon":-121.5088,"dome":False,"hr_bearing":270,"open_factor":0.5},
    "Philadelphia Phillies": {"lat":39.9056,"lon":-75.1665,"dome":False,"hr_bearing":225,"open_factor":0.5},
    "Pittsburgh Pirates":    {"lat":40.4469,"lon":-80.0057,"dome":False,"hr_bearing":270,"open_factor":0.6},
    "San Diego Padres":      {"lat":32.7076,"lon":-117.1570,"dome":False,"hr_bearing":270,"open_factor":0.8},
    "San Francisco Giants":  {"lat":37.7786,"lon":-122.3893,"dome":False,"hr_bearing":315,"open_factor":0.9},
    "Seattle Mariners":      {"lat":47.5914,"lon":-122.3325,"dome":True},
    "St. Louis Cardinals":   {"lat":38.6226,"lon":-90.1928,"dome":False,"hr_bearing":225,"open_factor":0.5},
    "Tampa Bay Rays":        {"lat":27.7683,"lon":-82.6534,"dome":True},
    "Texas Rangers":         {"lat":32.7473,"lon":-97.0825,"dome":True},
    "Toronto Blue Jays":     {"lat":43.6414,"lon":-79.3894,"dome":True},
    "Washington Nationals":  {"lat":38.8730,"lon":-77.0074,"dome":False,"hr_bearing":180,"open_factor":0.5},
}

PITCH_TYPE_MAP = {
    "FF": "wfa", "FA": "wfa",  # 4-seam fastball
    "SI": "wsi",               # sinker
    "SL": "wsl",               # slider
    "ST": "wsl",               # sweeper (treat as slider)
    "FC": "wfc",               # cutter
    "CH": "wch",               # changeup
    "CU": "wcu",               # curveball
    "KC": "wcu",               # knuckle curve
    "FS": "wfs",               # splitter
    "FO": "wfs",               # forkball
}

PITCH_DISPLAY = {
    "wfa":"Fastball","wsi":"Sinker","wsl":"Slider",
    "wfc":"Cutter","wch":"Changeup","wcu":"Curveball","wfs":"Splitter"
}

# ── Data Loading ──
async def fetch_savant_csv(url: str, session: httpx.AsyncClient) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        r = await session.get(url, headers=headers, timeout=60, follow_redirects=True)
        if not r.is_success:
            print(f"Savant fetch failed {r.status_code}: {url[:80]}")
            return pd.DataFrame()
        text = r.text.strip()
        if not text or text.startswith('<'):
            print(f"Savant returned HTML (blocked?): {url[:80]}")
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(text))
        return df
    except Exception as e:
        print(f"Savant fetch error: {e} — {url[:80]}")
        return pd.DataFrame()

def parse_player_name(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'last_name, first_name' column to 'name' column"""
    name_col = None
    for col in df.columns:
        if 'last_name' in col.lower() or 'first_name' in col.lower():
            name_col = col
            break
    if name_col and name_col in df.columns:
        df['name'] = df[name_col].apply(lambda x: reverse_name(str(x)) if pd.notna(x) else "")
    return df

def reverse_name(s: str) -> str:
    """Convert 'Last, First' to 'First Last'"""
    parts = s.split(', ', 1)
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}"
    return s

def calc_batter_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived stats from raw Savant columns"""
    df = parse_player_name(df)
    if 'slg_percent' in df.columns and 'batting_avg' in df.columns:
        df['iso'] = pd.to_numeric(df['slg_percent'], errors='coerce') - pd.to_numeric(df['batting_avg'], errors='coerce')
    # HR/FB: home_run / (ab * n_fb_percent/100)
    if 'home_run' in df.columns and 'ab' in df.columns and 'n_fb_percent' in df.columns:
        def calc_hrfb(row):
            ab = float(row.get('ab') or 0)
            fb_pct = float(row.get('n_fb_percent') or 0)
            hr = float(row.get('home_run') or 0)
            fb = ab * fb_pct / 100.0
            return (hr / fb * 100) if fb > 0 else 0
        df['hr_fb_pct'] = df.apply(calc_hrfb, axis=1)
    # HR rate per 600 PA
    if 'home_run' in df.columns and 'pa' in df.columns:
        df['hr_rate'] = df.apply(lambda r: (float(r.get('home_run') or 0) / max(float(r.get('pa') or 1), 1)) * 600, axis=1)
    # Rename columns to match model
    rename = {
        'barrel_batted_rate': 'barrel_pct',
        'exit_velocity_avg': 'exit_velo',
        'launch_angle_avg': 'launch_angle',
        'hard_hit_percent': 'hard_hit_pct',
        'pull_percent': 'pull_pct',
        'n_fb_percent': 'fb_pct',
        'k_percent': 'k_pct',
        'home_run': 'hr',
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df

def calc_pitcher_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived pitcher stats"""
    df = parse_player_name(df)
    rename = {
        'barrel_batted_rate': 'barrel_pct_allowed',
        'exit_velocity_avg': 'exit_velo_allowed',
        'hard_hit_percent': 'hard_hit_pct',
        'n_fb_percent': 'fb_pct',
        'k_percent': 'k_pct',
        'p_era': 'era',
        'home_run': 'hr',
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df

async def fetch_pitcher_ip(season=2026):
    """Fetch pitcher IP/HR9/ERA from MLB Stats API — /stats endpoint first to capture all pitchers"""
    try:
        ip_map = {}
        print("Fetching pitcher stats from MLB Stats API /stats endpoint...")
        url = f"{MLB_API}/stats?stats=season&group=pitching&gameType=R&season={season}&playerPool=All&limit=2000"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url)
            data = r.json()
        for stat_group in data.get("stats", []):
            for split in stat_group.get("splits", []):
                person = split.get("player", {})
                name = person.get("fullName", "")
                stat = split.get("stat", {})
                ip_str = stat.get("inningsPitched", "0") or "0"
                try: ip = float(ip_str)
                except: ip = 0
                gs_val = int(stat.get("gamesStarted", 0) or 0)
                hr9_str = stat.get("homeRunsPer9", "0") or "0"
                try: hr9 = float(hr9_str)
                except: hr9 = 0
                era_str = stat.get("era", "0") or "0"
                try: era = float(era_str)
                except: era = 0
                k9_str = stat.get("strikeoutsPer9Inn", "0") or "0"
                try: k9 = float(k9_str) if k9_str not in ("-.--","") else 0
                except: k9 = 0
                avg_ip = round(ip / gs_val, 1) if gs_val > 0 else 5.0
                if name:
                    ip_map[name.lower()] = {"ip": ip, "hr9": hr9, "era": era, "k9": k9, "gs": gs_val, "avg_ip": avg_ip, "name": name}
        print(f"Fetched IP data for {len(ip_map)} pitchers from MLB Stats API")
        # Supplemental leaders fallback if stats endpoint is sparse
        if len(ip_map) < 5:
            print("Stats endpoint sparse, supplementing with leaders endpoint...")
            url2 = (f"{MLB_API}/stats/leaders?leaderCategories=inningsPitched"
                    f"&season={season}&sportId=1&limit=500&statGroup=pitching&gameType=R")
            async with httpx.AsyncClient(timeout=20) as client:
                r2 = await client.get(url2)
                data2 = r2.json()
            for cat in data2.get("leagueLeaders", []):
                for leader in cat.get("leaders", []):
                    person = leader.get("person", {})
                    name = person.get("fullName", "")
                    ip = float(leader.get("value", 0) or 0)
                    nl = name.lower()
                    if name and nl not in ip_map:
                        ip_map[nl] = {"ip": ip, "hr9": 0, "era": 0, "name": name}
        return ip_map
    except Exception as e:
        print(f"MLB Stats IP fetch error: {e}")
        import traceback; traceback.print_exc()
        return {}

async def fetch_last5_games_batting():
    """Fetch last 5 games batting stats from MLB Stats API for all active hitters"""
    try:
        url = (f"{MLB_API}/stats?stats=lastXGames&lastXGames=5&group=hitting&gameType=R"
               f"&season={current_season()}&playerPool=All&limit=2000")
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url)
            data = r.json()
        l5g_map = {}
        for stat_group in data.get("stats", []):
            for split in stat_group.get("splits", []):
                person = split.get("player", {})
                name = person.get("fullName", "")
                stat = split.get("stat", {})
                if not name: continue
                try:
                    ab = int(stat.get("atBats", 0) or 0)
                    hr = int(stat.get("homeRuns", 0) or 0)
                    slg_str = stat.get("slg", "0") or "0"
                    slg = float(slg_str) if slg_str not in (".---", "", None) else 0.0
                    avg_str = stat.get("avg", "0") or "0"
                    avg = float(avg_str) if avg_str not in (".---", "", None) else 0.0
                    pa = int(stat.get("plateAppearances", 0) or 0)
                    so = int(stat.get("strikeOuts", 0) or 0)
                    iso = round(slg - avg, 3) if slg > 0 else 0.0
                    l5g_map[name.lower()] = {
                        "name": name, "ab": ab, "pa": pa,
                        "hr": hr, "slg": slg, "avg": avg, "iso": iso,
                        "k_pct": round(so / max(pa, 1) * 100, 1) if pa > 0 else 0.0,
                    }
                except Exception: continue
        print(f"Fetched last-5-games stats for {len(l5g_map)} batters")
        return l5g_map
    except Exception as e:
        print(f"Last 5 games fetch error: {e}")
        import traceback; traceback.print_exc()
        return {}

async def fetch_last8d_hr():
    """Fetch last 8 games HR count from MLB Stats API — reliable source for L8D HR count.
    Baseball Savant 8d CSV returns season HR totals for some players, so we use this instead."""
    try:
        url = (f"{MLB_API}/stats?stats=lastXGames&lastXGames=8&group=hitting&gameType=R"
               f"&season={current_season()}&playerPool=All&limit=2000")
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url)
            data = r.json()
        l8d_hr_map = {}
        for stat_group in data.get("stats", []):
            for split in stat_group.get("splits", []):
                person = split.get("player", {})
                name = person.get("fullName", "")
                stat = split.get("stat", {})
                if not name: continue
                try:
                    hr = int(stat.get("homeRuns", 0) or 0)
                    pa = int(stat.get("plateAppearances", 0) or 0)
                    l8d_hr_map[name.lower()] = {"hr": hr, "pa": pa, "name": name}
                except Exception: continue
        print(f"Fetched last-8-games HR data for {len(l8d_hr_map)} batters")
        return l8d_hr_map
    except Exception as e:
        print(f"Last 8 games HR fetch error: {e}")
        return {}

async def fetch_batter_games():
    """Fetch season games played + PA per batter from MLB Stats API for avg PA/game calculation"""
    try:
        url = (f"{MLB_API}/stats?stats=season&group=hitting&gameType=R"
               f"&season={current_season()}&playerPool=All&limit=2000")
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url)
            data = r.json()
        games_map = {}
        for stat_group in data.get("stats", []):
            for split in stat_group.get("splits", []):
                person = split.get("player", {})
                name = person.get("fullName", "")
                stat = split.get("stat", {})
                if not name: continue
                try:
                    games = int(stat.get("gamesPlayed", 0) or 0)
                    pa = int(stat.get("plateAppearances", 0) or 0)
                    ab = int(stat.get("atBats", 0) or 0)
                    if games > 0:
                        games_map[name.lower()] = {
                            "games": games,
                            "pa": pa,
                            "ab": ab,
                            "avg_pa_per_game": round(pa / games, 2),
                            "avg_ab_per_game": round(ab / games, 2),
                            "name": name,
                        }
                except Exception: continue
        print(f"Fetched games played data for {len(games_map)} batters")
        return games_map
    except Exception as e:
        print(f"Batter games fetch error: {e}")
        return {}

async def fetch_splits_mlb(season=2026):
    """Fetch batter and pitcher splits by handedness from MLB Stats API statSplits"""
    results = {
        "bat_vs_lhp": [], "bat_vs_rhp": [],
        "pit_vs_lhh": [], "pit_vs_rhh": [],
    }
    try:
        configs = [
            ("hitting",  "vl", "bat_vs_lhp"),
            ("hitting",  "vr", "bat_vs_rhp"),
            ("pitching", "vl", "pit_vs_lhh"),
            ("pitching", "vr", "pit_vs_rhh"),
        ]
        for group, sit_code, cache_key in configs:
            url = (f"{MLB_API}/stats?stats=statSplits&group={group}&gameType=R"
                   f"&season={season}&sportId=1&playerPool=ALL&limit=2000&sitCodes={sit_code}")
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(url)
                data = r.json()
            for stat_group in data.get("stats", []):
                for split in stat_group.get("splits", []):
                    person = split.get("player", {})
                    name = person.get("fullName", "")
                    stat = split.get("stat", {})
                    if not name: continue
                    try:
                        pa  = int(stat.get("battersFaced", 0) or stat.get("plateAppearances", 0) or 0)
                        hr  = int(stat.get("homeRuns", 0) or 0)
                        so  = int(stat.get("strikeOuts", 0) or 0)
                        ab  = int(stat.get("atBats", 0) or 0)
                        tb_str = stat.get("totalBases", "0") or "0"
                        try: tb = int(tb_str)
                        except: tb = 0
                        slg_str = stat.get("slg", ".000") or ".000"
                        try: slg = float(slg_str) if slg_str not in (".---","") else 0.0
                        except: slg = round(tb / max(ab, 1), 3) if ab > 0 else 0.0
                        avg_str = stat.get("avg", ".000") or ".000"
                        try: avg = float(avg_str) if avg_str not in (".---","") else 0.0
                        except: avg = 0.0
                        iso  = round(slg - avg, 3) if slg > 0 else 0.0
                        obp_str = stat.get("obp", ".000") or ".000"
                        try: obp = float(obp_str) if obp_str not in (".---","") else 0.0
                        except: obp = 0.0
                        k_pct = round(so / max(pa, 1) * 100, 1) if pa > 0 else 0.0
                        ip_str = stat.get("inningsPitched", "0") or "0"
                        try: ip = float(ip_str)
                        except: ip = pa / 4.0
                        # Use pre-calculated HR/9 if available
                        hr9_str = stat.get("homeRunsPer9", "0") or "0"
                        try: hr9 = float(hr9_str) if hr9_str not in ("-.--","") else 0.0
                        except: hr9 = round((hr / max(ip, 0.1)) * 9, 2) if ip > 0 else 0.0
                        results[cache_key].append({
                            "name":              name.strip(),
                            "pa": pa, "ab": ab, "hr": hr,
                            "slg": slg, "iso": iso, "avg": avg,
                            "woba": obp,  # OBP as wOBA proxy
                            "k_pct": k_pct,
                            "hr9": hr9, "ip": round(ip, 1),
                            "hard_hit_pct":      0,
                            "barrel_pct_allowed": 0,
                            "barrel_pct":        0,
                        })
                    except Exception:
                        continue
            print(f"{cache_key}: {len(results[cache_key])} rows (MLB statSplits sitCode={sit_code})")
    except Exception as e:
        print(f"MLB statSplits error: {e}")
        import traceback; traceback.print_exc()
    return results

async def fetch_team_stats(season=2026):
    """Fetch team hitting and pitching stats from MLB Stats API"""
    team_hitting = {}
    team_pitching = {}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Team hitting
            r = await client.get(f"{MLB_API}/teams/stats?stats=season&group=hitting&gameType=R&season={season}&sportId=1")
            data = r.json()
            for rec in data.get("stats", [{}])[0].get("splits", []):
                t = rec.get("team", {})
                s = rec.get("stat", {})
                name = t.get("name", "")
                if not name: continue
                pa = int(s.get("plateAppearances", 0) or 0)
                g  = int(s.get("gamesPlayed", 1) or 1)
                team_hitting[name] = {
                    "runs_per_g":  round(float(s.get("runs", 0) or 0) / max(g, 1), 2),
                    "hr_per_g":    round(float(s.get("homeRuns", 0) or 0) / max(g, 1), 2),
                    "avg":         float(s.get("avg", ".000").replace(".---", "0") or 0),
                    "obp":         float(s.get("obp", ".000").replace(".---", "0") or 0),
                    "slg":         float(s.get("slg", ".000").replace(".---", "0") or 0),
                    "k_pct":       round(float(s.get("strikeOuts", 0) or 0) / max(pa, 1) * 100, 1),
                    "games":       g,
                }
            # Team pitching
            r = await client.get(f"{MLB_API}/teams/stats?stats=season&group=pitching&gameType=R&season={season}&sportId=1")
            data = r.json()
            for rec in data.get("stats", [{}])[0].get("splits", []):
                t = rec.get("team", {})
                s = rec.get("stat", {})
                name = t.get("name", "")
                if not name: continue
                g = int(s.get("gamesPlayed", 1) or 1)
                ip_str = s.get("inningsPitched", "0") or "0"
                try: ip = float(ip_str)
                except: ip = 0
                team_pitching[name] = {
                    "era":         float(s.get("era", "4.50").replace("-.--", "4.50") or 4.50),
                    "whip":        float(s.get("whip", "1.30").replace("-.--", "1.30") or 1.30),
                    "hr_per_g":    round(float(s.get("homeRuns", 0) or 0) / max(g, 1), 2),
                    "hr9":         round(float(s.get("homeRuns", 0) or 0) / max(ip, 1) * 9, 2) if ip > 0 else 1.1,
                    "k_per_9":     float(s.get("strikeoutsPer9Inn", "8.0").replace("-.--", "8.0") or 8.0),
                    "runs_per_g":  round(float(s.get("runs", 0) or 0) / max(g, 1), 2),
                    "games":       g,
                }
        _cache["team_hitting"]  = team_hitting
        _cache["team_pitching"] = team_pitching
        print(f"team_hitting: {len(team_hitting)} teams, team_pitching: {len(team_pitching)} teams")

        # Fetch reliever-only stats — try pitcherTypes=RP, fallback to team pitching
        try:
            bullpen_stats = {}
            r_bp = await client.get(f"{MLB_API}/teams/stats?stats=season&group=pitching&gameType=R&season={season}&sportId=1&pitcherTypes=RP")
            if r_bp.is_success:
                bp_data = r_bp.json()
                for rec in bp_data.get("stats", [{}])[0].get("splits", []):
                    t = rec.get("team", {})
                    s = rec.get("stat", {})
                    name = t.get("name", "")
                    if not name: continue
                    ip_str = s.get("inningsPitched", "0") or "0"
                    try: ip = float(ip_str)
                    except: ip = 0
                    bullpen_stats[name] = {
                        "era":  float(s.get("era", "4.50").replace("-.--", "4.50") or 4.50),
                        "hr9":  round(float(s.get("homeRuns", 0) or 0) / max(ip, 1) * 9, 2) if ip > 0 else 1.2,
                        "whip": float(s.get("whip", "1.30").replace("-.--", "1.30") or 1.30),
                    }
            if bullpen_stats:
                _cache["team_bullpen"] = bullpen_stats
                print(f"team_bullpen: {len(bullpen_stats)} teams via RP filter")
            else:
                # Fallback — use team pitching as proxy for bullpen
                _cache["team_bullpen"] = {k: {"era": v.get("era", 4.50), "hr9": v.get("hr9", 1.2), "whip": 1.30}
                                           for k, v in team_pitching.items()}
                print(f"team_bullpen: using team pitching as fallback ({len(team_pitching)} teams)")
        except Exception as e:
            print(f"Bullpen stats error: {e}")
            _cache["team_bullpen"] = {k: {"era": v.get("era", 4.50), "hr9": v.get("hr9", 1.2), "whip": 1.30}
                                       for k, v in team_pitching.items()}
    except Exception as e:
        print(f"Team stats error: {e}")
        import traceback; traceback.print_exc()

async def load_all_savant_data():
    """Fetch all data from Baseball Savant + FanGraphs via pybaseball"""
    print("Loading data from Baseball Savant...")

    # Start pybaseball load in background thread (non-blocking)
    async with httpx.AsyncClient(timeout=60) as client:
        # Batter 2026
        df = await fetch_savant_csv(savant_batter_url(min_pa=10), client)
        if not df.empty:
            _cache["bat_2026"] = calc_batter_stats(df)
            print(f"bat_2026: {len(_cache['bat_2026'])} rows")

        # Batter 2025
        df = await fetch_savant_csv(savant_batter_url(year=current_season()-1, min_pa=50), client)
        if not df.empty:
            _cache["bat_2025"] = calc_batter_stats(df)
            print(f"bat_2025: {len(_cache['bat_2025'])} rows")

        # Batter 8d
        df = await fetch_savant_csv(savant_8d_url(), client)
        if not df.empty:
            _cache["bat_8d"] = calc_batter_stats(df)
            print(f"bat_8d: {len(_cache['bat_8d'])} rows")

        # Pitcher 2026
        df = await fetch_savant_csv(savant_pitcher_url(min_pa=5), client)
        if not df.empty:
            _cache["pit_2026"] = calc_pitcher_stats(df)
            print(f"pit_2026: {len(_cache['pit_2026'])} rows")

        # Pitcher 2025
        df = await fetch_savant_csv(savant_pitcher_url(year=current_season()-1, min_pa=50), client)
        if not df.empty:
            _cache["pit_2025"] = calc_pitcher_stats(df)
            print(f"pit_2025: {len(_cache['pit_2025'])} rows")

        # Pitch arsenal - pitcher
        await asyncio.sleep(3)
        df = await fetch_savant_csv(savant_pitch_arsenal_url("pitcher", year=current_season(), min_pa=1), client)
        if not df.empty:
            _cache["pit_arsenal"] = parse_player_name(df)
            print(f"pit_arsenal: {len(_cache['pit_arsenal'])} rows")
        else:
            print("pit_arsenal: 0 rows")

        # Pitch arsenal - batter
        await asyncio.sleep(3)
        df = await fetch_savant_csv(savant_pitch_arsenal_url("batter", year=current_season(), min_pa=1), client)
        if not df.empty:
            _cache["bat_arsenal"] = parse_player_name(df)
            print(f"bat_arsenal: {len(_cache['bat_arsenal'])} rows")
        else:
            print("bat_arsenal: 0 rows")

    # Fetch all handedness splits via MLB Stats API
    splits = await fetch_splits_mlb(current_season())
    for key, rows in splits.items():
        if rows:
            df_split = pd.DataFrame(rows)
            _cache[key] = df_split
            print(f"{key}: {len(df_split)} rows")

    # Pitcher IP from MLB Stats API
    ip_data = await fetch_pitcher_ip(current_season())
    _cache["player_ip"] = ip_data

    # Last 5 games batting from MLB Stats API
    l5g_data = await fetch_last5_games_batting()
    _cache["bat_l5g"] = l5g_data

    l8d_hr_data = await fetch_last8d_hr()
    _cache["bat_l8d_hr"] = l8d_hr_data

    games_data = await fetch_batter_games()
    _cache["bat_games"] = games_data

    _cache["last_updated"] = datetime.now().isoformat()
    _cache["ready"] = True
    print("All data loaded successfully!")

    # Team stats (non-blocking, runs after main data)
    await fetch_team_stats(current_season())

async def refresh_8d():
    """Refresh 8-day Savant data and last-5-games MLB API data"""
    async with httpx.AsyncClient(timeout=30) as client:
        df = await fetch_savant_csv(savant_8d_url(), client)
        if not df.empty:
            _cache["bat_8d"] = calc_batter_stats(df)
            print(f"bat_8d refreshed: {len(df)} rows")
    l5g_data = await fetch_last5_games_batting()
    if l5g_data:
        _cache["bat_l5g"] = l5g_data
        print(f"bat_l5g refreshed: {len(l5g_data)} players")
    l8d_hr_data = await fetch_last8d_hr()
    if l8d_hr_data:
        _cache["bat_l8d_hr"] = l8d_hr_data
        print(f"bat_l8d_hr refreshed: {len(l8d_hr_data)} players")
    _cache["last_8d_update"] = datetime.now().isoformat()

async def daily_refresh_loop():
    """Run in background — refresh all data daily at 9am, 8d every 2 hours"""
    while True:
        now = datetime.now()
        await asyncio.sleep(7200)
        try:
            await refresh_8d()
        except Exception as e:
            print(f"8d refresh error: {e}")
        if now.hour == 7:
            try:
                await load_all_savant_data()
            except Exception as e:
                print(f"Daily refresh error: {e}")
        # Save predictions at 1pm ET (18:00 UTC) — lineups mostly confirmed
        if now.hour == 11:
            try:
                await save_daily_predictions()
            except Exception as e:
                print(f"Prediction save error: {e}")
        # Record results at 2am ET — all games finished
        if now.hour == 2:
            try:
                yesterday = (date.today() - timedelta(days=1)).isoformat()
                await record_results(yesterday)
            except Exception as e:
                print(f"Result recording error: {e}")
        # Auto-recalibrate on Day 45 of each rotation round at 3am ET
        if now.hour == 3 and get_rotation_day() == 0 and get_rotation_round() > 1:
            try:
                print(f"Auto-recalibrating — Round {get_rotation_round()} Day 0")
                await recalibrate_model()
            except Exception as e:
                print(f"Auto-recalibrate error: {e}")
        # Save daily model log at 4am ET
        if now.hour == 4:
            try:
                await save_model_log(_model_weights)
            except Exception as e:
                print(f"Model log error: {e}")

# ── GitHub Storage ──
async def github_get_file(path: str):
    """Get a file from GitHub repo, returns (content_str, sha) or (None, None)"""
    if not GITHUB_TOKEN: return None, None
    try:
        url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{path}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, headers=headers)
        if r.status_code == 404: return None, None
        if not r.is_success: return None, None
        data = r.json()
        import base64
        content = base64.b64decode(data["content"]).decode("utf-8")
        return content, data["sha"]
    except Exception as e:
        print(f"GitHub get error: {e}")
        return None, None

async def github_put_file(path: str, content: str, message: str, sha: str = None):
    """Create or update a file in GitHub repo"""
    if not GITHUB_TOKEN:
        print("No GITHUB_TOKEN set — skipping GitHub write")
        return False
    try:
        import base64
        url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{path}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
        body = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
        }
        if sha: body["sha"] = sha
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.put(url, headers=headers, json=body)
        if r.is_success:
            print(f"GitHub write OK: {path}")
            return True
        print(f"GitHub write failed {r.status_code}: {r.text[:200]}")
        return False
    except Exception as e:
        print(f"GitHub put error: {e}")
        return False

async def save_daily_predictions():
    """Save today's predictions to GitHub — uses same data as /games endpoint for consistency"""
    if not _cache["ready"]: return
    today = date.today().isoformat()
    path = f"data/predictions/{today}.json"
    existing, sha = await github_get_file(path)
    # Allow overwrite if hit_hr is still null (predictions not yet recorded)
    if existing:
        import json
        try:
            ex_recs = json.loads(existing)
            if any(r.get("hit_hr") is not None for r in ex_recs):
                print(f"Predictions already recorded for {today} — skipping")
                return
            print(f"Overwriting pending predictions for {today}")
        except Exception:
            pass
    try:
        # Fetch today's schedule
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{MLB_API}/schedule?sportId=1&date={today}&hydrate=team,probablePitcher")
            data = r.json()
        records = []
        for game_date in data.get("dates", []):
            for game in game_date.get("games", []):
                if game.get("status", {}).get("abstractGameState") == "Final": continue
                gid = game["gamePk"]
                home_team = game["teams"]["home"]["team"]["name"]
                away_team = game["teams"]["away"]["team"]["name"]
                away_team_id = game["teams"]["away"]["team"]["id"]
                home_team_id = game["teams"]["home"]["team"]["id"]
                away_p = game["teams"]["away"].get("probablePitcher", {})
                home_p = game["teams"]["home"].get("probablePitcher", {})
                gtime = game.get("gameDate", "")

                away_p_hand = home_p_hand = "R"
                if away_p.get("id"):
                    info = await fetch_player_hand(away_p.get("id"))
                    away_p_hand = info.get("pitch_hand", "R")
                if home_p.get("id"):
                    info = await fetch_player_hand(home_p.get("id"))
                    home_p_hand = info.get("pitch_hand", "R")

                stadium = STADIUMS.get(home_team, {})
                temp, wind_speed, wind_dir = 70, 0, 0
                if not stadium.get("dome") and stadium.get("lat"):
                    temp, wind_speed, wind_dir = await fetch_weather(stadium["lat"], stadium["lon"], gtime)

                # Try confirmed lineup first, fall back to projected
                lineup_away, lineup_home = [], []
                try:
                    async with httpx.AsyncClient(timeout=15) as client:
                        r2 = await client.get(f"{MLB_API}/game/{gid}/boxscore")
                        box = r2.json()
                    teams = box.get("teams", {})
                    def extract(side):
                        players = teams.get(side, {}).get("players", {})
                        return sorted([p for p in players.values() if p.get("battingOrder") and int(p["battingOrder"]) <= 900],
                                      key=lambda x: int(x["battingOrder"]))[:9]
                    ca, ch = extract("away"), extract("home")
                    if ca: lineup_away = ca
                    if ch: lineup_home = ch
                except Exception: pass

                lineup_away_source = "confirmed"
                lineup_home_source = "confirmed"
                if not lineup_away:
                    proj, _ = await fetch_projected_lineup(away_team_id, away_team)
                    lineup_away = proj
                    lineup_away_source = "projected"
                if not lineup_home:
                    proj, _ = await fetch_projected_lineup(home_team_id, home_team)
                    lineup_home = proj
                    lineup_home_source = "projected"

                for batters, team, opp_p_name, opp_p_hand, lineup_src in [
                    (lineup_away, away_team, home_p.get("fullName","TBD"), home_p_hand, lineup_away_source),
                    (lineup_home, home_team, away_p.get("fullName","TBD"), away_p_hand, lineup_home_source),
                ]:
                    for batter in batters:
                        if "person" in batter:
                            name = batter.get("person", {}).get("fullName", "")
                            pid = batter.get("person", {}).get("id")
                        else:
                            name = batter.get("name", "")
                            pid = batter.get("id")
                        if not name: continue
                        bat_hand = "R"
                        if pid:
                            info = await fetch_player_hand(pid)
                            bat_hand = info.get("bat_side", "R")
                        if bat_hand == "S": bat_hand = "L" if opp_p_hand == "R" else "R"
                        park_factor = get_park_hr_factor(home_team, bat_hand)
                        wx_mult, _ = calc_weather_multiplier(home_team, wind_speed, wind_dir, temp, bat_hand)
                        hr_prob, breakdown, _, _, _, _, _ = compute_hr_probability(
                            name, bat_hand, opp_p_name, opp_p_hand, park_factor, wx_mult, home_team)
                        if hr_prob < 5: continue
                        top_pitches = get_pitcher_top_pitches(opp_p_name)[:2]
                        pitch1 = top_pitches[0] if len(top_pitches) > 0 else {}
                        pitch2 = top_pitches[1] if len(top_pitches) > 1 else {}
                        pitch_score, _ = compute_pitch_matchup(opp_p_name, name)
                        pa_data = get_avg_pa_per_game(name)
                        # Raw batter season stats
                        bc2 = get_batter_stats(name, 2026)
                        bp2 = get_batter_stats(name, 2025)
                        pa26 = bc2.get("pa", 0); pa25 = bp2.get("pa", 0)
                        bwc2, bwp2 = get_batter_blend_weights(pa26, pa25)
                        barrel_s = round(blend(bc2.get("barrel_pct",0), bp2.get("barrel_pct",0), bwc2, bwp2), 1)
                        la_s     = round(blend(bc2.get("launch_angle",0), bp2.get("launch_angle",0), bwc2, bwp2), 1)
                        ev_s     = round(blend(bc2.get("exit_velo",0), bp2.get("exit_velo",0), bwc2, bwp2), 1)
                        iso_s    = round(blend(bc2.get("iso",0), bp2.get("iso",0), bwc2, bwp2), 3)
                        hh_s     = round(blend(bc2.get("hard_hit_pct",0), bp2.get("hard_hit_pct",0), bwc2, bwp2), 1)
                        k_s      = round(blend(bc2.get("k_pct",0), bp2.get("k_pct",0), bwc2, bwp2), 1)
                        # Raw batter L8D stats
                        b8d2 = get_batter_8d(name)
                        barrel_l8d = round(b8d2.get("barrel_pct",0), 1)
                        la_l8d     = round(b8d2.get("launch_angle",0), 1)
                        ev_l8d     = round(b8d2.get("exit_velo",0), 1)
                        iso_l8d    = round(b8d2.get("iso",0), 3)
                        hh_l8d     = round(b8d2.get("hard_hit_pct",0), 1)
                        pa_l8d     = int(b8d2.get("pa",0))
                        # Raw batter split vs pitcher hand
                        b_split2   = get_batter_split(name, opp_p_hand)
                        iso_split  = round(b_split2.get("iso",0), 3)
                        slg_split  = round(b_split2.get("slg",0), 3)
                        hr_split   = int(b_split2.get("hr",0))
                        pa_split   = int(b_split2.get("pa",0))
                        # Raw pitcher stats
                        pc2  = get_pitcher_stats(opp_p_name, 2026)
                        pp2b = get_pitcher_stats(opp_p_name, 2025)
                        ip26 = pc2.get("ip",0)
                        pwc2, pwp2 = get_pitcher_blend_weights(ip26, pp2b.get("ip",0))
                        pit_hr9_s   = round(blend(pc2.get("hr9",0), pp2b.get("hr9",0), pwc2, pwp2), 2)
                        pit_era_s   = round(blend(pc2.get("era",0), pp2b.get("era",0), pwc2, pwp2), 2)
                        pit_hh_s    = round(blend(pc2.get("hard_hit_pct",0), pp2b.get("hard_hit_pct",0), pwc2, pwp2), 1)
                        pit_k9_s    = round(blend(pc2.get("k9",0), pp2b.get("k9",0), pwc2, pwp2), 1)
                        # Pitcher split vs batter hand
                        p_split2    = get_pitcher_split(opp_p_name, bat_hand)
                        pit_hr9_vs  = round(p_split2.get("hr9",0), 2)
                        pit_slg_vs  = round(p_split2.get("slg",0), 3)
                        pit_k_vs    = round(p_split2.get("k_pct",0), 1)
                        pit_ip_vs   = round(p_split2.get("ip",0), 1)
                        records.append({
                            # Identity
                            "date": today, "name": name, "team": team,
                            "opp_pitcher": opp_p_name, "opp_pitcher_hand": opp_p_hand,
                            "bat_hand": bat_hand, "home_team": home_team,
                            "lineup_source": lineup_src,
                            # Model output
                            "model_hr_pct": hr_prob, "hit_hr": None,
                            # ── BATTER SEASON ──
                            "barrel_pct_season": barrel_s,
                            "la_season": la_s,
                            "ev_season": ev_s,
                            "iso_season": iso_s,
                            "hard_hit_season": hh_s,
                            "k_pct_season": k_s,
                            "hr_season": int(bc2.get("hr",0)),
                            "pa_season": pa26,
                            # ── BATTER L8D ──
                            "barrel_pct_l8d": barrel_l8d,
                            "la_l8d": la_l8d,
                            "ev_l8d": ev_l8d,
                            "iso_l8d": iso_l8d,
                            "hard_hit_l8d": hh_l8d,
                            "pa_l8d": pa_l8d,
                            "l8d_hr": get_l8d_hr(name),
                            # ── BATTER SPLIT vs PITCHER HAND ──
                            "iso_vs_hand": iso_split,
                            "slg_vs_hand": slg_split,
                            "hr_vs_hand": hr_split,
                            "pa_vs_hand": pa_split,
                            # ── MODEL USED (blended) ──
                            "barrel_pct_used": breakdown.get("barrel_use",0),
                            "la_used": breakdown.get("la_use",0),
                            # ── PITCHER SEASON ──
                            "pit_hr9_season": pit_hr9_s,
                            "pit_era_season": pit_era_s,
                            "pit_hard_hit_season": pit_hh_s,
                            "pit_k9_season": pit_k9_s,
                            "pit_ip_season": round(ip26,1),
                            # ── PITCHER SPLIT vs BATTER HAND ──
                            "pit_hr9_vs_hand": pit_hr9_vs,
                            "pit_slg_vs_hand": pit_slg_vs,
                            "pit_k_vs_hand": pit_k_vs,
                            "pit_ip_vs_hand": pit_ip_vs,
                            # ── CONTEXT ──
                            "park_factor": breakdown.get("park_factor",1.0),
                            "weather_mult": breakdown.get("weather_mult",1.0),
                            "bullpen_hr9": breakdown.get("bullpen_hr9",1.2),
                            "bullpen_vuln": breakdown.get("bullpen_vuln",1.0),
                            # ── MODEL MULTIPLIERS ──
                            "barrel_mult": breakdown.get("barrel_mult",1.0),
                            "la_mult": breakdown.get("la_mult",1.0),
                            "pit_vuln_mult": breakdown.get("pit_vuln_mult",1.0),
                            "bat_platoon_mult": breakdown.get("bat_platoon_mult",1.0),
                            "pit_platoon_mult": breakdown.get("pit_platoon_mult",1.0),
                            "hot_cold_mult": breakdown.get("hot_cold_mult",1.0),
                            "k_mult": breakdown.get("k_mult",1.0),
                            # ── PITCH DATA ──
                            "pitch_matchup_score": round(pitch_score,2),
                            "pitch1_type": pitch1.get("name",""),
                            "pitch1_usage": pitch1.get("usage",0),
                            "pitch1_delta": round(pitch1.get("batter_rv",0) - pitch1.get("pit_rv",0), 2) if pitch1 else 0,
                            "pitch2_type": pitch2.get("name",""),
                            "pitch2_usage": pitch2.get("usage",0),
                            "pitch2_delta": round(pitch2.get("batter_rv",0) - pitch2.get("pit_rv",0), 2) if pitch2 else 0,
                            "combined_pitch_delta": round(
                                (pitch1.get("usage",0)/100 * (pitch1.get("batter_rv",0) - pitch1.get("pit_rv",0)) if pitch1 else 0) +
                                (pitch2.get("usage",0)/100 * (pitch2.get("batter_rv",0) - pitch2.get("pit_rv",0)) if pitch2 else 0), 2
                            ),
                            # ── OPPORTUNITY ──
                            "avg_pa_per_game": pa_data.get("avg_pa_per_game",3.1),
                            "avg_ab_per_game": pa_data.get("avg_ab_per_game",2.8),
                            "games_played": pa_data.get("games",0),
                            # ── ROTATION METADATA ──
                            "rotation_round": get_rotation_round(),
                            "rotation_day": get_rotation_day(),
                            # ── ROUND 2 CANDIDATES (stored from day 1, evaluated at day 45) ──
                            "fb_pct_season": round(blend(bc2.get("fb_pct",0), bp2.get("fb_pct",0), bwc2, bwp2), 1),
                            "pull_pct_season": round(blend(bc2.get("pull_pct",0), bp2.get("pull_pct",0), bwc2, bwp2), 1),
                            "pit_fb_pct_allowed": round(blend(pc2.get("fb_pct",0), pp2b.get("fb_pct",0), pwc2, pwp2), 1),
                            "hard_hit_l8d": hh_l8d,
                            "k_pct_l8d": round(b8d2.get("k_pct",0), 1),
                            # ── ROUND 3 CANDIDATES ──
                            "pit_era_season": pit_era_s,
                            "pit_era_diff": round(pit_era_s - 4.20, 2) if pit_era_s > 0 else 0,
                            "pit_hr_fb_pct": round(blend(pc2.get("hr_fb_pct",0), pp2b.get("hr_fb_pct",0), pwc2, pwp2), 1),
                            "lineup_k_pct": 0,  # populated at game time in future
                            # ── ROUND 4 CANDIDATES ──
                            "pit_k9_season": pit_k9_s,
                        })
        if not records:
            print(f"No predictions to save for {today}")
            return
        import json
        content = json.dumps(records, indent=2)
        await github_put_file(path, content, f"predictions: {today} ({len(records)} batters)", sha)
        print(f"Saved {len(records)} predictions for {today}")
    except Exception as e:
        print(f"save_daily_predictions error: {e}")
        import traceback; traceback.print_exc()

async def record_results(target_date: str):
    """Fetch actual HR results for target_date and update the predictions file"""
    if not GITHUB_TOKEN: return
    path = f"data/predictions/{target_date}.json"
    content, sha = await github_get_file(path)
    if not content:
        print(f"No predictions file found for {target_date}")
        return
    import json
    try:
        records = json.loads(content)
    except Exception:
        return
    # Fetch box scores for that date
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{MLB_API}/schedule?sportId=1&date={target_date}&hydrate=team")
            sched = r.json()
        hr_hitters = set()
        actual_ab = {}  # name.lower() -> ab count
        for game_date in sched.get("dates", []):
            for game in game_date.get("games", []):
                if game.get("status", {}).get("abstractGameState") != "Final": continue
                gid = game["gamePk"]
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        r2 = await client.get(f"{MLB_API}/game/{gid}/boxscore")
                        box = r2.json()
                    for side in ["away", "home"]:
                        for _, p in box.get("teams", {}).get(side, {}).get("players", {}).items():
                            stats = p.get("stats", {}).get("batting", {})
                            name = p.get("person", {}).get("fullName", "")
                            if not name: continue
                            ab = int(stats.get("atBats", 0) or 0)
                            actual_ab[name.lower()] = ab
                            if int(stats.get("homeRuns", 0) or 0) > 0:
                                hr_hitters.add(name.lower())
                except Exception: continue
        # Update records with actual results — DNP if fewer than 2 AB
        updated = 0
        dnp_count = 0
        for rec in records:
            if rec.get("hit_hr") is None:
                nl = rec["name"].lower()
                ab = actual_ab.get(nl, 0)
                # Check partial name match too
                if ab == 0:
                    last = nl.split()[-1]
                    for k, v in actual_ab.items():
                        if last in k:
                            ab = v
                            break
                if ab < 2:
                    rec["hit_hr"] = "DNP"  # Did not play / not enough AB for ML
                    rec["actual_ab"] = ab
                    dnp_count += 1
                else:
                    rec["hit_hr"] = 1 if nl in hr_hitters else 0
                    rec["actual_ab"] = ab
                updated += 1
        content_updated = json.dumps(records, indent=2)
        await github_put_file(path, content_updated, f"results: {target_date} ({len(hr_hitters)} HRs, {dnp_count} DNP)", sha)
        print(f"Recorded results for {target_date}: {len(hr_hitters)} HR hitters, {dnp_count} DNP, {updated} records updated")
    except Exception as e:
        print(f"record_results error: {e}")
        import traceback; traceback.print_exc()

def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)
    loop.close()

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=run_async, args=(load_all_savant_data(),), daemon=True).start()
    asyncio.create_task(daily_refresh_loop())
    asyncio.create_task(load_model_weights())

# ── Player matching ──
def fuzzy_match(name: str, df: pd.DataFrame, col="name"):
    if df is None or df.empty or col not in df.columns:
        return None
    nl = name.lower().strip()
    # Exact match first
    exact = df[df[col].str.lower().str.strip() == nl]
    if not exact.empty:
        return exact.iloc[0]
    # Last name match
    parts = nl.split()
    if not parts: return None
    last = parts[-1]
    matches = df[df[col].str.lower().str.contains(last, na=False)]
    if len(matches) == 1:
        return matches.iloc[0]
    if len(matches) > 1:
        # Try first name too
        first = parts[0]
        refined = matches[matches[col].str.lower().str.contains(first, na=False)]
        if not refined.empty:
            return refined.iloc[0]
        # Multiple last name matches, no first name match — too ambiguous, return None
        # This prevents e.g. "Murakami" matching the wrong player
        return None
    return None

def gs(row, *keys, default=0.0):
    """Get stat from row trying multiple keys"""
    if row is None:
        return default
    for key in keys:
        val = row.get(key, None)
        if val is not None and str(val) not in ('nan', 'None', ''):
            try:
                return float(val)
            except:
                pass
    return default

# ── Stat getters ──
def get_batter_stats(name, year=2026):
    cur = current_season()
    df = _cache["bat_2026"] if year == cur else _cache["bat_2025"]
    row = fuzzy_match(name, df)
    if row is None:
        return {}
    stats = {
        "pa": gs(row, "pa"),
        "barrel_pct": gs(row, "barrel_pct"),
        "exit_velo": gs(row, "exit_velo"),
        "launch_angle": gs(row, "launch_angle"),
        "hard_hit_pct": gs(row, "hard_hit_pct"),
        "fb_pct": gs(row, "fb_pct"),
        "pull_pct": gs(row, "pull_pct"),
        "iso": gs(row, "iso"),
        "slg_percent": gs(row, "slg_percent"),
        "batting_avg": gs(row, "batting_avg"),
        "k_pct": gs(row, "k_pct"),
        "hr_fb_pct": gs(row, "hr_fb_pct"),
        "hr": gs(row, "hr"),
    }
    return stats
def get_batter_8d(name):
    df = _cache["bat_8d"]
    row = fuzzy_match(name, df)
    if row is None:
        return {}
    pa = gs(row, "pa")
    hr = gs(row, "hr")
    slg = gs(row, "slg_percent")
    avg = gs(row, "batting_avg")
    iso = slg - avg if slg > 0 else 0
    return {
        "pa": pa, "hr": hr,
        "barrel_pct": gs(row, "barrel_pct"),
        "exit_velo": gs(row, "exit_velo"),
        "launch_angle": gs(row, "launch_angle"),
        "hard_hit_pct": gs(row, "hard_hit_pct"),
        "pull_pct": gs(row, "pull_pct"),
        "iso": iso, "k_pct": gs(row, "k_pct"),
        "slg": slg, "avg": avg,
        "hr_rate": (hr / max(pa, 1)) * 600 if pa > 0 else 0,
    }

def get_batter_l5g(name):
    nl = name.lower().strip()
    data = _cache["bat_l5g"]
    if nl in data: return data[nl]
    last = nl.split()[-1]
    for k, v in data.items():
        if last in k: return v
    return {}

def get_l8d_hr(name):
    """Get reliable L8D HR count from MLB Stats API lastXGames=8"""
    nl = name.lower().strip()
    data = _cache["bat_l8d_hr"]
    if nl in data: return data[nl].get("hr", 0)
    last = nl.split()[-1]
    for k, v in data.items():
        if last in k: return v.get("hr", 0)
    return 0

def get_avg_pa_per_game(name):
    """Get batter's avg PA per game this season — key ML feature for opportunity"""
    nl = name.lower().strip()
    data = _cache.get("bat_games", {})
    if nl in data: return data[nl]
    last = nl.split()[-1]
    for k, v in data.items():
        if last in k: return v
    return {"games": 0, "avg_pa_per_game": 3.1, "avg_ab_per_game": 2.8}

def get_batter_split(name, pit_hand):
    df = _cache["bat_vs_lhp"] if pit_hand == "L" else _cache["bat_vs_rhp"]
    row = fuzzy_match(name, df)
    if row is None:
        return {}
    return {
        "pa":         gs(row, "pa"),
        "hr":         gs(row, "hr"),
        "iso":        gs(row, "iso"),
        "slg":        gs(row, "slg"),
        "woba":       gs(row, "woba"),
        "k_pct":      gs(row, "k_pct"),
        "barrel_pct": gs(row, "barrel_pct"),
        "hr_rate":    (gs(row, "hr") / max(gs(row, "pa"), 1)) * 600 if gs(row, "pa") > 0 else 0,
    }

def get_pitcher_stats(name, year=2026):
    cur = current_season()
    df = _cache["pit_2026"] if year == cur else _cache["pit_2025"]
    row = fuzzy_match(name, df)
    nl = name.lower().strip()
    ip_data = _cache["player_ip"].get(nl, {})
    # Try last name match for IP data
    if not ip_data:
        last = nl.split()[-1]
        for k, v in _cache["player_ip"].items():
            if last in k:
                ip_data = v
                break
    ip = ip_data.get("ip", 0)
    hr9 = ip_data.get("hr9", 0)
    era = ip_data.get("era", 0)
    k9  = ip_data.get("k9", 0)
    avg_ip = ip_data.get("avg_ip", 5.0)
    gs_val = ip_data.get("gs", 0)
    if row is None:
        return {"era": era, "ip": ip, "hr9": hr9, "k9": k9, "avg_ip": avg_ip, "gs": gs_val,
                "hard_hit_pct": 0, "barrel_pct_allowed": 0, "fb_pct": 0, "k_pct": 0, "hr_fb_pct": 0}
    return {
        "era": era or gs(row, "era"),
        "ip": ip, "hr9": hr9, "k9": k9, "avg_ip": avg_ip, "gs": gs_val,
        "hard_hit_pct": gs(row, "hard_hit_pct"),
        "barrel_pct_allowed": gs(row, "barrel_pct_allowed"),
        "fb_pct": gs(row, "fb_pct"),
        "k_pct": gs(row, "k_pct"),
        "hr_fb_pct": 0,
    }

def get_pitcher_split(name, vs_hand):
    df = _cache["pit_vs_lhh"] if vs_hand == "L" else _cache["pit_vs_rhh"]
    row = fuzzy_match(name, df)
    if row is None:
        return {}
    pa  = gs(row, "pa")
    hr  = gs(row, "hr")
    ip  = gs(row, "ip") if gs(row, "ip") > 0 else pa / 4.0
    return {
        "pa":           pa,
        "ip":           round(ip, 1),
        "hr":           hr,
        "hr9":          gs(row, "hr9"),
        "k_pct":        gs(row, "k_pct"),
        "slg":          gs(row, "slg"),
        "woba":         gs(row, "woba"),
        "iso":          gs(row, "iso"),
        "hard_hit_pct": 0,
        "barrel_pct":   0,
    }

def get_pitcher_top_pitches(pitcher_name):
    df = _cache["pit_arsenal"]
    if df.empty:
        return []
    last = pitcher_name.split()[-1].lower()
    matches = df[df["name"].str.lower().str.contains(last, na=False)]
    if matches.empty:
        return []
    # If multiple rows (same pitcher different contexts), take the one with most PA
    if "pa" in matches.columns and len(matches) > 1:
        matches = matches.sort_values("pa", ascending=False)
    pitches = []
    seen_codes = set()
    seen_types = set()
    for _, row in matches.iterrows():
        pt = str(row.get("pitch_type", "")).upper()
        if pt in seen_types: continue  # skip duplicate pitch types from multiple rows
        code = PITCH_TYPE_MAP.get(pt)
        if not code: continue
        if code in seen_codes: continue  # skip duplicate codes
        usage = gs(row, "pitch_usage") * 100 if gs(row, "pitch_usage") <= 1 else gs(row, "pitch_usage")
        rv = gs(row, "run_value_per_100")
        if usage >= 5:
            pitches.append({
                "code": code,
                "name": row.get("pitch_name", PITCH_DISPLAY.get(code, code)),
                "usage": round(usage, 1),
                "pit_rv": round(rv, 2),
                "pitch_type": pt,
            })
            seen_codes.add(code)
            seen_types.add(pt)
    pitches.sort(key=lambda x: x["usage"], reverse=True)
    pitches = pitches[:3]
    # Normalize usage to sum to 100% across top pitches
    total_usage = sum(p["usage"] for p in pitches)
    if total_usage > 0 and total_usage != 100.0:
        for p in pitches:
            p["usage"] = round(p["usage"] / total_usage * 100, 1)
    return pitches

def get_batter_pitch_rv(batter_name, pitch_code):
    df = _cache["bat_arsenal"]
    if df.empty:
        return None
    last = batter_name.split()[-1].lower()
    matches = df[df["name"].str.lower().str.contains(last, na=False)]
    if matches.empty:
        return None
    # Find pitch type
    target_types = [k for k, v in PITCH_TYPE_MAP.items() if v == pitch_code]
    for _, row in matches.iterrows():
        pt = str(row.get("pitch_type", "")).upper()
        if pt in target_types:
            return gs(row, "run_value_per_100")
    return None

def compute_pitch_matchup(pitcher_name, batter_name):
    top_pitches = get_pitcher_top_pitches(pitcher_name)
    if not top_pitches:
        return 0.0, []
    details = []
    total_bonus = 0.0
    for pitch in top_pitches:
        code = pitch["code"]
        usage = pitch["usage"] / 100.0
        pit_rv = pitch["pit_rv"]
        bat_rv = get_batter_pitch_rv(batter_name, code)
        if bat_rv is None:
            continue
        combined = (bat_rv * 0.6) + (pit_rv * 0.4)
        bonus = max(min(combined * usage * 1.5, 4), -4)
        total_bonus += bonus
        details.append({
            "name": pitch["name"],
            "usage": pitch["usage"],
            "pit_rv": round(pit_rv, 2),
            "bat_rv": round(bat_rv, 2),
            "combined": round(combined, 2),
            "bonus": round(bonus, 2),
        })
    return round(max(min(total_bonus, 8), -8), 2), details

# ── Model helpers ──
def blend(v1, v2, w1, w2):
    c, p = float(v1 or 0), float(v2 or 0)
    if c == 0 and p == 0: return 0
    if c == 0: return p
    if p == 0: return c
    return c * w1 + p * w2

def get_batter_blend_weights(pa_2026, pa_2025):
    pa = float(pa_2026 or 0)
    pa25 = float(pa_2025 or 0)
    # If no 2025 MLB data (new MLB player), use whatever 2026 data we have
    if pa25 < 10:
        return 1.0, 0.0  # no career MLB data → use 2026 only, even if small
    if pa >= 200:   return 1.0, 0.0
    elif pa >= 150: w = 0.80; return w, 1.0 - w
    elif pa >= 100: w = 0.60; return w, 1.0 - w
    elif pa >= 50:  w = 0.30; return w, 1.0 - w
    else:           return 0.0, 1.0  # <50 PA but has 2025 data → use career

def get_pitcher_blend_weights(ip_2026, ip_2025):
    ip = float(ip_2026 or 0)
    if ip >= 30: return 1.0, 0.0
    elif ip >= 10: w = (ip - 10) / 20.0; return 0.5 + w * 0.5, 0.5 - w * 0.5
    elif ip > 0: return 0.5, 0.5
    else: return 0.0, 1.0

def get_park_hr_factor(home_team, batter_hand):
    pf = PARK_HR_FACTORS.get(home_team, {"L": 1.0, "R": 1.0})
    return pf.get(batter_hand if batter_hand in ("L", "R") else "R", 1.0)

def angle_diff(a, b):
    diff = abs(a - b) % 360
    return diff if diff <= 180 else 360 - diff

def calc_weather_multiplier(home_team, wind_speed, wind_direction, temperature, batter_hand="R"):
    stadium = STADIUMS.get(home_team)
    if not stadium: return 1.0, "Unknown"
    if stadium.get("dome"): return 1.0, "Dome"
    hr_bearing = stadium.get("hr_bearing", 225)
    open_factor = stadium.get("open_factor", 0.5)
    diff = angle_diff(wind_direction, hr_bearing)
    alignment = math.cos(math.radians(diff))
    speed_factor = 0 if wind_speed < 5 else 0.3 if wind_speed < 10 else 0.7 if wind_speed < 16 else 1.0
    wind_mult = 1.0 + (alignment * speed_factor * 0.12 * open_factor)
    temp_mult = 1.06 if temperature >= 80 else 1.02 if temperature >= 70 else 0.91 if temperature < 50 else 0.96 if temperature < 60 else 1.0
    if abs(alignment) <= 0.5 and wind_speed >= 10:
        cross_diff = (wind_direction - hr_bearing) % 360
        direction_label = "Favors Lefties" if 45 < cross_diff < 225 else "Favors Righties"
    elif alignment > 0.5 and wind_speed >= 10: direction_label = "Blowing Out"
    elif alignment < -0.5 and wind_speed >= 10: direction_label = "Blowing In"
    elif wind_speed < 5: direction_label = "Calm"
    else: direction_label = "Crosswind"
    return round(wind_mult * temp_mult, 3), direction_label

def sigmoid_to_prob(raw_score):
    centered = (raw_score - 50) / 18.0
    sigmoid = 1 / (1 + math.exp(-centered))
    return round(min(max(0.02 + sigmoid * 0.25, 0.02), 0.25) * 100, 1)

def safe_mult(value, lg_avg, weight_key="", sample=None, min_sample=0, cap_high=2.50, cap_low=0.30):
    """
    Safe multiplier that returns 1.0 (neutral) when:
    - value is missing, zero, or None
    - sample size is below minimum threshold
    Never collapses to 0, never goes haywire on tiny samples.
    """
    if value is None or value == 0:
        return 1.0  # missing stat — neutral, doesn't help or hurt
    if min_sample > 0 and sample is not None and sample < min_sample:
        return 1.0  # insufficient sample — neutral until we have real data
    w = W(weight_key) if weight_key else 1.0
    raw = (float(value) / float(lg_avg)) ** w
    return max(min(raw, cap_high), cap_low)

def compute_hr_prob_multiplicative(
        name, bat_hand, opp_p_name, opp_p_hand, park_factor, weather_mult, home_team=""):
    """
    Multiplicative HR probability model.
    P(HR) = Base Rate × Barrel% × LA × Pitcher Vuln × Batter Platoon ×
            Pitcher Platoon × Park × Weather × Hot/Cold × K% penalty
    Hard cap: 28%
    """
    # ── Data fetch ──
    bc  = get_batter_stats(name, 2026)
    bp  = get_batter_stats(name, 2025)
    b8d = get_batter_8d(name)
    b_split_vs_hand = get_batter_split(name, opp_p_hand)   # batter vs pitcher hand
    b_split_opp     = get_batter_split(name, "R" if opp_p_hand == "L" else "L")  # vs opposite hand
    p_split_vs_bat  = get_pitcher_split(opp_p_name, bat_hand)  # pitcher vs batter hand

    pa_26 = bc.get("pa", 0); pa_25 = bp.get("pa", 0)
    bwc, bwp = get_batter_blend_weights(pa_26, pa_25)
    has_8d = b8d.get("pa", 0) >= 3
    total_pa = pa_26 + pa_25

    # ── Step 1: Base HR rate (career/season blend by PA) ──
    # Blend weights: <150 PA → 30% season / 70% career; 150-300 → 60/40; 300+ → 85/15
    hr_season = bc.get("hr", 0)
    hr_career  = blend(bc.get("hr", 0), bp.get("hr", 0), bwc, bwp)
    pa_season  = max(pa_26, 1)

    # HR/PA rate from season
    hr_per_pa_season = hr_season / pa_season if pa_season > 0 else 0

    # Career blend rate
    total_pa_safe = max(total_pa, 1)
    hr_per_pa_career = hr_career / max(pa_25 + pa_26, 1) if (pa_25 + pa_26) > 0 else 0.035

    # PA-weighted blend — 200+ PA = trust season fully
    if pa_26 >= 200:
        base_rate = hr_per_pa_season
    elif pa_26 >= 150:
        base_rate = hr_per_pa_season * 0.80 + hr_per_pa_career * 0.20
    elif pa_26 >= 100:
        base_rate = hr_per_pa_season * 0.60 + hr_per_pa_career * 0.40
    elif pa_26 >= 50:
        base_rate = hr_per_pa_season * 0.30 + hr_per_pa_career * 0.70
    else:
        base_rate = hr_per_pa_career  # too early — use career only

    # Floor: use league avg ~2.8% if no data
    if base_rate <= 0:
        base_rate = 0.028
    base_rate = min(base_rate, 0.12)  # no single batter truly > 12% per PA

    # Small sample confidence gate (not a hard cut — just dampens)
    if total_pa < 30:   base_rate = base_rate * 0.55 + 0.028 * 0.45
    elif total_pa < 60: base_rate = base_rate * 0.75 + 0.028 * 0.25

    running = base_rate

    # ── Step 2: Barrel% — season + L8D weighted separately via safe_mult ──
    LG_BARREL = LEAGUE_CONSTANTS["lg_barrel_pct"]
    barrel_season = blend(bc.get("barrel_pct", 0), bp.get("barrel_pct", 0), bwc, bwp)
    barrel_l8d    = b8d.get("barrel_pct", 0) if has_8d else 0
    barrel_season_mult = safe_mult(barrel_season, LG_BARREL, "barrel_season_w", pa_26, 20)
    barrel_l8d_mult    = safe_mult(barrel_l8d, LG_BARREL, "barrel_l8d_w",
                                   b8d.get("pa", 0) if has_8d else 0, 8)
    if has_8d and b8d.get("pa", 0) >= 8:
        barrel_mult = barrel_season_mult * 0.60 + barrel_l8d_mult * 0.40
    else:
        barrel_mult = barrel_season_mult
    barrel_use = barrel_season if barrel_season > 0 else LG_BARREL
    running *= barrel_mult

    # ── Step 3: Launch angle — season + L8D weighted separately via safe_mult ──
    la_season = blend(bc.get("launch_angle", 0), bp.get("launch_angle", 0), bwc, bwp)
    la_l8d    = b8d.get("launch_angle", 0) if has_8d else 0

    def la_to_raw(la):
        if not la or la <= 0: return None
        if 25 <= la <= 35:   return 1.00
        elif 20 <= la < 25:  return 0.90
        elif 35 < la <= 40:  return 0.90
        elif 18 <= la < 20:  return 0.80
        elif 40 < la <= 45:  return 0.80
        else:                return 0.75

    la_s_raw = la_to_raw(la_season)
    la_l_raw = la_to_raw(la_l8d)
    # Apply weights as exponent on the raw LA multiplier
    la_season_mult = (la_s_raw ** W("la_season_w")) if la_s_raw else 1.0
    la_l8d_mult    = (la_l_raw ** W("la_l8d_w")) if la_l_raw and has_8d and b8d.get("pa",0) >= 8 else 1.0
    if has_8d and b8d.get("pa", 0) >= 8 and la_l_raw:
        la_mult = la_season_mult * 0.60 + la_l8d_mult * 0.40
    else:
        la_mult = la_season_mult
    la_use = la_season if la_season > 0 else 20.0
    running *= la_mult

    # ── Step 4: Pitcher vulnerability — season + vs-hand via safe_mult ──
    pc = get_pitcher_stats(opp_p_name, 2026)
    pp2 = get_pitcher_stats(opp_p_name, 2025)
    ip_26 = pc.get("ip", 0)
    pwc, pwp = get_pitcher_blend_weights(ip_26, pp2.get("ip", 0))

    LG_HR9 = LEAGUE_CONSTANTS["lg_hr9"]
    LG_HH  = LEAGUE_CONSTANTS["lg_hard_hit"]
    pit_hr9_season  = blend(pc.get("hr9", 0), pp2.get("hr9", 0), pwc, pwp)
    pit_hr9_vs_hand = p_split_vs_bat.get("hr9", 0)
    pit_hard        = blend(pc.get("hard_hit_pct", 0), pp2.get("hard_hit_pct", 0), pwc, pwp)
    pit_ip_vs_hand  = p_split_vs_bat.get("ip", 0)
    total_ip        = ip_26 + pp2.get("ip", 0)

    m_hr9_s  = safe_mult(pit_hr9_season,  LG_HR9, "pit_hr9_season_w",  total_ip, 10)
    m_hr9_vs = safe_mult(pit_hr9_vs_hand, LG_HR9, "pit_hr9_vs_hand_w", pit_ip_vs_hand, 5)
    m_hard   = safe_mult(pit_hard, LG_HH, "", total_ip, 10)

    # Combine: average of available signals (don't multiply — correlated stats)
    pit_signals = []
    if total_ip >= 10 and pit_hr9_season > 0:  pit_signals.append(m_hr9_s)
    if pit_ip_vs_hand >= 5 and pit_hr9_vs_hand > 0: pit_signals.append(m_hr9_vs)
    if total_ip >= 10 and pit_hard > 0:         pit_signals.append(m_hard)
    pit_vuln_mult = sum(pit_signals) / len(pit_signals) if pit_signals else 1.0
    pit_vuln_mult = max(min(pit_vuln_mult, 1.80), 0.50)
    running *= pit_vuln_mult

    # ── Step 5: Batter platoon — ISO vs hand via safe_mult ──
    iso_vs_hand   = b_split_vs_hand.get("iso", 0)
    iso_overall   = blend(bc.get("iso", 0), bp.get("iso", 0), bwc, bwp)
    split_pa      = b_split_vs_hand.get("pa", 0)
    # Need both iso_vs_hand and iso_overall as ratio — use safe_mult on ratio
    if iso_overall > 0 and iso_vs_hand > 0 and split_pa >= 30:
        bat_platoon_raw = iso_vs_hand / iso_overall
        bat_platoon_mult = safe_mult(bat_platoon_raw, 1.0, "bat_platoon_w",
                                     split_pa, 30, cap_high=1.60, cap_low=0.60)
    else:
        bat_platoon_mult = 1.0
    running *= bat_platoon_mult

    # ── Step 6: Pitcher platoon — SLG vs hand via safe_mult ──
    slg_vs_bat      = p_split_vs_bat.get("slg", 0)
    p_split_opp     = get_pitcher_split(opp_p_name, "L" if bat_hand == "R" else "R")
    split_ip_vs_bat = p_split_vs_bat.get("ip", 0)
    slg_sources     = [x for x in [slg_vs_bat, p_split_opp.get("slg", 0)] if x > 0]
    slg_overall_pit = sum(slg_sources) / len(slg_sources) if slg_sources else 0
    if slg_overall_pit > 0 and slg_vs_bat > 0 and split_ip_vs_bat >= 5:
        pit_platoon_raw  = slg_vs_bat / slg_overall_pit
        pit_platoon_mult = safe_mult(pit_platoon_raw, 1.0, "pit_platoon_w",
                                     split_ip_vs_bat, 5, cap_high=1.60, cap_low=0.60)
    else:
        pit_platoon_mult = 1.0
    running *= pit_platoon_mult

    # ── Step 7: Park multiplier ──
    park_w = W("park_w")
    park_mult_applied = park_factor ** park_w if park_factor > 0 else 1.0
    running *= park_mult_applied

    # ── Step 8: Weather multiplier ──
    weather_w = W("weather_w")
    weather_mult_applied = weather_mult ** weather_w if weather_mult > 0 else 1.0
    running *= weather_mult_applied

    # ── Step 9: Hot/cold — display signal only, NOT in model ──
    # Removed from calculation — L8D HR count is shown on the table as a visual signal
    # ML will determine if it actually matters. Keeping calc for breakdown display only.
    hot_cold_mult = 1.0
    if has_8d and b8d.get("pa", 0) >= 8:
        pa_8d = b8d.get("pa", 0)
        hr_8d_count = get_l8d_hr(name)
        hr_8d_rate  = hr_8d_count / pa_8d
        if base_rate > 0:
            ratio = hr_8d_rate / base_rate
            hot_cold_mult = max(min(ratio, 1.20), 0.85)
    # NOT applied to running — hot_cold_mult stored for ML analysis only
    # running *= hot_cold_mult  <-- removed

    # ── Step 10: K% penalty — safe_mult aware ──
    k_season = blend(bc.get("k_pct", 0), bp.get("k_pct", 0), bwc, bwp)
    k_w = W("k_pct_w")
    if k_season >= 35:   k_mult = 0.88 ** k_w
    elif k_season >= 30: k_mult = 0.94 ** k_w
    elif k_season >= 25: k_mult = 0.97 ** k_w
    else:                k_mult = 1.0
    if k_season == 0:    k_mult = 1.0  # missing K% — neutral
    running *= k_mult

    # ── Hard cap + bullpen blend ──
    LG_BULLPEN_HR9 = LEAGUE_CONSTANTS["lg_bullpen_hr9"]
    bullpen_data   = _cache.get("team_bullpen", {}).get(home_team, {})
    bullpen_hr9    = bullpen_data.get("hr9", LG_BULLPEN_HR9)
    bullpen_vuln   = safe_mult(bullpen_hr9, LG_BULLPEN_HR9, "bullpen_w",
                               cap_high=1.80, cap_low=0.50)
    # Bullpen component — uses batter skill + context + bullpen vuln
    bullpen_component = (base_rate * barrel_mult * la_mult * bat_platoon_mult *
                         park_mult_applied * weather_mult_applied * k_mult * bullpen_vuln)
    bullpen_w_blend = W("bullpen_w") if W("bullpen_w") != 1.0 else 0.25
    running = (running * (1 - bullpen_w_blend)) + (bullpen_component * bullpen_w_blend)

    hr_prob = round(min(running * 100, LEAGUE_CONSTANTS["hr_prob_cap"]), 1)

    # ── Build breakdown for frontend ──
    pitch_bonus, pitch_details = compute_pitch_matchup(opp_p_name, name)
    archetype = get_archetype(barrel_season, k_season,
                              blend(bc.get("fb_pct", 0), bp.get("fb_pct", 0), bwc, bwp),
                              iso_overall if iso_overall else blend(bc.get("iso",0), bp.get("iso",0), bwc, bwp))
    trend = get_trend(b8d, bc)

    reasons = []
    if barrel_season >= 12: reasons.append(f"Barrel {barrel_season:.1f}%")
    if iso_vs_hand > 0.220: reasons.append(f"ISO vs hand .{int(iso_vs_hand*1000):03d}")
    if pit_hr9_season > 1.3: reasons.append(f"SP {pit_hr9_season:.1f} HR/9")
    if pit_hard > 40: reasons.append(f"SP {pit_hard:.1f}% HH")
    if park_factor >= 1.15: reasons.append("HR-friendly park")
    elif park_factor <= 0.90: reasons.append("Pitcher-friendly park")

    platoon_tag = None
    if bat_platoon_mult >= 1.20:
        platoon_tag = f"Batter strong vs {opp_p_hand}HP"
    if pit_platoon_mult >= 1.20:
        platoon_tag = (platoon_tag + " + " if platoon_tag else "") + f"SP weak vs {bat_hand}HB"

    n_components = len(pit_signals)
    conf = "High" if n_components >= 2 and pa_26 >= 50 else "Medium" if n_components >= 1 else "Low"
    blend_note = f"{int(bwc*100)}% 2026/{int(bwp*100)}% 2025" + (" + 8d" if has_8d else "")

    breakdown = {
        "base_rate": round(base_rate * 100, 2),
        "barrel_use": round(barrel_use, 1), "barrel_season": round(barrel_season, 1),
        "barrel_l8d": round(barrel_l8d, 1), "barrel_mult": round(barrel_mult, 3),
        "la_use": round(la_use, 1), "la_season": round(la_season, 1),
        "la_l8d": round(la_l8d, 1), "la_mult": round(la_mult, 3),
        "pit_hr9": round(pit_hr9_season, 2), "pit_hard": round(pit_hard, 1),
        "pit_hr9_vs_hand": round(pit_hr9_vs_hand, 2),
        "pit_vuln_mult": round(pit_vuln_mult, 3),
        "bat_platoon_mult": round(bat_platoon_mult, 3),
        "pit_platoon_mult": round(pit_platoon_mult, 3),
        "iso_vs_hand": round(iso_vs_hand, 3), "iso_overall": round(iso_overall, 3),
        "slg_vs_bat": round(slg_vs_bat, 3) if split_ip_vs_bat >= 5 else 0,
        "split_pa": split_pa,
        "split_ip_vs_bat": round(split_ip_vs_bat, 1),
        # Context
        "park_factor": round(park_factor, 3),
        "weather_mult": round(weather_mult, 3),
        "hot_cold_mult": round(hot_cold_mult, 3),
        "k_mult": round(k_mult, 3), "k_season": round(k_season, 1),
        # Running
        "hr_prob": hr_prob, "has_8d": has_8d,
        "blend_note": blend_note,
        "pit_blend_note": f"{int(pwc*100)}% 2026 / {int(pwp*100)}% 2025 ({ip_26:.0f} IP)",
        # Legacy fields for Research tab compatibility
        "barrel_s": round(barrel_use, 1), "s1_barrel": round(barrel_mult * 10, 2),
        "iso_use": round(iso_vs_hand if iso_vs_hand > 0 else iso_overall, 3),
        "la_s": round(la_use, 1), "s1_la": round(la_mult * 9, 2),
        "pull_s": round(blend(bc.get("pull_pct", 0), bp.get("pull_pct", 0), bwc, bwp), 1),
        "s1_pull": 0, "fb_s": 0, "s1_fb": 0, "s1_iso": 0,
        "hr_rate_8d": round(b8d.get("hr", 0) / max(b8d.get("pa", 1), 1) * 600, 1) if has_8d else 0,
        "s1_hr8d": round(hot_cold_mult, 3),
        "batter_score": round(running * 100, 2),
        "k_s": round(k_season, 1), "k_cap": round(k_mult, 3),
        "pit_modifier": round(pit_vuln_mult, 3),
        "platoon_magnitude": round((pit_platoon_mult - 1.0) * 100, 1),
        "hr9_split": round(p_split_vs_bat.get("hr9", 0), 2),
        "hr9_season": round(pit_hr9_season, 2),
        "split_ip": round(split_ip_vs_bat, 1),
        "split_brl": round(b_split_vs_hand.get("barrel_pct", 0), 1),
        "split_iso": round(iso_vs_hand, 3),
        "split_slg": round(b_split_vs_hand.get("slg", 0), 3),
        "split_woba": round(b_split_vs_hand.get("woba", 0), 3),
        "split_hr": int(b_split_vs_hand.get("hr", 0)),
        "split_pa": split_pa,
        "split_k_pct": round(b_split_vs_hand.get("k_pct", 0), 1),
        "hr_season": int(bc.get("hr", 0)),
        "pa_8d": int(b8d.get("pa", 0)),
        "barrel_8d_raw": round(b8d.get("barrel_pct", 0), 1),
        "iso_8d": round(b8d.get("iso", 0), 3),
        "pull_8d": round(b8d.get("pull_pct", 0), 1),
        "la_8d_raw": round(la_l8d, 1),
        "pitch_bonus": pitch_bonus, "pitch_breakdown": pitch_details,
        "after_k": round(running * 100, 1), "after_context": round(running * 100, 1),
        "n_pit_components": n_components,
        # ── Data confidence score — X/8 real stats ──
        # Each of the 8 active model slots gets 1 point if it has real data (not a default)
        "data_conf": {
            "barrel":       1 if barrel_season > 0 and pa_26 >= 20 else 0,
            "la":           1 if la_season > 0 and pa_26 >= 20 else 0,
            "pit_hr9":      1 if pit_hr9_season > 0 and total_ip >= 10 else 0,
            "pit_hr9_hand": 1 if pit_hr9_vs_hand > 0 and pit_ip_vs_hand >= 5 else 0,
            "iso_vs_hand":  1 if iso_vs_hand > 0 and split_pa >= 30 else 0,
            "park":         1 if park_factor != 1.0 else 0,
            "pitch_delta":  1 if pitch_bonus != 0 else 0,
            "bat_platoon":  1 if bat_platoon_mult != 1.0 else 0,
        },
        "pit_slg_overall": round(slg_overall_pit, 3),
        "bullpen_hr9": round(bullpen_hr9, 2),
        "bullpen_vuln": round(bullpen_vuln, 3),
    }
    return hr_prob, breakdown, archetype, trend, reasons, platoon_tag, conf

def get_archetype(barrel_pct, k_pct, fb_pct, iso):
    if barrel_pct >= 10 and k_pct >= 28: return "Boom/Bust"
    elif barrel_pct >= 10 and k_pct < 22: return "Pure Power"
    elif barrel_pct >= 7 and fb_pct >= 38: return "Power"
    elif iso >= 0.180 and k_pct < 20: return "Balanced"
    elif k_pct >= 28: return "High K"
    else: return "Contact"

def get_trend(b8d, bc):
    if not b8d or b8d.get("pa", 0) < 3: return "Steady"
    score = 0
    hr_rate = b8d.get("hr_rate", 0)
    if hr_rate > 25: score += 2
    elif hr_rate > 12: score += 1
    elif hr_rate == 0 and b8d.get("pa", 0) >= 10: score -= 1
    brl_8d = b8d.get("barrel_pct", 0)
    brl_s = bc.get("barrel_pct", 0)
    if brl_8d > 0 and brl_s > 0:
        diff = brl_8d - brl_s
        if diff >= 5: score += 2
        elif diff >= 2: score += 1
        elif diff <= -5: score -= 2
        elif diff <= -2: score -= 1
    iso_8d = b8d.get("iso", 0)
    iso_s = bc.get("iso", 0)
    if iso_8d > 0 and iso_s > 0:
        if iso_8d - iso_s >= 0.080: score += 1
        elif iso_8d - iso_s <= -0.080: score -= 1
    if score >= 2: return "Heating Up"
    elif score <= -2: return "Cooling Off"
    return "Steady"

def compute_hr_probability(name, bat_hand, opp_p_name, opp_p_hand, park_factor, weather_mult, home_team=""):
    return compute_hr_prob_multiplicative(name, bat_hand, opp_p_name, opp_p_hand, park_factor, weather_mult, home_team)

def _compute_hr_probability_legacy(name, bat_hand, opp_p_name, opp_p_hand, park_factor, weather_mult):
    bc = get_batter_stats(name, 2026)
    bp = get_batter_stats(name, 2025)
    b8d = get_batter_8d(name)
    b_split = get_batter_split(name, opp_p_hand)

    pa_26 = bc.get("pa", 0); pa_25 = bp.get("pa", 0)
    bwc, bwp = get_batter_blend_weights(pa_26, pa_25)
    has_8d = b8d.get("pa", 0) >= 3
    w_s = 0.70 if has_8d else 1.0
    w_8 = 0.30 if has_8d else 0.0

    def blend3(s26, s25, d8):
        s = blend(s26, s25, bwc, bwp)
        return round(s * w_s + d8 * w_8, 2) if (has_8d and d8 > 0) else round(s, 2)

    barrel_s = blend3(bc.get("barrel_pct", 0), bp.get("barrel_pct", 0), b8d.get("barrel_pct", 0))
    fb_s     = blend3(bc.get("fb_pct", 0), bp.get("fb_pct", 0), 0)
    pull_s   = blend3(bc.get("pull_pct", 0), bp.get("pull_pct", 0), b8d.get("pull_pct", 0))
    la_s     = blend3(bc.get("launch_angle", 0), bp.get("launch_angle", 0), b8d.get("launch_angle", 0))
    k_s      = blend3(bc.get("k_pct", 0), bp.get("k_pct", 0), 0)
    hr_fb_s  = 0

    iso_split  = b_split.get("iso", 0) if b_split.get("pa", 0) >= 20 else 0
    iso_season = blend(bc.get("iso", 0), bp.get("iso", 0), bwc, bwp)
    iso_base   = iso_split if iso_split > 0 else iso_season
    iso_8d     = b8d.get("iso", 0) if has_8d else 0
    iso_use    = round(iso_base * w_s + iso_8d * w_8, 3) if iso_8d > 0 else iso_base

    hr_rate_8d    = b8d.get("hr_rate", 0)
    barrel_season = blend(bc.get("barrel_pct", 0), bp.get("barrel_pct", 0), bwc, bwp)
    barrel_8d_raw = b8d.get("barrel_pct", 0)
    trend = get_trend(b8d, bc)

    pitch_bonus, pitch_details = compute_pitch_matchup(opp_p_name, name)

    # Step 1 — Batter score
    s1_barrel = round(min(barrel_s / 15.0, 1.0) * 30, 2)
    s1_iso    = round(min(iso_use / 0.280, 1.0) * 14, 2)
    s1_fb     = 0
    s1_pull   = round(min(pull_s / 50.0, 1.0) * 8, 2)
    s1_la     = round(min(max(la_s - 10, 0) / 20.0, 1.0) * 9, 2) if la_s > 0 else 0
    s1_hrfb   = 0
    s1_hr8d   = round(min(hr_rate_8d / 40.0, 1.0) * 5, 2) if has_8d and b8d.get("pa", 0) >= 10 else 0

    batter_score = s1_barrel + s1_iso + s1_pull + s1_la + s1_hr8d + pitch_bonus

    total_pa = pa_26 + pa_25
    if total_pa < 30: batter_score *= 0.55
    elif total_pa < 60: batter_score *= 0.75
    elif pa_26 < 10: batter_score *= 0.80

    archetype = get_archetype(barrel_season, k_s, fb_s, iso_season)

    # Step 2 — Pitcher modifier
    pc = get_pitcher_stats(opp_p_name, 2026)
    pp = get_pitcher_stats(opp_p_name, 2025)
    ip_26 = pc.get("ip", 0); ip_25 = pp.get("ip", 0)
    pwc, pwp = get_pitcher_blend_weights(ip_26, ip_25)

    p_split  = get_pitcher_split(opp_p_name, bat_hand)
    split_ip = p_split.get("ip", 0)

    hr9_season = blend(pc.get("hr9", 0), pp.get("hr9", 0), pwc, pwp)
    hr9_split  = p_split.get("hr9", 0) if split_ip >= 5 else 0

    if hr9_split > 0 and hr9_season > 0:
        split_ratio  = hr9_split / hr9_season
        split_weight = min(split_ip / 30.0, 0.80)
        pit_hr9      = hr9_season * (1 - split_weight) + hr9_split * split_weight
        platoon_magnitude = round((split_ratio - 1.0) * 100, 1)
    else:
        pit_hr9 = hr9_season
        platoon_magnitude = 0.0

    pit_hard = blend(pc.get("hard_hit_pct", 0), pp.get("hard_hit_pct", 0), pwc, pwp)
    pit_brl  = blend(pc.get("barrel_pct_allowed", 0), pp.get("barrel_pct_allowed", 0), pwc, pwp)
    pit_fb   = blend(pc.get("fb_pct", 0), pp.get("fb_pct", 0), pwc, pwp)

    m_hr9  = 1.0 + (pit_hr9 - 1.15) / 1.15 * 0.40 if pit_hr9 > 0 else 1.0
    m_hard = 1.0 + (pit_hard - 32.0) / 32.0 * 0.25 if pit_hard > 0 else 1.0
    m_brl  = 1.0 + (pit_brl - 7.5) / 7.5 * 0.20  if pit_brl > 0 else 1.0
    m_fb   = 1.0 + (pit_fb - 36.0) / 36.0 * 0.15  if pit_fb > 0 else 1.0

    components = [(m_hr9, 0.40, pit_hr9 > 0), (m_hard, 0.25, pit_hard > 0),
                  (m_brl, 0.20, pit_brl > 0), (m_fb, 0.15, pit_fb > 0)]
    active = [(m, w) for m, w, has in components if has]
    if not active:
        pit_modifier = 1.0
    else:
        total_w = sum(w for _, w in active)
        pit_modifier = sum(m * w / total_w for m, w in active)
    pit_modifier = round(max(min(pit_modifier, 1.65), 0.55), 3)
    n_components = len(active)

    # Step 3 — K% gate
    k_cap = 1.0
    if k_s >= 35: k_cap = 0.75
    elif k_s >= 30: k_cap = 0.88
    elif k_s >= 28: k_cap = 0.94
    after_k = batter_score * k_cap

    # Step 4 — Context
    after_context = round(after_k * pit_modifier * park_factor * weather_mult, 1)

    # Step 5 — Sigmoid
    hr_prob = sigmoid_to_prob(after_context)

    platoon_tag = None
    if split_ip >= 5 and hr9_split > 0 and hr9_season > 0:
        split_ratio = hr9_split / hr9_season
        hand_label  = "LHB" if bat_hand == "L" else "RHB"
        if split_ratio >= 1.3:
            platoon_tag = f"SP weak vs {hand_label} ({hr9_split:.1f} HR/9)"
        elif split_ratio <= 0.7:
            platoon_tag = f"SP strong vs {hand_label} ({hr9_split:.1f} HR/9)"

    conf       = "High" if n_components >= 3 and pa_26 >= 50 else "Medium" if n_components >= 2 else "Low"
    blend_note = f"{int(bwc*100)}% 2026/{int(bwp*100)}% 2025" + (" + 30% 8d" if has_8d else "")

    reasons = []
    if barrel_s > 10: reasons.append(f"Barrel {barrel_s:.1f}%")
    if iso_use > 0.200: reasons.append(f"ISO .{int(iso_use*1000):03d}")
    if pit_hr9 > 1.3: reasons.append(f"SP {pit_hr9:.1f} HR/9")
    if pit_hard > 38: reasons.append(f"SP {pit_hard:.1f}% HH")
    if park_factor >= 1.15: reasons.append("HR-friendly park")
    elif park_factor <= 0.90: reasons.append("Pitcher-friendly park")

    breakdown = {
        "barrel_s": round(barrel_s, 1), "s1_barrel": s1_barrel,
        "barrel_season": round(barrel_season, 1), "barrel_8d": round(barrel_8d_raw, 1),
        "iso_use": round(iso_use, 3), "s1_iso": s1_iso, "iso_8d": round(iso_8d, 3),
        "pull_8d": round(b8d.get("pull_pct", 0), 1),
        "la_8d": round(b8d.get("launch_angle", 0), 1),
        "pa_8d": int(b8d.get("pa", 0)),
        "fb_s": round(fb_s, 1), "s1_fb": s1_fb,
        "pull_s": round(pull_s, 1), "s1_pull": s1_pull,
        "la_s": round(la_s, 1), "s1_la": s1_la,
        "hr_fb_s": round(hr_fb_s, 1), "s1_hrfb": s1_hrfb,
        "hr_rate_8d": round(hr_rate_8d, 1), "s1_hr8d": s1_hr8d,
        "has_8d": has_8d, "batter_score": round(batter_score, 1),
        "k_s": round(k_s, 1), "k_cap": k_cap,
        "pit_hr9": round(pit_hr9, 2), "pit_hard": round(pit_hard, 1),
        "pit_brl": round(pit_brl, 1), "pit_modifier": pit_modifier,
        "n_pit_components": n_components, "platoon_magnitude": platoon_magnitude,
        "hr9_split": round(hr9_split, 2), "hr9_season": round(hr9_season, 2),
        "split_ip": round(split_ip, 1),
        "pitch_bonus": pitch_bonus, "pitch_breakdown": pitch_details,
        "after_k": round(after_k, 1), "park_factor": park_factor,
        "weather_mult": weather_mult, "after_context": after_context, "hr_prob": hr_prob,
        "blend_note": blend_note,
        "pit_blend_note": f"{int(pwc*100)}% 2026 / {int(pwp*100)}% 2025 ({ip_26:.0f} IP)",
        # Split stats for dropdown
        "split_brl":  round(b_split.get("barrel_pct", 0), 1),
        "split_iso":  round(b_split.get("iso", 0), 3),
        "split_slg":  round(b_split.get("slg", 0), 3),
        "split_woba": round(b_split.get("woba", 0), 3),
        "split_hr":   int(b_split.get("hr", 0)),
        "split_pa":   int(b_split.get("pa", 0)),
        "hr_season":  int(bc.get("hr", 0)),
        "l8d_hr_reliable": get_l8d_hr(name),
    }
    return hr_prob, breakdown, archetype, trend, reasons, platoon_tag, conf

# ── Weather ──
async def fetch_weather(lat, lon, game_time_utc):
    try:
        hour = 18
        if game_time_utc:
            try:
                dt = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
                hour = dt.hour
            except: pass
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&hourly=temperature_2m,windspeed_10m,winddirection_10m"
               f"&temperature_unit=fahrenheit&windspeed_unit=mph&forecast_days=2&timezone=auto")
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url); d = r.json()
        hourly = d.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        speeds = hourly.get("windspeed_10m", [])
        dirs = hourly.get("winddirection_10m", [])
        idx = min(hour, len(temps) - 1)
        for i, t in enumerate(times):
            if f"T{hour:02d}:" in t: idx = i; break
        return (round(temps[idx]) if idx < len(temps) else 70,
                round(speeds[idx]) if idx < len(speeds) else 0,
                round(dirs[idx]) if idx < len(dirs) else 0)
    except: return 70, 0, 0

async def fetch_player_hand(player_id):
    if player_id in _cache["player_hands"]:
        return _cache["player_hands"][player_id]
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(f"{MLB_API}/people/{player_id}")
            d = r.json()
        person = d.get("people", [{}])[0]
        result = {
            "bat_side": person.get("batSide", {}).get("code", "") or "R",
            "pitch_hand": person.get("pitchHand", {}).get("code", "") or "R",
            "name": person.get("fullName", "")
        }
        _cache["player_hands"][player_id] = result
        return result
    except:
        return {"bat_side": "R", "pitch_hand": "R", "name": ""}

async def fetch_projected_lineup(team_id, team_name):
    try:
        end = date.today(); start = end - timedelta(days=10)
        url = f"{MLB_API}/schedule?sportId=1&teamId={team_id}&startDate={start}&endDate={end}&hydrate=boxscore"
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url); d = r.json()
        recent_games = []
        for de in reversed(d.get("dates", [])):
            for g in de.get("games", []):
                if g.get("status", {}).get("abstractGameState") == "Final":
                    recent_games.append(g["gamePk"])
            if len(recent_games) >= 5: break
        player_data = defaultdict(lambda: {"name": "", "appearances": 0, "orders": [], "id": 0})
        for gid in recent_games[:5]:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(f"{MLB_API}/game/{gid}/boxscore"); box = r.json()
                for side in ["away", "home"]:
                    td = box.get("teams", {}).get(side, {})
                    if team_name.lower() not in td.get("team", {}).get("name", "").lower(): continue
                    for _, p in td.get("players", {}).items():
                        order = p.get("battingOrder")
                        if order and int(order) <= 900:
                            person = p.get("person", {})
                            pid = person.get("id", 0)
                            player_data[pid]["name"] = person.get("fullName", "")
                            player_data[pid]["id"] = pid
                            player_data[pid]["appearances"] += 1
                            player_data[pid]["orders"].append(int(order) // 100)
            except: continue
        projected = [{"id": d["id"], "name": d["name"], "appearances": d["appearances"],
                      "avg_order": sum(d["orders"]) / len(d["orders"])}
                     for d in player_data.values() if d["appearances"] >= 2 and d["name"]]
        projected.sort(key=lambda x: x["avg_order"])
        return projected[:9], "projected"
    except: return [], "projected"

async def fetch_dk_hr_props():
    if not ODDS_API_KEY: return {}
    try:
        url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events?apiKey={ODDS_API_KEY}&dateFormat=iso"
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url)
            if not r.is_success: return {}
            events = r.json()
        props = {}
        for event in events[:15]:
            event_id = event.get("id", "")
            try:
                prop_url = (f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds?"
                            f"apiKey={ODDS_API_KEY}&regions=us&markets=batter_home_runs"
                            f"&oddsFormat=american&bookmakers=betrivers")
                async with httpx.AsyncClient(timeout=10) as client:
                    pr = await client.get(prop_url)
                    if not pr.is_success: continue
                    pd_data = pr.json()
                for bk in pd_data.get("bookmakers", []):
                    if bk.get("key") != "betrivers": continue
                    for mkt in bk.get("markets", []):
                        for outcome in mkt.get("outcomes", []):
                            pname = outcome.get("description") or outcome.get("name", "")
                            price = outcome.get("price", 0)
                            if pname and price: props[pname.lower()] = price
            except: continue
        return props
    except: return {}

async def fetch_pitcher_k_props():
    """Fetch pitcher strikeout prop lines from BetRivers/DraftKings/FanDuel via Odds API"""
    if not ODDS_API_KEY: return {}
    try:
        url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events?apiKey={ODDS_API_KEY}&dateFormat=iso"
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url)
            if not r.is_success: return {}
            events = r.json()
        k_props = {}
        for event in events[:15]:
            event_id = event.get("id", "")
            try:
                prop_url = (f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds?"
                            f"apiKey={ODDS_API_KEY}&regions=us"
                            f"&markets=pitcher_strikeouts,pitcher_outs"
                            f"&oddsFormat=american&bookmakers=betrivers,draftkings,fanduel")
                async with httpx.AsyncClient(timeout=10) as client:
                    pr = await client.get(prop_url)
                    if not pr.is_success: continue
                    pd_data = pr.json()
                for bk in pd_data.get("bookmakers", []):
                    for mkt in bk.get("markets", []):
                        mkt_key = mkt.get("key", "")
                        for outcome in mkt.get("outcomes", []):
                            pname = outcome.get("description") or outcome.get("name", "")
                            line  = outcome.get("point", 0)
                            side  = outcome.get("name", "")
                            price = outcome.get("price", 0)
                            if pname and line and side == "Over":
                                key = pname.lower()
                                if key not in k_props or mkt_key == "pitcher_strikeouts":
                                    k_props[key] = {
                                        "line": line, "price": price,
                                        "market": mkt_key, "book": bk.get("title", ""),
                                    }
            except: continue
        print(f"Pitcher K props fetched: {len(k_props)} pitchers")
        return k_props
    except Exception as e:
        print(f"Pitcher K props error: {e}")
        return {}

def match_pitcher_k_prop(pitcher_name, k_props):
    if not k_props: return None
    nl = pitcher_name.lower()
    if nl in k_props: return k_props[nl]
    last = nl.split()[-1]
    for k, v in k_props.items():
        if last in k: return v
    return None

def match_dk_odds(player_name, props):
    if not props: return None
    nl = player_name.lower()
    if nl in props: return props[nl]
    last = nl.split()[-1]
    for k, v in props.items():
        if last in k: return v
    return None

def fmt_odds(o):
    if o is None: return None
    return f"+{int(o)}" if o > 0 else str(int(o))

def pit_display(p_name, p_hand):
    pc = get_pitcher_stats(p_name, 2026)
    pp = get_pitcher_stats(p_name, 2025)
    ip_26 = pc.get("ip", 0)
    pwc, pwp = get_pitcher_blend_weights(ip_26, pp.get("ip", 0))
    top_pitches = get_pitcher_top_pitches(p_name)
    vs_L = get_pitcher_split(p_name, "L")
    vs_R = get_pitcher_split(p_name, "R")
    nl = p_name.lower().strip()
    ip_data = _cache["player_ip"].get(nl, {})
    if not ip_data:
        last = nl.split()[-1]
        for k, v in _cache["player_ip"].items():
            if last in k: ip_data = v; break
    k9_val  = blend(pc.get("k9", 0), pp.get("k9", 0), pwc, pwp)
    avg_ip  = ip_data.get("avg_ip", 5.0) or 5.0
    gs_val  = ip_data.get("gs", 0)
    return {
        "name": p_name, "hand": p_hand,
        "era": round(blend(pc.get("era", 0), pp.get("era", 0), pwc, pwp), 2) or None,
        "hr9": round(blend(pc.get("hr9", 0), pp.get("hr9", 0), pwc, pwp), 2) or None,
        "hard_hit_pct": round(blend(pc.get("hard_hit_pct", 0), pp.get("hard_hit_pct", 0), pwc, pwp), 1) or None,
        "barrel_pct": round(blend(pc.get("barrel_pct_allowed", 0), pp.get("barrel_pct_allowed", 0), pwc, pwp), 1) or None,
        "ip_2026": round(ip_26, 1),
        "blend_note": f"{int(pwc*100)}% 2026 / {int(pwp*100)}% 2025",
        "vs_L_hr9":  round(vs_L.get("hr9", 0), 2) if vs_L.get("pa", 0) >= 1 else None,
        "vs_R_hr9":  round(vs_R.get("hr9", 0), 2) if vs_R.get("pa", 0) >= 1 else None,
        "vs_L_k":    round(vs_L.get("k_pct", 0), 1) if vs_L.get("pa", 0) >= 1 else None,
        "vs_R_k":    round(vs_R.get("k_pct", 0), 1) if vs_R.get("pa", 0) >= 1 else None,
        "vs_L_slg":  round(vs_L.get("slg", 0), 3) if vs_L.get("pa", 0) >= 1 else None,
        "vs_R_slg":  round(vs_R.get("slg", 0), 3) if vs_R.get("pa", 0) >= 1 else None,
        "vs_L_woba": round(vs_L.get("woba", 0), 3) if vs_L.get("pa", 0) >= 1 else None,
        "vs_R_woba": round(vs_R.get("woba", 0), 3) if vs_R.get("pa", 0) >= 1 else None,
        "top_pitches": [{"name": p["name"], "usage": p["usage"]} for p in top_pitches],
        "k9": round(k9_val, 1) if k9_val > 0 else None,
        "avg_ip": round(avg_ip, 1),
        "gs": gs_val,
    }

# ── API Endpoints ──
@app.get("/history")
async def get_history():
    """Return all historical prediction/result files from GitHub"""
    if not GITHUB_TOKEN:
        return {"error": "GitHub not configured", "records": []}
    import json
    try:
        # List all files in data/predictions/
        url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/data/predictions"
        headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, headers=headers)
        if r.status_code == 404:
            return {"records": [], "dates": [], "message": "No history yet — predictions start saving at 1pm ET daily"}
        if not r.is_success:
            return {"error": f"GitHub error {r.status_code}", "records": []}
        files = r.json()
        all_records = []
        dates = []
        # Fetch each file (most recent 30 days)
        for f in sorted(files, key=lambda x: x["name"], reverse=True)[:30]:
            fname = f["name"]
            if not fname.endswith(".json"): continue
            d = fname.replace(".json", "")
            dates.append(d)
            content, _ = await github_get_file(f"data/predictions/{fname}")
            if not content: continue
            try:
                recs = json.loads(content)
                all_records.extend(recs)
            except Exception: continue
        return {"records": all_records, "dates": dates}
    except Exception as e:
        print(f"History endpoint error: {e}")
        return {"error": str(e), "records": []}

@app.get("/debug-l8d")
async def debug_l8d(player: str = "Murakami"):
    """Debug L8D data — check what Savant returns vs season stats"""
    from datetime import date as date_cls
    b8d = get_batter_8d(player)
    bc = get_batter_stats(player, 2026)
    url = savant_8d_url()
    return {
        "player_searched": player,
        "l8d_url": url,
        "l8d_stats": b8d,
        "season_stats": {
            "pa": bc.get("pa"), "hr": bc.get("hr"),
            "barrel_pct": bc.get("barrel_pct"),
            "launch_angle": bc.get("launch_angle"),
            "k_pct": bc.get("k_pct"),
            "iso": bc.get("iso"),
        },
        "l8d_cache_size": len(_cache["bat_8d"]) if not _cache["bat_8d"].empty else 0,
        "same_as_season": b8d.get("barrel_pct") == bc.get("barrel_pct"),
    }

@app.post("/recalibrate")
async def manual_recalibrate():
    """Manually trigger model recalibration — requires 50+ completed records"""
    result = await recalibrate_model()
    return result

@app.get("/model-weights")
async def get_model_weights():
    """Return current model weights, rotation status, and model log"""
    rotation_start = date(2026, 4, 13)
    days_since_start = (date.today() - rotation_start).days
    days_left = ROTATION_DAYS - get_rotation_day()
    return {
        "weights": _model_weights,
        "rotation": {
            "round": get_rotation_round(),
            "day": get_rotation_day(),
            "days_left": days_left,
            "rotation_start": rotation_start.isoformat(),
            "days_since_start": days_since_start,
        },
        "rotation_schedule": ROTATION_SCHEDULE,
        "league_constants": LEAGUE_CONSTANTS,
    }

@app.post("/save-predictions")
async def manual_save_predictions():
    """Manually trigger saving today's predictions"""
    await save_daily_predictions()
    return {"status": "done", "date": date.today().isoformat()}

@app.post("/record-results")
async def manual_record_results(target_date: str = None):
    """Manually trigger recording results for a date"""
    d = target_date or (date.today() - timedelta(days=1)).isoformat()
    await record_results(d)
    return {"status": "done", "date": d}

@app.get("/")
def root():
    return {
        "status": "Sharp MLB HR Model — Baseball Savant Edition",
        "data_ready": _cache["ready"],
        "last_updated": _cache["last_updated"],
        "rows": {k: len(v) for k, v in _cache.items() if isinstance(v, pd.DataFrame)}
    }

@app.get("/status")
def status():
    return {
        "ready": _cache["ready"],
        "last_updated": _cache["last_updated"],
        "last_8d_update": _cache["last_8d_update"],
        # Batter data
        "bat_2026": len(_cache["bat_2026"]),
        "bat_2025": len(_cache["bat_2025"]),
        "bat_8d": len(_cache["bat_8d"]),
        "bat_l5g": len(_cache["bat_l5g"]),
        "bat_vs_lhp": len(_cache["bat_vs_lhp"]),
        "bat_vs_rhp": len(_cache["bat_vs_rhp"]),
        # Pitcher data
        "pit_2026": len(_cache["pit_2026"]),
        "pit_2025": len(_cache["pit_2025"]),
        "pit_vs_lhh": len(_cache["pit_vs_lhh"]),
        "pit_vs_rhh": len(_cache["pit_vs_rhh"]),
        "pit_arsenal": len(_cache["pit_arsenal"]),
        "bat_arsenal": len(_cache["bat_arsenal"]),
        # New caches
        "bat_l8d_hr": len(_cache.get("bat_l8d_hr", {})),
        "bat_games": len(_cache.get("bat_games", {})),
        "team_hitting": len(_cache.get("team_hitting", {})),
        "team_pitching": len(_cache.get("team_pitching", {})),
        "team_bullpen": len(_cache.get("team_bullpen", {})),
        "player_hands": len(_cache.get("player_hands", {})),
        "player_ip": len(_cache.get("player_ip", {})),
        # Model weights
        "model_calibrated": _model_weights.get("last_calibrated"),
        "model_round": get_rotation_round(),
        "model_day": get_rotation_day(),
    }

@app.post("/reload")
async def reload_data():
    _games_cache.clear()  # invalidate so next /games fetch is fresh
    threading.Thread(target=run_async, args=(load_all_savant_data(),), daemon=True).start()
    return {"status": "Reloading data from Baseball Savant"}

@app.get("/games")
async def get_games(date: str = None, refresh: bool = False):
    if not _cache["ready"]:
        return {"games": [], "loading": True, "message": "Data loading — try again in 30 seconds."}

    from datetime import date as date_cls
    today = date if date else date_cls.today().isoformat()
    date = None  # clear to avoid shadowing

    # ── Response cache — return cached result if < 15 min old and not forced refresh ──
    cached = _games_cache.get(today)
    if cached and not refresh and (datetime.now() - cached["ts"]).total_seconds() < GAMES_CACHE_TTL:
        return cached["data"]

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{MLB_API}/schedule?sportId=1&date={today}&hydrate=team,probablePitcher")
        data = r.json()

    dk_props = await fetch_dk_hr_props()
    k_props  = await fetch_pitcher_k_props()
    dates = data.get("dates", [])
    if not dates: return {"games": [], "date": today, "loading": False}

    # ── Batch-fetch ALL player IDs needed upfront ──
    all_player_ids = set()
    games_list = dates[0].get("games", [])
    for game in games_list:
        if game.get("status", {}).get("abstractGameState") == "Final": continue
        for side in ["away", "home"]:
            pid = game["teams"][side].get("probablePitcher", {}).get("id")
            if pid: all_player_ids.add(pid)

    # Batch fetch all pitcher hands in parallel (batters added below after lineups)
    async def batch_fetch_hands(pids):
        tasks = [fetch_player_hand(pid) for pid in pids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {pid: (r if not isinstance(r, Exception) else {"bat_side": "R", "pitch_hand": "R"})
                for pid, r in zip(pids, results)}

    await batch_fetch_hands(all_player_ids)  # warms the cache for pitchers

    games_out = []
    for game in games_list:
        if game.get("status", {}).get("abstractGameState") == "Final": continue

        gid = game["gamePk"]
        away_team = game["teams"]["away"]["team"]["name"]
        home_team = game["teams"]["home"]["team"]["name"]
        away_team_id = game["teams"]["away"]["team"]["id"]
        home_team_id = game["teams"]["home"]["team"]["id"]
        away_p = game["teams"]["away"].get("probablePitcher", {})
        home_p = game["teams"]["home"].get("probablePitcher", {})
        gtime = game.get("gameDate", "")

        away_p_hand = home_p_hand = "R"
        if away_p.get("id"):
            info = await fetch_player_hand(away_p.get("id"))
            away_p_hand = info.get("pitch_hand", "R")
        if home_p.get("id"):
            info = await fetch_player_hand(home_p.get("id"))
            home_p_hand = info.get("pitch_hand", "R")

        stadium = STADIUMS.get(home_team, {})
        temp, wind_speed, wind_dir = 70, 0, 0
        if not stadium.get("dome") and stadium.get("lat"):
            temp, wind_speed, wind_dir = await fetch_weather(stadium["lat"], stadium["lon"], gtime)
        wx_mult, wx_label = calc_weather_multiplier(home_team, wind_speed, wind_dir, temp)

        lineup_away, lineup_home = [], []
        lineup_away_status = lineup_home_status = "projected"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(f"{MLB_API}/game/{gid}/boxscore"); box = r.json()
            teams = box.get("teams", {})
            def extract(side):
                players = teams.get(side, {}).get("players", {})
                return sorted([p for p in players.values() if p.get("battingOrder") and int(p["battingOrder"]) <= 900],
                              key=lambda x: int(x["battingOrder"]))[:9]
            ca, ch = extract("away"), extract("home")
            if ca: lineup_away = ca; lineup_away_status = "confirmed"
            if ch: lineup_home = ch; lineup_home_status = "confirmed"
        except: pass

        if not lineup_away: lineup_away, _ = await fetch_projected_lineup(away_team_id, away_team)
        if not lineup_home: lineup_home, _ = await fetch_projected_lineup(home_team_id, home_team)

        all_batters = []

        async def process(batter, team, opp_p_name, opp_p_hand, is_proj):
            if "person" in batter:
                name = batter.get("person", {}).get("fullName", "")
                pid = batter.get("person", {}).get("id")
                bat_hand = batter.get("person", {}).get("batSide", {}).get("code", "")
            else:
                name = batter.get("name", ""); pid = batter.get("id"); bat_hand = ""

            if pid:
                info = await fetch_player_hand(pid)
                if info.get("bat_side"): bat_hand = info["bat_side"]
            if not bat_hand: bat_hand = "R"
            if bat_hand == "S": bat_hand = "L" if opp_p_hand == "R" else "R"

            park_factor = get_park_hr_factor(home_team, bat_hand)
            batter_wx_mult, _ = calc_weather_multiplier(home_team, wind_speed, wind_dir, temp, bat_hand)

            hr_prob, breakdown, archetype, trend, reasons, platoon_tag, conf = compute_hr_probability(
                name, bat_hand, opp_p_name, opp_p_hand, park_factor, batter_wx_mult, home_team)

            bc = get_batter_stats(name, 2026)
            bp = get_batter_stats(name, 2025)
            pa_26 = bc.get("pa", 0); pa_25 = bp.get("pa", 0)
            bwc, bwp = get_batter_blend_weights(pa_26, pa_25)
            b8d = get_batter_8d(name)
            bl5g = get_batter_l5g(name)

            all_batters.append({
                "name": name, "team": team, "hr_prob": hr_prob,
                "archetype": archetype, "trend": trend, "confidence": conf,
                "reasons": reasons, "opp_pitcher": opp_p_name,
                "bat_hand": bat_hand, "opp_p_hand": opp_p_hand,
                "park_factor": round(park_factor, 2),
                "l8d_hr_count": get_l8d_hr(name),
                "season": {
                    "barrel": round(blend(bc.get("barrel_pct", 0), bp.get("barrel_pct", 0), bwc, bwp), 1),
                    "ev":     round(blend(bc.get("exit_velo", 0), bp.get("exit_velo", 0), bwc, bwp), 1),
                    "la":     round(blend(bc.get("launch_angle", 0), bp.get("launch_angle", 0), bwc, bwp), 1),
                    "hh":     round(blend(bc.get("hard_hit_pct", 0), bp.get("hard_hit_pct", 0), bwc, bwp), 1),
                    "iso":    round(blend(bc.get("iso", 0), bp.get("iso", 0), bwc, bwp), 3),
                    "slg":    round(blend(bc.get("slg_percent", 0), bp.get("slg_percent", 0), bwc, bwp), 3),
                    "avg":    round(blend(bc.get("batting_avg", 0), bp.get("batting_avg", 0), bwc, bwp), 3),
                    "k":      round(blend(bc.get("k_pct", 0), bp.get("k_pct", 0), bwc, bwp), 1),
                    "pull":   round(blend(bc.get("pull_pct", 0), bp.get("pull_pct", 0), bwc, bwp), 1),
                    "hr":     int(bc.get("hr", 0)),
                },
                "l8d": {
                    "pa":     int(b8d.get("pa", 0)),
                    "barrel": round(b8d.get("barrel_pct", 0), 1),
                    "ev":     round(b8d.get("exit_velo", 0), 1),
                    "la":     round(b8d.get("launch_angle", 0), 1),
                    "hh":     round(b8d.get("hard_hit_pct", 0), 1),
                    "iso":    round(b8d.get("iso", 0), 3),
                    "slg":    round(b8d.get("slg", 0), 3),
                    "avg":    round(b8d.get("avg", 0), 3),
                    "pull":   round(b8d.get("pull_pct", 0), 1),
                    "k_pct":  round(b8d.get("k_pct", 0), 1),
                },
                "l5g": {
                    "ab":  int(bl5g.get("ab", 0)),
                    "hr":  int(bl5g.get("hr", 0)),
                    "slg": round(bl5g.get("slg", 0), 3),
                    "avg": round(bl5g.get("avg", 0), 3),
                    "iso": round(bl5g.get("iso", 0), 3),
                },
                "dk_odds": fmt_odds(match_dk_odds(name, dk_props)),
                "projected": is_proj, "platoon_tag": platoon_tag,
                "breakdown": breakdown,
            })

        # Pre-fetch all batter hands in parallel before processing
        all_lineup = list(lineup_away) + list(lineup_home)
        batter_pids = set()
        for b in all_lineup:
            pid = b.get("person", {}).get("id") if "person" in b else b.get("id")
            if pid: batter_pids.add(pid)
        if batter_pids:
            await batch_fetch_hands(batter_pids)  # warms cache — process() hits cache not network

        # Process all batters in parallel
        away_tasks = [process(b, away_team, home_p.get("fullName", "TBD"), home_p_hand, lineup_away_status == "projected") for b in lineup_away]
        home_tasks = [process(b, home_team, away_p.get("fullName", "TBD"), away_p_hand, lineup_home_status == "projected") for b in lineup_home]
        await asyncio.gather(*away_tasks, *home_tasks)

        away_lineup_ordered = [b for b in all_batters if b["team"] == away_team]
        home_lineup_ordered = [b for b in all_batters if b["team"] == home_team]
        all_batters.sort(key=lambda x: x["hr_prob"], reverse=True)

        # ── Game Totals ──
        park_factor_neutral = 1.0  # neutral for runs (park factors are HR-specific)
        away_lineup_hr_sum  = round(sum(b["hr_prob"] for b in away_lineup_ordered) / 100, 3)
        home_lineup_hr_sum  = round(sum(b["hr_prob"] for b in home_lineup_ordered) / 100, 3)
        away_th = _cache["team_hitting"].get(away_team, {})
        home_th = _cache["team_hitting"].get(home_team, {})
        away_tp = _cache["team_pitching"].get(away_team, {})
        home_tp = _cache["team_pitching"].get(home_team, {})

        # Expected runs: blend team runs/g with starter ERA signal
        away_pit_stats = get_pitcher_stats(away_p.get("fullName", "TBD"), 2026)
        home_pit_stats = get_pitcher_stats(home_p.get("fullName", "TBD"), 2026)
        lg_era = 4.20
        away_starter_factor = 1 + (away_pit_stats.get("era", lg_era) - lg_era) / lg_era * 0.3 if away_pit_stats.get("era") else 1.0
        home_starter_factor = 1 + (home_pit_stats.get("era", lg_era) - lg_era) / lg_era * 0.3 if home_pit_stats.get("era") else 1.0
        away_runs_exp = round((home_th.get("runs_per_g", 4.5) * home_starter_factor * wx_mult), 2)
        home_runs_exp = round((away_th.get("runs_per_g", 4.5) * away_starter_factor * wx_mult), 2)
        total_runs_exp = round(away_runs_exp + home_runs_exp, 2)

        # ── Strikeouts + K Props ──
        away_lineup_k = round(sum(b["season"].get("k", 0) for b in away_lineup_ordered) / max(len(away_lineup_ordered), 1), 1)
        home_lineup_k = round(sum(b["season"].get("k", 0) for b in home_lineup_ordered) / max(len(home_lineup_ordered), 1), 1)
        away_pit_k9  = away_pit_stats.get("k9", 0)
        home_pit_k9  = home_pit_stats.get("k9", 0)
        away_avg_ip  = away_pit_stats.get("avg_ip", 5.0) or 5.0
        home_avg_ip  = home_pit_stats.get("avg_ip", 5.0) or 5.0
        lg_k_pct = 22.5
        away_exp_k = round(away_pit_k9 * (away_avg_ip / 9) * (home_lineup_k / lg_k_pct), 1) if away_pit_k9 > 0 else 0
        home_exp_k = round(home_pit_k9 * (home_avg_ip / 9) * (away_lineup_k / lg_k_pct), 1) if home_pit_k9 > 0 else 0

        # K prop lines from Odds API
        away_k_prop = match_pitcher_k_prop(away_p.get("fullName", "TBD"), k_props)
        home_k_prop = match_pitcher_k_prop(home_p.get("fullName", "TBD"), k_props)
        away_k_edge = round(away_exp_k - away_k_prop["line"], 1) if away_k_prop and away_exp_k > 0 else None
        home_k_edge = round(home_exp_k - home_k_prop["line"], 1) if home_k_prop and home_exp_k > 0 else None

        # Build pitcher display objects with K data attached
        away_pit_obj = pit_display(away_p.get("fullName", "TBD"), away_p_hand)
        home_pit_obj = pit_display(home_p.get("fullName", "TBD"), home_p_hand)
        for obj, exp_k, k_prop, k_edge, opp_t, opp_lk in [
            (away_pit_obj, away_exp_k, away_k_prop, away_k_edge, home_team, home_lineup_k),
            (home_pit_obj, home_exp_k, home_k_prop, home_k_edge, away_team, away_lineup_k),
        ]:
            obj["exp_k"]       = exp_k
            obj["k_prop"]      = k_prop
            obj["k_edge"]      = k_edge
            obj["opp_team"]    = opp_t
            obj["opp_lineup_k"] = opp_lk

        games_out.append({
            "game_id": gid, "away": away_team, "home": home_team, "time": gtime,
            "away_pitcher": away_pit_obj,
            "home_pitcher": home_pit_obj,
            "top_hr_candidates": all_batters,
            "away_lineup": away_lineup_ordered,
            "home_lineup": home_lineup_ordered,
            "lineup_away_status": lineup_away_status,
            "lineup_home_status": lineup_home_status,
            "weather": {"label": wx_label, "temp": temp, "wind_speed": wind_speed, "wind_dir": wind_dir},
            "totals": {
                "away_exp_hr":    away_lineup_hr_sum,
                "home_exp_hr":    home_lineup_hr_sum,
                "total_exp_hr":   round(away_lineup_hr_sum + home_lineup_hr_sum, 2),
                "away_exp_runs":  away_runs_exp,
                "home_exp_runs":  home_runs_exp,
                "total_exp_runs": total_runs_exp,
                "away_runs_pg":   away_th.get("runs_per_g", 0),
                "home_runs_pg":   home_th.get("runs_per_g", 0),
                "away_hr_pg":     away_th.get("hr_per_g", 0),
                "home_hr_pg":     home_th.get("hr_per_g", 0),
                "away_era":       away_tp.get("era", 0),
                "home_era":       home_tp.get("era", 0),
                "away_k_pg":      away_tp.get("k_per_9", 0),
                "home_k_pg":      home_tp.get("k_per_9", 0),
            },
            "strikeouts": {
                "away_exp_k":    away_exp_k,
                "home_exp_k":    home_exp_k,
                "away_lineup_k": away_lineup_k,
                "home_lineup_k": home_lineup_k,
                "away_pit_name": away_p.get("fullName", "TBD"),
                "home_pit_name": home_p.get("fullName", "TBD"),
                "away_pit_k9":   round(away_pit_k9, 1),
                "home_pit_k9":   round(home_pit_k9, 1),
                "away_avg_ip":   round(away_avg_ip, 1),
                "home_avg_ip":   round(home_avg_ip, 1),
                "away_k_prop":   away_k_prop,
                "home_k_prop":   home_k_prop,
                "away_k_edge":   away_k_edge,
                "home_k_edge":   home_k_edge,
            },
        })

    result = {"games": games_out, "date": today, "loading": False}
    _games_cache[today] = {"data": result, "ts": datetime.now()}
    return result

@app.get("/research")
async def research(player: str, date: str = None):
    from datetime import date as date_cls
    today = date if date else date_cls.today().isoformat()
    date = None  # clear to avoid shadowing
    if not _cache["ready"]:
        return {"error": "Data loading — try again in 30 seconds"}

    bc = get_batter_stats(player, 2026)
    bp = get_batter_stats(player, 2025)
    b8d = get_batter_8d(player)
    bl5g = get_batter_l5g(player)
    pa_26 = bc.get("pa", 0); pa_25 = bp.get("pa", 0)
    bwc, bwp = get_batter_blend_weights(pa_26, pa_25)
    has_8d = b8d.get("pa", 0) >= 3
    w_s, w_8 = (0.70, 0.30) if has_8d else (1.0, 0.0)

    def blend3(s26, s25, d8):
        s = blend(s26, s25, bwc, bwp)
        return round(s * w_s + d8 * w_8, 3) if (has_8d and d8 > 0) else round(s, 3)

    stats = {
        "name": player, "pa_2026": pa_26, "pa_2025": pa_25,
        "blend_note": f"{int(bwc*100)}% 2026 / {int(bwp*100)}% 2025",
        "season_2026": bc,
        "last_8d": b8d,
        "last_5g": bl5g,
        "blended": {
            "barrel_pct":   blend3(bc.get("barrel_pct", 0), bp.get("barrel_pct", 0), b8d.get("barrel_pct", 0)),
            "iso":          blend3(bc.get("iso", 0), bp.get("iso", 0), b8d.get("iso", 0)),
            "pull_pct":     blend3(bc.get("pull_pct", 0), bp.get("pull_pct", 0), b8d.get("pull_pct", 0)),
            "launch_angle": blend3(bc.get("launch_angle", 0), bp.get("launch_angle", 0), b8d.get("launch_angle", 0)),
            "hard_hit_pct": blend3(bc.get("hard_hit_pct", 0), bp.get("hard_hit_pct", 0), b8d.get("hard_hit_pct", 0)),
        },
        "splits": {
            "vs_lhp": get_batter_split(player, "L"),
            "vs_rhp": get_batter_split(player, "R"),
        },
        "pitch_values": {code: get_batter_pitch_rv(player, code) for code in ["wfa","wsl","wfc","wch","wcu","wsi","wfs"]},
    }

    matchup = None
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{MLB_API}/schedule?sportId=1&date={today}&hydrate=team,probablePitcher")
            schedule = r.json()
        for game_date in schedule.get("dates", []):
            for game in game_date.get("games", []):
                for side in ["away", "home"]:
                    opp_side = "home" if side == "away" else "away"
                    opp_p = game["teams"][opp_side].get("probablePitcher", {})
                    opp_p_name = opp_p.get("fullName", "TBD")
                    opp_p_id = opp_p.get("id")
                    opp_p_hand = "R"
                    if opp_p_id:
                        info = await fetch_player_hand(opp_p_id)
                        opp_p_hand = info.get("pitch_hand", "R")
                    pc = get_pitcher_stats(opp_p_name, 2026)
                    pp2 = get_pitcher_stats(opp_p_name, 2025)
                    ip_26 = pc.get("ip", 0)
                    pwc, pwp = get_pitcher_blend_weights(ip_26, pp2.get("ip", 0))
                    pitch_bonus, pitch_details = compute_pitch_matchup(opp_p_name, player)
                    matchup = {
                        "home_team": game["teams"]["home"]["team"]["name"],
                        "away_team": game["teams"]["away"]["team"]["name"],
                        "game_time": game.get("gameDate", ""),
                        "pitcher_name": opp_p_name,
                        "pitcher_hand": opp_p_hand,
                        "pitcher_stats": {
                            "era": round(blend(pc.get("era", 0), pp2.get("era", 0), pwc, pwp), 2),
                            "hr9": round(blend(pc.get("hr9", 0), pp2.get("hr9", 0), pwc, pwp), 2),
                            "hard_hit_pct": round(blend(pc.get("hard_hit_pct", 0), pp2.get("hard_hit_pct", 0), pwc, pwp), 1),
                            "barrel_pct_allowed": round(blend(pc.get("barrel_pct_allowed", 0), pp2.get("barrel_pct_allowed", 0), pwc, pwp), 1),
                            "ip_2026": ip_26,
                            "blend_note": f"{int(pwc*100)}% 2026 / {int(pwp*100)}% 2025",
                        },
                        "pitch_matchup": pitch_details,
                        "pitch_bonus": pitch_bonus,
                    }
                    break
            if matchup: break
    except Exception as e:
        print(f"Research matchup error: {e}")

    return {"player": stats, "matchup": matchup, "date": today}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
