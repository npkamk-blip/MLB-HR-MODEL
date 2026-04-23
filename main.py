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
    "lg_hr_per_pa":    0.028,  # league avg HR/PA (~10% per game at 3.8 PA/game)
    "lg_era":          4.20,   # league avg ERA
    "max_hr_per_pa":   0.12,   # ceiling on any batter base rate
    "hr_prob_cap":     28.0,   # hard cap — prevents runaway multiplier stacking
}

# ── Model Weights (learned from ML, updated every 45 days) ──
# All start at 1.0 — neutral. Recalibration moves them based on what predicted HRs.
# Exponent weights: effective_mult = raw_mult ** weight
# weight > 1.0 = stat matters MORE than assumed
# weight < 1.0 = stat matters LESS than assumed
DEFAULT_WEIGHTS = {
    # Batter contact quality — season
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
    # Batter L8D advanced
    "xwoba_l8d_w":        1.0,
    "xslg_l8d_w":         1.0,
    "bat_speed_l8d_w":    1.0,
    "slg_l8d_w":          1.0,
    "xslg_gap_l8d_w":     1.0,
    "l8d_hr_w":           1.0,
    # Batter other
    "fb_pct_season_w":    1.0,
    "pull_pct_season_w":  1.0,
    "k_pct_l8d_w":        1.0,
    "hot_cold_mult_w":    1.0,
    # Pitcher vulnerability
    "pit_hr9_season_w":   1.0,
    "pit_hr9_vs_hand_w":  1.0,
    "pit_slg_season_w":   1.0,
    "pit_slg_vs_hand_w":  1.0,
    "pit_hard_hit_w":     1.0,
    "pit_fb_pct_w":       1.0,
    "pit_era_diff_w":     1.0,
    "pit_k9_season_w":    1.0,
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
    # New stats
    "chase_rate_w":       1.0,   # batter chase rate — lower = better discipline
    "pit_stuff_plus_w":   1.0,   # pitcher Stuff+ — higher = harder to hit
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

# ── Decision Tree model storage ──
_dt_model = None        # trained sklearn DecisionTreeClassifier
_dt_features = []       # ordered feature list the tree was trained on
_dt_medians = {}        # per-feature medians for missing value imputation

TREE_STATS = [
    # Batter season
    "barrel_pct_season", "hard_hit_season", "ev_season", "iso_season",
    "la_season", "pull_pct_season",
    # Batter L8D
    "barrel_pct_l8d", "hard_hit_l8d", "ev_l8d", "bat_speed_l8d",
    "xwoba_l8d", "xslg_l8d", "slg_l8d", "iso_l8d", "k_pct_l8d", "la_l8d",
    # Batter splits
    "iso_vs_hand", "slg_vs_hand",
    # Pitcher season
    "pit_hr9_season", "pit_era_season", "pit_hard_hit_season", "pit_k9_season",
    # Pitcher vs hand
    "pit_hr9_vs_hand", "pit_slg_vs_hand",
    # Context
    "park_factor", "weather_mult", "bullpen_hr9",
    "bat_platoon_mult", "pit_platoon_mult",
    "pitch_matchup_score",
]

# Stats where zero = genuinely dominant (not missing) IF companion IP > 0
ZERO_DOMINANT = {
    "pit_hr9_season":  "pit_ip_season",
    "pit_hr9_vs_hand": "pit_ip_vs_hand",
    "pit_slg_vs_hand": "pit_ip_vs_hand",
}

def get_tree_depth(n_records):
    """Auto-scale tree depth based on record count."""
    if n_records < 1500:  return 4, 30
    if n_records < 2500:  return 5, 20
    if n_records < 5000:  return 6, 15
    if n_records < 10000: return 7, 10
    return 8, 5

def get_calibration_factor(dataset_hr_rate, n_records, true_hr_rate=2.8):
    """
    Calibration corrects for biased training population early on.
    
    Removal strategy (Option 2 — let data fix itself):
    - Below 3000 records: dataset is still filling in, apply calibration
    - 3000+ records: dataset HR rate has stabilized and reflects true
      deployment population — tree outputs are already correct, factor = 1.0
    
    Between 1500-3000: linear blend from calibrated to raw
    so the transition is smooth not a sudden jump.
    """
    # Above 3000 records — tree trained on stable population, no correction needed
    if n_records >= 3000:
        return 1.0

    # Below 1500 records — full calibration
    if n_records < 1500:
        if dataset_hr_rate <= 0:
            return 1.0
        return true_hr_rate / dataset_hr_rate

    # 1500-3000 records — linear blend toward 1.0
    # At 1500: full calibration. At 3000: factor = 1.0
    blend = (n_records - 1500) / (3000 - 1500)  # 0.0 at 1500, 1.0 at 3000
    full_factor = true_hr_rate / dataset_hr_rate if dataset_hr_rate > 0 else 1.0
    return full_factor + blend * (1.0 - full_factor)

def clean_feature_value(rec, stat, medians, pit_ip_season=0, pit_ip_vs_hand=0):
    """Return clean float value for a stat from a record, using medians for missing."""
    v = rec.get(stat)
    # Handle zero-means-dominant stats
    if stat in ZERO_DOMINANT:
        ip_field = ZERO_DOMINANT[stat]
        ip = rec.get(ip_field, 0) or 0
        min_ip = 5 if "vs_hand" in stat else 1
        if ip >= min_ip:
            # Real data — zero IS a signal (dominant pitcher)
            return float(v) if v is not None else 0.0
        else:
            # No IP = no data — use season HR9 as fallback
            fallback = rec.get("pit_hr9_season", medians.get(stat, 1.1))
            return float(fallback) if fallback else medians.get(stat, 1.1)
    # Regular stats — zero means missing
    if v is None or v == 0:
        return medians.get(stat, 0.0)
    return float(v)

async def recalibrate_model(save_to_github: bool = True):
    """
    Train Decision Tree on all completed prediction records.
    Tree depth scales automatically with record count.
    Missing values filled with per-stat medians from real data.
    save_to_github=False on startup to prevent deploy loop.
    """
    global _model_weights, _dt_model, _dt_features, _dt_medians
    import json

    # ── Load all completed records ──
    all_records = []
    try:
        if not GITHUB_TOKEN:
            return {"error": "No GitHub token"}
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(
                f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/data/predictions",
                headers={"Authorization": f"token {GITHUB_TOKEN}"}
            )
            if not r.is_success:
                return {"error": f"GitHub API error: {r.status_code}"}
            files = r.json()
        for f in files:
            if not isinstance(f, dict): continue
            if not f.get("name", "").endswith(".json"): continue
            content, _ = await github_get_file(f"data/predictions/{f['name']}")
            if content:
                try:
                    recs = json.loads(content)
                    all_records.extend(recs)
                except: pass
    except Exception as e:
        print(f"Recalibration data load error: {e}")
        return {"error": str(e)}

    completed = [r for r in all_records if r.get("hit_hr") in [0, 1]]
    n = len(completed)
    if n < 50:
        return {"error": f"Not enough data — need 50+, have {n}"}

    print(f"Decision Tree training on {n} records")

    # ── Compute per-stat medians from real non-zero values ──
    medians = {}
    for stat in TREE_STATS:
        if stat in ZERO_DOMINANT:
            # For dominant stats, median of all records with real IP
            ip_field = ZERO_DOMINANT[stat]
            min_ip = 5 if "vs_hand" in stat else 1
            real = [float(r[stat]) for r in completed
                    if r.get(stat) is not None
                    and (r.get(ip_field, 0) or 0) >= min_ip]
        else:
            real = [float(r[stat]) for r in completed
                    if r.get(stat) is not None and r.get(stat) != 0]
        if real:
            real.sort()
            medians[stat] = real[len(real) // 2]
        else:
            medians[stat] = 0.0

    # ── Build feature matrix ──
    features = TREE_STATS  # all 30 stats
    X_rows = []
    y_vals = []
    for rec in completed:
        pit_ip_s  = rec.get("pit_ip_season", 0) or 0
        pit_ip_vh = rec.get("pit_ip_vs_hand", 0) or 0
        row = [clean_feature_value(rec, stat, medians, pit_ip_s, pit_ip_vh)
               for stat in features]
        X_rows.append(row)
        y_vals.append(int(rec["hit_hr"]))

    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    X = np.array(X_rows)
    y = np.array(y_vals)

    depth, min_leaf = get_tree_depth(n)
    print(f"Tree params: max_depth={depth}, min_samples_leaf={min_leaf}")

    tree = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=min_leaf,
        max_features="sqrt",       # √30 ≈ 5-6 features per split — prevents one stat dominating
        class_weight="balanced",   # corrects HR/non-HR imbalance
        random_state=42
    )
    tree.fit(X, y)

    # ── Extract feature importance ──
    importances = {
        features[i]: round(float(tree.feature_importances_[i]), 4)
        for i in range(len(features))
        if tree.feature_importances_[i] > 0
    }
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    # ── Extract tree structure for display ──
    tree_info = tree.tree_
    n_leaves = tree_info.n_node_samples[tree_info.children_left == -1].shape[0]
    hr_rates_at_leaves = []
    for i in range(tree_info.node_count):
        if tree_info.children_left[i] == -1:  # leaf
            vals = tree_info.value[i][0]
            total = sum(vals)
            hr_rate = round(vals[1] / total * 100, 1) if total > 0 else 0
            hr_rates_at_leaves.append(hr_rate)

    # ── Compute calibration factor ──
    dataset_hr_rate = round(sum(y_vals) / len(y_vals) * 100, 1)
    cal_factor = get_calibration_factor(dataset_hr_rate, n)
    print(f"Dataset HR rate: {dataset_hr_rate}% → calibration factor: {cal_factor:.3f}")

    # ── Store tree globally ──
    _dt_model = tree
    _dt_features = features
    _dt_medians = medians

    # ── Save metadata to GitHub ──
    new_weights = {
        **_model_weights,
        "calibration_round": _model_weights.get("calibration_round", 0) + 1,
        "last_calibrated": date.today().isoformat(),
        "records_used": n,
        "model_type": "decision_tree",
        "tree_depth": depth,
        "tree_min_leaf": min_leaf,
        "tree_n_leaves": n_leaves,
        "feature_importances": importances,
        "feature_medians": {k: round(v, 4) for k, v in medians.items()},
        "active_stats": list(importances.keys())[:10],
        "hr_rates_at_leaves": sorted(hr_rates_at_leaves),
        "tree_hr_rate": dataset_hr_rate,
        "calibration_factor": round(cal_factor, 4),
        "next_depth_upgrade": (
            1500 if n < 1500 else
            2500 if n < 2500 else
            5000 if n < 5000 else
            10000 if n < 10000 else "random_forest"
        ),
    }
    _model_weights = new_weights
    if save_to_github:
        await save_model_weights(new_weights)
        await save_model_log(new_weights)
    else:
        print("Startup tree: skipping GitHub save to prevent deploy loop")

    top_features = list(importances.items())[:8]
    print(f"Tree trained — depth={depth}, {n_leaves} leaves")
    print(f"Top features: {top_features}")

    return {
        "status": "done",
        "records_used": n,
        "model_type": "decision_tree",
        "tree_depth": depth,
        "min_samples_leaf": min_leaf,
        "n_leaves": n_leaves,
        "top_features": top_features,
        "leaf_hr_rates": sorted(hr_rates_at_leaves),
        "feature_importances": importances,
    }


def get_rotation_round():
    """Rotation system removed — returns calibration round from weights instead"""
    return int(_model_weights.get("calibration_round", 1))

def get_rotation_day():
    """Rotation system removed — returns 0"""
    return 0

ROTATION_DAYS = 0
ROTATION_SCHEDULE = {}

SAVANT_BASE = "https://baseballsavant.mlb.com"

def savant_batter_url(year=None, min_pa=10, extra=""):
    yr = year or current_season()
    return (f"{SAVANT_BASE}/leaderboard/custom?year={yr}&type=batter&filter=&sort=4"
            f"&sortDir=desc&min={min_pa}&selections=pa,ab,hit,home_run,strikeout,"
            f"k_percent,slg_percent,batting_avg,barrel_batted_rate,exit_velocity_avg,"
            f"launch_angle_avg,hard_hit_percent,pull_percent,n_fb_percent,"
            f"oz_swing_percent{extra}&csv=true")
            # oz_swing_percent = chase rate (swing% on pitches outside zone)

def savant_pitcher_url(year=None, min_pa=5, extra=""):
    yr = year or current_season()
    return (f"{SAVANT_BASE}/leaderboard/custom?year={yr}&type=pitcher&filter=&sort=4"
            f"&sortDir=desc&min={min_pa}&selections=pa,home_run,barrel_batted_rate,"
            f"exit_velocity_avg,hard_hit_percent,k_percent,p_era,n_fb_percent,"
            f"p_stuff_plus{extra}&csv=true")
            # p_stuff_plus = Stuff+ composite pitch quality metric

def savant_pitch_arsenal_url(ptype="pitcher", year=None, min_pa=1):
    yr = year or current_season()
    return (f"{SAVANT_BASE}/leaderboard/pitch-arsenal-stats?type={ptype}"
            f"&pitchType=&year={yr}&team=&min={min_pa}&csv=true")

def current_season():
    """Return current MLB season year"""
    today = date.today()
    # MLB season runs roughly March-November
    return today.year if today.month >= 3 else today.year - 1

def savant_contact_log_url():
    """Pitch-by-pitch contact events — last 3 days only to keep size small and avoid timeout.
    group_by=pitch returns individual rows needed for the contact log table."""
    cutoff = (date.today() - timedelta(days=3)).isoformat()
    today_str = (date.today() + timedelta(days=1)).isoformat()
    return (f"{SAVANT_BASE}/statcast_search/csv?all=true"
            f"&hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull="
            f"&hfC=&hfSea={current_season()}%7C&hfSit=&player_type=batter&hfOuts=&hfOpponent="
            f"&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={cutoff}"
            f"&game_date_lt={today_str}&hfMon=&hfInfield=&team=&position=&hfRO="
            f"&home_road=&hfFlag=&metric_1=&hfInn=&min_pitches=0&min_results=0"
            f"&group_by=pitch&sort_col=game_date&player_event_sort=api_p_release_speed"
            f"&sort_order=desc&min_pas=0&type=details")

def savant_8d_url():
    """Statcast pitch-by-pitch search — reliable date filtering, includes bat speed + xStats.
    We aggregate this ourselves rather than relying on the broken leaderboard date filter."""
    cutoff = (date.today() - timedelta(days=8)).isoformat()
    today_str = (date.today() + timedelta(days=1)).isoformat()  # tomorrow to include today's games
    return (f"{SAVANT_BASE}/statcast_search/csv?all=true"
            f"&hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull="
            f"&hfC=&hfSea={current_season()}%7C&hfSit=&player_type=batter&hfOuts=&hfOpponent="
            f"&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={cutoff}"
            f"&game_date_lt={today_str}&hfMon=&hfInfield=&team=&position=&hfRO="
            f"&home_road=&hfFlag=&metric_1=&hfInn=&min_pitches=0&min_results=0"
            f"&group_by=name&sort_col=xwoba&player_event_sort=api_p_release_speed"
            f"&sort_order=desc&min_pas=0&type=details")

_cache = {
    "bat_2026":     pd.DataFrame(),
    "bat_8d":       pd.DataFrame(),
    "bat_l5g":      {},
    "bat_l8d_hr":   {},
    "bat_games":    {},
    "bat_vs_lhp":   pd.DataFrame(),
    "bat_vs_rhp":   pd.DataFrame(),
    "pit_2026":     pd.DataFrame(),
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
_contact_log = {}   # { player_name_lower: [ {date, pitch_type, ev, la, dist, bat_speed, result}, ... ] }
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
    # hr_bearing_R = direction RHB pull HRs go (to LF, ~NW for most parks)
    # hr_bearing_L = direction LHB pull HRs go (to RF, ~SE for most parks)
    # open_factor = how much wind affects the park (1.0 = fully open like Wrigley)
    "Arizona Diamondbacks":  {"lat":33.4453,"lon":-112.0667,"dome":True},
    "Atlanta Braves":        {"lat":33.8907,"lon":-84.4677,"dome":False,"hr_bearing_R":300,"hr_bearing_L":130,"open_factor":0.5},
    "Baltimore Orioles":     {"lat":39.2838,"lon":-76.6217,"dome":False,"hr_bearing_R":310,"hr_bearing_L":140,"open_factor":0.6},
    "Boston Red Sox":        {"lat":42.3467,"lon":-71.0972,"dome":False,"hr_bearing_R":290,"hr_bearing_L":120,"open_factor":0.7},
    "Chicago Cubs":          {"lat":41.9484,"lon":-87.6553,"dome":False,"hr_bearing_R":305,"hr_bearing_L":135,"open_factor":1.0},
    "Chicago White Sox":     {"lat":41.8299,"lon":-87.6338,"dome":False,"hr_bearing_R":320,"hr_bearing_L":150,"open_factor":0.5},
    "Cincinnati Reds":       {"lat":39.0979,"lon":-84.5082,"dome":False,"hr_bearing_R":300,"hr_bearing_L":130,"open_factor":0.6},
    "Cleveland Guardians":   {"lat":41.4954,"lon":-81.6854,"dome":False,"hr_bearing_R":295,"hr_bearing_L":125,"open_factor":0.6},
    "Colorado Rockies":      {"lat":39.7559,"lon":-104.9942,"dome":False,"hr_bearing_R":310,"hr_bearing_L":140,"open_factor":0.7},
    "Detroit Tigers":        {"lat":42.3390,"lon":-83.0485,"dome":False,"hr_bearing_R":280,"hr_bearing_L":110,"open_factor":0.5},
    "Houston Astros":        {"lat":29.7573,"lon":-95.3555,"dome":True},
    "Kansas City Royals":    {"lat":39.0517,"lon":-94.4803,"dome":False,"hr_bearing_R":315,"hr_bearing_L":145,"open_factor":0.6},
    "Los Angeles Angels":    {"lat":33.8003,"lon":-117.8827,"dome":False,"hr_bearing_R":300,"hr_bearing_L":130,"open_factor":0.5},
    "Los Angeles Dodgers":   {"lat":34.0739,"lon":-118.2400,"dome":False,"hr_bearing_R":315,"hr_bearing_L":145,"open_factor":0.5},
    "Miami Marlins":         {"lat":25.7781,"lon":-80.2197,"dome":True},
    "Milwaukee Brewers":     {"lat":43.0282,"lon":-87.9712,"dome":True},
    "Minnesota Twins":       {"lat":44.9817,"lon":-93.2778,"dome":False,"hr_bearing_R":300,"hr_bearing_L":130,"open_factor":0.6},
    "New York Mets":         {"lat":40.7571,"lon":-73.8458,"dome":False,"hr_bearing_R":310,"hr_bearing_L":140,"open_factor":0.5},
    "New York Yankees":      {"lat":40.8296,"lon":-73.9262,"dome":False,"hr_bearing_R":290,"hr_bearing_L":120,"open_factor":0.6},
    "Oakland Athletics":     {"lat":38.5726,"lon":-121.5088,"dome":False,"hr_bearing_R":305,"hr_bearing_L":135,"open_factor":0.5},
    "Philadelphia Phillies": {"lat":39.9056,"lon":-75.1665,"dome":False,"hr_bearing_R":300,"hr_bearing_L":130,"open_factor":0.5},
    "Pittsburgh Pirates":    {"lat":40.4469,"lon":-80.0057,"dome":False,"hr_bearing_R":310,"hr_bearing_L":140,"open_factor":0.6},
    "San Diego Padres":      {"lat":32.7076,"lon":-117.1570,"dome":False,"hr_bearing_R":305,"hr_bearing_L":135,"open_factor":0.8},
    "San Francisco Giants":  {"lat":37.7786,"lon":-122.3893,"dome":False,"hr_bearing_R":320,"hr_bearing_L":150,"open_factor":0.9},
    "Seattle Mariners":      {"lat":47.5914,"lon":-122.3325,"dome":True},
    "St. Louis Cardinals":   {"lat":38.6226,"lon":-90.1928,"dome":False,"hr_bearing_R":295,"hr_bearing_L":125,"open_factor":0.5},
    "Tampa Bay Rays":        {"lat":27.7683,"lon":-82.6534,"dome":True},
    "Texas Rangers":         {"lat":32.7473,"lon":-97.0825,"dome":True},
    "Toronto Blue Jays":     {"lat":43.6414,"lon":-79.3894,"dome":True},
    "Washington Nationals":  {"lat":38.8730,"lon":-77.0074,"dome":False,"hr_bearing_R":300,"hr_bearing_L":130,"open_factor":0.5},
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
async def fetch_savant_csv(url: str, session: httpx.AsyncClient, timeout: int = 120) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        r = await session.get(url, headers=headers, timeout=timeout, follow_redirects=True)
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
        'oz_swing_percent': 'chase_rate',  # swing% on pitches outside zone
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
        'p_stuff_plus': 'stuff_plus',  # composite pitch quality — 100 = avg, higher = better
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df

def calc_statcast_8d(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch-by-pitch Statcast CSV into per-player L8D stats.
    Groups by batter name and computes: barrel%, avg EV, avg LA, hard hit%, 
    bat speed, xwOBA, xSLG, pull%, PA, HR, K%, ISO, SLG, AVG."""
    df = df.copy()
    # Normalize name: "Last, First" -> "First Last"
    if 'player_name' in df.columns:
        df['name'] = df['player_name'].apply(lambda x: reverse_name(str(x)) if pd.notna(x) else "")
    else:
        return pd.DataFrame()

    results = []
    for name, grp in df.groupby('name'):
        if not name: continue
        # All pitches for K% and PA
        pa_events = grp[grp['events'].notna() & (grp['events'] != '')]
        pa = len(pa_events)
        if pa < 1: continue

        hr  = len(pa_events[pa_events['events'] == 'home_run'])
        so  = len(pa_events[pa_events['events'].isin(['strikeout','strikeout_double_play'])])
        k_pct = round(so / pa * 100, 1) if pa > 0 else 0.0

        # Contact events only (for Statcast metrics)
        contact = grp[grp['launch_speed'].notna() & (grp['launch_speed'] > 0)]
        n_contact = len(contact)

        avg_ev   = round(contact['launch_speed'].mean(), 1) if n_contact > 0 else 0.0
        avg_la   = round(contact['launch_angle'].mean(), 1) if n_contact > 0 else 0.0
        hard_hit = round(len(contact[contact['launch_speed'] >= 95]) / n_contact * 100, 1) if n_contact > 0 else 0.0
        barrels  = len(contact[contact['launch_speed_angle'] == 6]) if 'launch_speed_angle' in contact.columns else 0
        barrel_pct = round(barrels / n_contact * 100, 1) if n_contact > 0 else 0.0

        # Bat speed (all swings)
        swings = grp[grp['bat_speed'].notna() & (grp['bat_speed'] > 0)]
        avg_bat_speed = round(swings['bat_speed'].mean(), 1) if len(swings) > 0 else 0.0

        # Expected stats
        xwoba = round(contact['estimated_woba_using_speedangle'].dropna().mean(), 3) if n_contact > 0 else 0.0
        xslg  = round(contact['estimated_slg_using_speedangle'].dropna().mean(), 3) if n_contact > 0 else 0.0

        # Pull%
        pull_events = contact[contact['hc_x'].notna()]
        if len(pull_events) > 0:
            stand = grp['stand'].iloc[0] if 'stand' in grp.columns else 'R'
            if stand == 'L':
                pulls = len(pull_events[pull_events['hc_x'] > 170])
            else:
                pulls = len(pull_events[pull_events['hc_x'] < 100])
            pull_pct = round(pulls / len(pull_events) * 100, 1)
        else:
            pull_pct = 0.0

        # Traditional stats from woba/iso values
        woba_vals = pa_events['woba_value'].dropna()
        iso_vals  = pa_events['iso_value'].dropna()
        avg_iso   = round(iso_vals.mean(), 3) if len(iso_vals) > 0 else 0.0
        # Approximate SLG from xSLG since actual SLG isn't directly in the CSV
        slg = xslg  # use xSLG as proxy — available for all contact
        avg_val = round(len(pa_events[pa_events['events'].isin(['single','double','triple','home_run'])]) / pa, 3) if pa > 0 else 0.0

        results.append({
            'name': name,
            'pa': pa, 'hr': hr, 'k_pct': k_pct,
            'barrel_pct': barrel_pct,
            'exit_velo': avg_ev,
            'launch_angle': avg_la,
            'hard_hit_pct': hard_hit,
            'bat_speed': avg_bat_speed,
            'xwoba': xwoba,
            'xslg': xslg,
            'pull_pct': pull_pct,
            'iso': avg_iso,
            'slg': slg,
            'batting_avg': avg_val,
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)

def _build_contact_log(df: pd.DataFrame):
    """Build contact log from pitch-by-pitch Statcast CSV (group_by=pitch).
    Stores last 8 batted ball events per player in _contact_log cache."""
    if df is None or df.empty: return
    df = df.copy()
    if 'player_name' in df.columns:
        df['name'] = df['player_name'].apply(lambda x: reverse_name(str(x)) if pd.notna(x) else "")
    else:
        return
    PITCH_SHORT = {
        '4-Seam Fastball': '4-Seam', 'Sinker': 'Sinker', 'Slider': 'Slider',
        'Sweeper': 'Sweeper', 'Changeup': 'Change', 'Curveball': 'Curve',
        'Cutter': 'Cutter', 'Splitter': 'Split', 'Knuckle Curve': 'K-Curve',
        'Fastball': 'FB',
    }
    # Only keep batted ball events
    contact = df[df['launch_speed'].notna() & (df['launch_speed'] > 0) & df['events'].notna()].copy()
    for name, grp in contact.groupby('name'):
        if not name: continue
        grp_sorted = grp.sort_values('game_date', ascending=False).head(8)
        events = []
        for _, row in grp_sorted.iterrows():
            result = str(row.get('events', '') or '').strip()
            if not result or result == 'nan': continue
            pitch_name = str(row.get('pitch_name', '') or '').strip()
            pitch_short = PITCH_SHORT.get(pitch_name, pitch_name[:6] if pitch_name else '--')
            try:
                events.append({
                    'date':       str(row.get('game_date', ''))[-5:],
                    'pitcher':    reverse_name(str(row.get('pitcher_name', '') or '').strip()),
                    'pitch_type': pitch_short,
                    'ev':         round(float(row['launch_speed']), 1),
                    'angle':      round(float(row['launch_angle']), 1) if pd.notna(row.get('launch_angle')) else 0,
                    'distance':   int(float(row['hit_distance_sc'])) if pd.notna(row.get('hit_distance_sc')) and float(row.get('hit_distance_sc', 0) or 0) > 0 else 0,
                    'bat_speed':  round(float(row['bat_speed']), 1) if pd.notna(row.get('bat_speed')) and float(row.get('bat_speed', 0) or 0) > 0 else 0,
                    'result':     result,
                })
            except Exception: continue
        if events:
            _contact_log[name.lower()] = events

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
    """Fetch last 8 games full batting stats from MLB Stats API.
    Reliable rolling window — Savant date filtering is broken for many players.
    Returns ISO, SLG, AVG, K%, HR, PA for each batter over their last 8 games."""
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
                    hr  = int(stat.get("homeRuns", 0) or 0)
                    pa  = int(stat.get("plateAppearances", 0) or 0)
                    ab  = int(stat.get("atBats", 0) or 0)
                    so  = int(stat.get("strikeOuts", 0) or 0)
                    slg_str = stat.get("slg", "0") or "0"
                    slg = float(slg_str) if slg_str not in (".---","") else 0.0
                    avg_str = stat.get("avg", "0") or "0"
                    avg = float(avg_str) if avg_str not in (".---","") else 0.0
                    iso = round(slg - avg, 3) if slg > 0 else 0.0
                    k_pct = round(so / max(pa, 1) * 100, 1) if pa > 0 else 0.0
                    l8d_hr_map[name.lower()] = {
                        "hr": hr, "pa": pa, "ab": ab,
                        "slg": slg, "avg": avg, "iso": iso,
                        "k_pct": k_pct, "name": name,
                    }
                except Exception: continue
        print(f"Fetched last-8-games stats for {len(l8d_hr_map)} batters")
        return l8d_hr_map
    except Exception as e:
        print(f"Last 8 games fetch error: {e}")
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

        # Batter 8d — aggregated stats per player (group_by=name)
        df = await fetch_savant_csv(savant_8d_url(), client)
        if not df.empty:
            _cache["bat_8d"] = calc_statcast_8d(df)
            print(f"bat_8d: {len(_cache['bat_8d'])} rows")
        else:
            print("bat_8d: 0 rows")

        # Contact log — separate 3-day window fetch (group_by=pitch, ~9MB)
        df_contact = await fetch_savant_csv(savant_contact_log_url(), client, timeout=180)
        if not df_contact.empty:
            _build_contact_log(df_contact)
            print(f"contact_log: {len(_contact_log)} players")
        else:
            print("contact_log: 0 players (will retry on next refresh)")

        # Pitcher 2026
        df = await fetch_savant_csv(savant_pitcher_url(min_pa=5), client)
        if not df.empty:
            _cache["pit_2026"] = calc_pitcher_stats(df)
            print(f"pit_2026: {len(_cache['pit_2026'])} rows")

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
    """Refresh 8-day data — aggregated stats + contact log"""
    async with httpx.AsyncClient(timeout=200) as client:
        # 8d aggregated stats (group_by=name)
        df = await fetch_savant_csv(savant_8d_url(), client)
        if not df.empty:
            agg = calc_statcast_8d(df)
            if not agg.empty:
                _cache["bat_8d"] = agg
                print(f"bat_8d refreshed: {len(agg)} players")
        # Contact log — separate fetch with 3-day window (~9MB, needs long timeout)
        df_contact = await fetch_savant_csv(savant_contact_log_url(), client, timeout=180)
        if not df_contact.empty:
            _build_contact_log(df_contact)
            _games_cache.clear()
            print(f"contact_log refreshed: {len(_contact_log)} players")
        else:
            print("contact_log: CSV empty — will retry next refresh")
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
    """Run in background — check every hour for scheduled tasks.
    Schedule (all ET):
    - 9am  — full Savant data refresh (season stats, splits, arsenal)
    - 11am — bat_8d refresh (Savant fully updated by now with last night's games)
    - 12pm — save daily predictions (uses fresh 11am L8D data, locked for the day)
    - 2am  — record results (all games finished)
    - 3am  — auto-recalibrate (Sundays only)
    - 4am  — save model log
    """
    # Track which tasks already ran today so redeploys mid-day don't skip them
    _scheduled_ran = {}  # { "YYYY-MM-DD:task" : True }

    while True:
        await asyncio.sleep(600)  # check every 10 min instead of hourly — catches missed windows
        now = datetime.utcnow()
        et_now = now - timedelta(hours=4)   # UTC → EDT (Apr–Nov)
        et_hour = et_now.hour
        et_date = et_now.date().isoformat()
        is_sunday = et_now.weekday() == 6

        def already_ran(task):
            return _scheduled_ran.get(f"{et_date}:{task}", False)

        def mark_ran(task):
            _scheduled_ran[f"{et_date}:{task}"] = True
            # Trim old keys (keep only today)
            for k in list(_scheduled_ran.keys()):
                if not k.startswith(et_date):
                    del _scheduled_ran[k]

        # 9am ET — full Savant season data refresh (window: 9:00–9:59)
        if et_hour == 9 and not already_ran("9am_refresh"):
            try:
                await load_all_savant_data()
                mark_ran("9am_refresh")
                print("9am ET: Full Savant refresh complete")
            except Exception as e:
                print(f"Daily refresh error: {e}")

        # 11am ET — bat_8d refresh (window: 11:00–11:59)
        if et_hour == 11 and not already_ran("11am_8d"):
            try:
                await refresh_8d()
                mark_ran("11am_8d")
                print("11am ET: bat_8d locked for the day")
            except Exception as e:
                print(f"8d refresh error: {e}")

        # 12pm ET — save predictions (window: 12:00–12:59)
        if et_hour == 12 and not already_ran("12pm_save"):
            try:
                await save_daily_predictions()
                mark_ran("12pm_save")
                print("12pm ET: Predictions saved with locked L8D data")
            except Exception as e:
                print(f"Prediction save error: {e}")

        # 2am ET — record results (window: 2:00–2:59)
        if et_hour == 2 and not already_ran("2am_results"):
            try:
                yesterday = (date.today() - timedelta(days=1)).isoformat()
                await record_results(yesterday)
                mark_ran("2am_results")
            except Exception as e:
                print(f"Result recording error: {e}")

        # 3am ET — auto-recalibrate Sundays (window: 3:00–3:59)
        if et_hour == 3 and is_sunday and not already_ran("3am_recal"):
            try:
                print(f"Sunday auto-recalibrate — {_model_weights.get('records_used',0)} records")
                await recalibrate_model()
                mark_ran("3am_recal")
            except Exception as e:
                print(f"Auto-recalibrate error: {e}")

        # Auto-retrain when record count crosses a depth threshold
        # Checks daily at 3am — if records crossed 1500/2500/5000/10000 since last train
        if et_hour == 3 and not already_ran("3am_depth_check"):
            try:
                mark_ran("3am_depth_check")
                current_records = _model_weights.get("records_used", 0)
                next_upgrade = _model_weights.get("next_depth_upgrade", 1500)
                if isinstance(next_upgrade, int) and current_records >= next_upgrade:
                    print(f"Auto depth upgrade: {current_records} records >= {next_upgrade} threshold")
                    result = await recalibrate_model()
                    new_depth = _model_weights.get("tree_depth", 4)
                    cal = _model_weights.get("calibration_factor", 1.0)
                    print(f"Depth upgraded to {new_depth}, cal_factor={cal:.3f}")
            except Exception as e:
                print(f"Depth upgrade check error: {e}")

        # 4am ET — save model log (window: 4:00–4:59)
        if et_hour == 4 and not already_ran("4am_log"):
            try:
                await save_model_log(_model_weights)
                mark_ran("4am_log")
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

async def save_daily_predictions(force: bool = False):
    """Save today's predictions to GitHub — uses same data as /games endpoint for consistency"""
    if not _cache["ready"]: return
    today = date.today().isoformat()
    path = f"data/predictions/{today}.json"
    existing, sha = await github_get_file(path)
    # Allow overwrite if hit_hr is still null (predictions not yet recorded)
    # force=True skips the guard entirely (used by /resave-today which handles hit_hr preservation itself)
    if existing and not force:
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

                # ── Get all batters with recent ABs for each team ──
                # Pull active roster from MLB API, cross-reference with bat_8d
                # Anyone with PA in last 8 days gets processed — DNP tag handles non-starters

                async def get_team_active_batters(team_id, team_name):
                    """Get roster players who have recent 8d data"""
                    try:
                        async with httpx.AsyncClient(timeout=10) as rc:
                            r3 = await rc.get(f"{MLB_API}/teams/{team_id}/roster?rosterType=active&season={current_season()}")
                            roster = r3.json().get("roster", [])
                        players = []
                        for p in roster:
                            pos = p.get("position", {}).get("type", "")
                            if pos == "Pitcher": continue  # skip pitchers
                            name = p.get("person", {}).get("fullName", "")
                            if not name: continue
                            b8d = get_batter_8d(name)
                            if b8d.get("pa", 0) >= 1:
                                players.append(name)
                        return players if players else []
                    except Exception:
                        return []

                away_players = await get_team_active_batters(away_team_id, away_team)
                home_players = await get_team_active_batters(home_team_id, home_team)

                # Fall back to projected lineup if roster fetch fails
                if not away_players:
                    proj, _ = await fetch_projected_lineup(away_team_id, away_team)
                    away_players = [p.get("name","") or p.get("person",{}).get("fullName","") for p in proj]
                    away_players = [n for n in away_players if n]
                if not home_players:
                    proj, _ = await fetch_projected_lineup(home_team_id, home_team)
                    home_players = [p.get("name","") or p.get("person",{}).get("fullName","") for p in proj]
                    home_players = [n for n in home_players if n]

                lineup_src = "bat_8d_roster"

                for player_list, team, opp_p_name, opp_p_hand in [
                    (away_players, away_team, home_p.get("fullName","TBD"), home_p_hand),
                    (home_players, home_team, away_p.get("fullName","TBD"), away_p_hand),
                ]:
                    for name in player_list:
                        if not name: continue
                        bat_hand = "R"
                        nl = name.lower().strip()
                        cached_hand = _cache.get("player_hands", {}).get(nl, {})
                        if cached_hand:
                            bat_hand = cached_hand.get("bat_side", "R")
                        if bat_hand == "S": bat_hand = "L" if opp_p_hand == "R" else "R"
                        park_factor = get_park_hr_factor(home_team, bat_hand)
                        wx_mult, _ = calc_weather_multiplier(home_team, wind_speed, wind_dir, temp, bat_hand)
                        hr_prob, breakdown, _, _, _, _, _ = compute_hr_probability(
                            name, bat_hand, opp_p_name, opp_p_hand, park_factor, wx_mult, home_team)
                        if hr_prob < 2: continue  # save near-everything for full tree training population
                        top_pitches = get_pitcher_top_pitches(opp_p_name)[:2]
                        pitch1 = top_pitches[0] if len(top_pitches) > 0 else {}
                        pitch2 = top_pitches[1] if len(top_pitches) > 1 else {}
                        pitch_score, _ = compute_pitch_matchup(opp_p_name, name)
                        pa_data = get_avg_pa_per_game(name)
                        # Raw batter season stats — 2026 only
                        bc2 = get_batter_stats(name, 2026)
                        barrel_s = round(bc2.get("barrel_pct",0), 1)
                        la_s     = round(bc2.get("launch_angle",0), 1)
                        ev_s     = round(bc2.get("exit_velo",0), 1)
                        iso_s    = round(bc2.get("iso",0), 3)
                        hh_s     = round(bc2.get("hard_hit_pct",0), 1)
                        k_s      = round(bc2.get("k_pct",0), 1)
                        # Raw batter L8D stats
                        b8d2 = get_batter_8d(name)
                        barrel_l8d   = round(b8d2.get("barrel_pct",0), 1)
                        la_l8d       = round(b8d2.get("launch_angle",0), 1)
                        ev_l8d       = round(b8d2.get("exit_velo",0), 1)
                        iso_l8d      = round(b8d2.get("iso",0), 3)
                        hh_l8d       = round(b8d2.get("hard_hit_pct",0), 1)
                        pa_l8d       = int(b8d2.get("pa",0))
                        slg_l8d      = round(b8d2.get("slg",0), 3)
                        xslg_l8d     = round(b8d2.get("xslg",0), 3)
                        xslg_gap_l8d = round(b8d2.get("xslg",0) - b8d2.get("slg",0), 3) if b8d2.get("xslg",0) > 0 else 0
                        xwoba_l8d    = round(b8d2.get("xwoba",0), 3)
                        bat_speed_l8d = round(b8d2.get("bat_speed",0), 1)
                        # Raw batter split vs pitcher hand
                        b_split2   = get_batter_split(name, opp_p_hand)
                        iso_split  = round(b_split2.get("iso",0), 3)
                        slg_split  = round(b_split2.get("slg",0), 3)
                        hr_split   = int(b_split2.get("hr",0))
                        pa_split   = int(b_split2.get("pa",0))
                        # Raw pitcher stats — 2026 only
                        pc2  = get_pitcher_stats(opp_p_name, 2026)
                        ip26 = pc2.get("ip",0)
                        pit_hr9_s   = round(pc2.get("hr9",0), 2)
                        pit_era_s   = round(pc2.get("era",0), 2)
                        pit_hh_s    = round(pc2.get("hard_hit_pct",0), 1)
                        pit_k9_s    = round(pc2.get("k9",0), 1)
                        pit_is_new  = pc2.get("is_new", False)
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
                            "pa_season": int(bc2.get("pa", 0)),
                            # ── BATTER L8D ──
                            "barrel_pct_l8d": barrel_l8d,
                            "la_l8d": la_l8d,
                            "ev_l8d": ev_l8d,
                            "iso_l8d": iso_l8d,
                            "hard_hit_l8d": hh_l8d,
                            "pa_l8d": pa_l8d,
                            "l8d_hr": get_l8d_hr(name),
                            "slg_l8d": slg_l8d,
                            "xslg_l8d": xslg_l8d,
                            "xslg_gap_l8d": xslg_gap_l8d,
                            "xwoba_l8d": xwoba_l8d,
                            "bat_speed_l8d": bat_speed_l8d,
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
                            "games_played": pa_data.get("games",0),
                            # ── ROTATION METADATA ──
                            "rotation_round": get_rotation_round(),
                            "rotation_day": get_rotation_day(),
                            # ── ROUND 2 CANDIDATES (stored from day 1, evaluated at day 45) ──
                            "fb_pct_season": round(bc2.get("fb_pct",0), 1),
                            "pull_pct_season": round(bc2.get("pull_pct",0), 1),
                            "pit_fb_pct_allowed": round(pc2.get("fb_pct",0), 1),
                            "hard_hit_l8d": hh_l8d,
                            "k_pct_l8d": round(b8d2.get("k_pct",0), 1),
                            # ── NEW STATS ──
                            "chase_rate_season": round(bc2.get("chase_rate",0), 1),
                            "pit_stuff_plus": round(pc2.get("stuff_plus",100), 1),
                            "pitcher_is_new": pit_is_new,
                            # ── ROUND 3 CANDIDATES ──
                            "pit_era_season": pit_era_s,
                            "pit_era_diff": round(pit_era_s - 4.20, 2) if pit_era_s > 0 else 0,
                            "pit_hr_fb_pct": round(pc2.get("hr_fb_pct",0), 1),
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

async def record_results(target_date: str, force: bool = False):
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
                            nl = name.lower().strip()
                            ab = int(stats.get("atBats", 0) or 0)
                            actual_ab[nl] = ab
                            if int(stats.get("homeRuns", 0) or 0) > 0:
                                hr_hitters.add(nl)
                                import unicodedata
                                normalized = unicodedata.normalize('NFD', nl)
                                normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
                                hr_hitters.add(normalized)
                                actual_ab[normalized] = ab
                except Exception: continue

        updated = 0
        dnp_count = 0
        hr_count = 0
        for rec in records:
            # Process if null OR if force=True and currently DNP (may be wrong)
            if rec.get("hit_hr") is None or (force and rec.get("hit_hr") == "DNP"):
                nl = rec["name"].lower().strip()
                # Normalize accented chars for matching
                import unicodedata
                nl_norm = unicodedata.normalize('NFD', nl)
                nl_norm = ''.join(c for c in nl_norm if unicodedata.category(c) != 'Mn')

                # Try exact match
                ab = actual_ab.get(nl, actual_ab.get(nl_norm, -1))
                hit = nl in hr_hitters or nl_norm in hr_hitters

                # Try last name match
                if ab == -1:
                    last = nl.split()[-1]
                    last_norm = nl_norm.split()[-1]
                    for k, v in actual_ab.items():
                        k_parts = k.split()
                        if k_parts and (k_parts[-1] == last or k_parts[-1] == last_norm):
                            ab = v
                            if k in hr_hitters:
                                hit = True
                            break

                if ab == -1:
                    ab = 0

                # Only mark DNP if truly 0 AB and not a known HR hitter
                if ab == 0 and not hit:
                    rec["hit_hr"] = "DNP"
                    rec["actual_ab"] = 0
                    dnp_count += 1
                else:
                    rec["hit_hr"] = 1 if hit else 0
                    rec["actual_ab"] = max(ab, 0)
                    if hit: hr_count += 1
                    updated += 1

        content_updated = json.dumps(records, indent=2)
        await github_put_file(path, content_updated,
            f"results: {target_date} ({len(hr_hitters)} HRs, {dnp_count} DNP, force={force})", sha)
        print(f"Recorded results for {target_date}: {len(hr_hitters)} HR hitters, "
              f"{dnp_count} DNP, {updated} updated, {hr_count} HRs confirmed")
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
    asyncio.create_task(startup_catchup())
    asyncio.create_task(startup_train_tree())

async def startup_train_tree():
    """Train Decision Tree on startup — waits for Savant data to fully load first."""
    # Wait for cache to be ready before training tree
    # Polls every 10 seconds, gives up after 5 minutes
    for _ in range(30):
        await asyncio.sleep(10)
        if _cache.get("ready"):
            break
    else:
        print("Startup tree: cache never became ready — skipping tree training")
        return
    try:
        print("Startup: cache ready, training Decision Tree...")
        result = await recalibrate_model(save_to_github=False)
        if isinstance(result, dict) and result.get("status") == "done":
            print(f"Startup tree OK: depth={result.get('tree_depth')}, "
                  f"leaves={result.get('n_leaves')}, "
                  f"cal={_model_weights.get('calibration_factor', 1.0):.3f}")
        else:
            print(f"Startup tree result: {result}")
    except Exception as e:
        import traceback
        print(f"Startup tree training failed (non-fatal): {e}")
        traceback.print_exc()

async def startup_catchup():
    """On startup:
    1. Check if yesterday results were missed and record them
    2. Check if today predictions not saved yet and save them
    """
    await asyncio.sleep(60)  # wait for data to load first
    import json

    # Record yesterday results if missed
    try:
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        ypath = f"data/predictions/{yesterday}.json"
        ycontent, _ = await github_get_file(ypath)
        if ycontent:
            records = json.loads(ycontent)
            nulls = [r for r in records if r.get("hit_hr") is None]
            if nulls:
                print(f"Startup catchup: {len(nulls)} unrecorded results for {yesterday} — recording now")
                await record_results(yesterday)
            else:
                print(f"Startup catchup: {yesterday} results already complete")
    except Exception as e:
        print(f"Startup catchup (results) error: {e}")

    # Save today predictions if not yet saved
    try:
        today = date.today().isoformat()
        tpath = f"data/predictions/{today}.json"
        tcontent, _ = await github_get_file(tpath)
        if not tcontent:
            print(f"Startup catchup: no predictions for {today} — saving now")
            await save_daily_predictions()
        else:
            print(f"Startup catchup: {today} predictions already saved")
    except Exception as e:
        print(f"Startup catchup (predictions) error: {e}")

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
    df = _cache["bat_2026"]
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
        "chase_rate": gs(row, "chase_rate"),
        "iso": gs(row, "iso"),
        "slg_percent": gs(row, "slg_percent"),
        "batting_avg": gs(row, "batting_avg"),
        "k_pct": gs(row, "k_pct"),
        "hr_fb_pct": gs(row, "hr_fb_pct"),
        "hr": gs(row, "hr"),
    }
    return stats
def get_batter_8d(name):
    """L8D stats from two sources:
    1. bat_8d cache — pitch-by-pitch Statcast aggregated by calc_statcast_8d
       gives: barrel%, EV, LA, hard hit%, bat speed, xwOBA, xSLG, pull%, K%, HR, PA
    2. bat_l8d_hr cache — MLB API lastXGames=8
       gives: PA, HR, ISO, SLG, AVG, K% (reliable counting stats)
    Statcast source is preferred for Statcast metrics, MLB API for counting stats."""
    # ── Statcast aggregated data (pitch-by-pitch) ──
    df = _cache["bat_8d"]
    row = fuzzy_match(name, df)

    # ── MLB API counting stats (reliable) ──
    nl = name.lower().strip()
    mlb_data = _cache.get("bat_l8d_hr", {})
    mlb = mlb_data.get(nl)
    if not mlb:
        last = nl.split()[-1]
        for k, v in mlb_data.items():
            if last in k: mlb = v; break

    # No data at all
    if row is None and (not mlb or mlb.get("pa", 0) == 0):
        return {}

    # Statcast metrics from aggregated pitch-by-pitch
    barrel_pct = gs(row, "barrel_pct") if row is not None else 0.0
    exit_velo  = gs(row, "exit_velo")  if row is not None else 0.0
    launch_angle = gs(row, "launch_angle") if row is not None else 0.0
    hard_hit_pct = gs(row, "hard_hit_pct") if row is not None else 0.0
    pull_pct   = gs(row, "pull_pct")   if row is not None else 0.0
    bat_speed  = gs(row, "bat_speed")  if row is not None else 0.0
    xwoba      = gs(row, "xwoba")      if row is not None else 0.0
    xslg       = gs(row, "xslg")       if row is not None else 0.0

    # Counting stats: MLB API primary, Statcast fallback
    if mlb and mlb.get("pa", 0) > 0:
        pa    = mlb.get("pa", 0)
        hr    = mlb.get("hr", 0)
        iso   = mlb.get("iso", 0.0)
        slg   = mlb.get("slg", 0.0)
        avg   = mlb.get("avg", 0.0)
        k_pct = mlb.get("k_pct", 0.0)
    elif row is not None:
        pa    = int(gs(row, "pa"))
        hr    = gs(row, "hr")
        iso   = gs(row, "iso")
        slg   = gs(row, "slg")
        avg   = gs(row, "batting_avg")
        k_pct = gs(row, "k_pct")
    else:
        return {}

    if pa == 0:
        return {}

    return {
        "pa": pa, "hr": hr,
        "barrel_pct":    barrel_pct,
        "exit_velo":     exit_velo,
        "launch_angle":  launch_angle,
        "hard_hit_pct":  hard_hit_pct,
        "pull_pct":      pull_pct,
        "bat_speed":     bat_speed,
        "xwoba":         xwoba,
        "xslg":          xslg,
        "iso": iso, "k_pct": k_pct,
        "slg": slg, "avg": avg,
        "hr_rate": (hr / max(pa, 1)) * 600 if pa > 0 else 0,
    }

def get_contact_log(name):
    """Get last 8 batted ball events for a player from the contact log cache"""
    nl = name.lower().strip()
    if nl in _contact_log: return _contact_log[nl]
    last = nl.split()[-1]
    for k, v in _contact_log.items():
        if last in k: return v
    return []

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
    df = _cache["pit_2026"]
    row = fuzzy_match(name, df)

    nl = name.lower().strip()
    ip_data = _cache["player_ip"].get(nl, {})
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

    # NEW pitcher flag — no 2026 data at all, treat as neutral
    is_new = (ip == 0 and (row is None))

    if row is None:
        return {"era": era, "ip": ip, "hr9": hr9, "k9": k9, "avg_ip": avg_ip, "gs": gs_val,
                "hard_hit_pct": 0, "barrel_pct_allowed": 0, "fb_pct": 0, "k_pct": 0,
                "stuff_plus": 100, "is_new": is_new}
    return {
        "era": era or gs(row, "era"),
        "ip": ip, "hr9": hr9, "k9": k9, "avg_ip": avg_ip, "gs": gs_val,
        "hard_hit_pct": gs(row, "hard_hit_pct"),
        "barrel_pct_allowed": gs(row, "barrel_pct_allowed"),
        "fb_pct": gs(row, "fb_pct"),
        "k_pct": gs(row, "k_pct"),
        "stuff_plus": gs(row, "stuff_plus") or 100,
        "is_new": is_new,
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
    # Always use 2026 data only — previous year not relevant by betting season
    # April thin data is acceptable — we're not betting until May 31
    # Safety net: if truly no 2026 data at all, fall back to 2025 temporarily
    pa = float(pa_2026 or 0)
    if pa == 0:
        return 0.0, 1.0  # no 2026 ABs yet — use 2025 as placeholder
    return 1.0, 0.0

def get_pitcher_blend_weights(ip_2026, ip_2025):
    # Always use 2026 data only — previous year not relevant by betting season
    # April thin data is acceptable — we're not betting until May 31
    # Safety net: if truly no 2026 data at all, fall back to 2025 temporarily
    ip = float(ip_2026 or 0)
    if ip == 0:
        return 0.0, 1.0  # no 2026 IP yet — use 2025 as placeholder
    return 1.0, 0.0

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
    # Use handedness-specific HR bearing
    # RHB pull to LF (~NW), LHB pull to RF (~SE)
    if batter_hand == "L":
        hr_bearing = stadium.get("hr_bearing_L", stadium.get("hr_bearing", 135))
    else:
        hr_bearing = stadium.get("hr_bearing_R", stadium.get("hr_bearing", 305))
    open_factor = stadium.get("open_factor", 0.5)
    # Open-Meteo gives wind direction as where wind comes FROM (meteorological convention)
    # We need where it's blowing TO — flip 180 degrees
    wind_toward = (wind_direction + 180) % 360
    diff = angle_diff(wind_toward, hr_bearing)
    alignment = math.cos(math.radians(diff))
    speed_factor = 0 if wind_speed < 5 else 0.3 if wind_speed < 10 else 0.7 if wind_speed < 16 else 1.0
    wind_mult = 1.0 + (alignment * speed_factor * 0.12 * open_factor)
    temp_mult = 1.06 if temperature >= 80 else 1.02 if temperature >= 70 else 0.91 if temperature < 50 else 0.96 if temperature < 60 else 1.0
    # Direction label based on batter handedness
    if wind_speed < 5:
        direction_label = "Calm"
    elif alignment > 0.5:
        direction_label = "Blowing Out"
    elif alignment < -0.5:
        direction_label = "Blowing In"
    elif abs(alignment) <= 0.5:
        direction_label = "Favors Righties" if batter_hand == "R" and alignment > 0 else "Favors Lefties" if batter_hand == "L" and alignment > 0 else "Crosswind"
    else:
        direction_label = "Crosswind"
    return round(wind_mult * temp_mult, 3), direction_label

def sigmoid_to_prob(raw_score):
    """
    No sigmoid — direct pass-through with floor and cap.
    base_rate × batter_score × matchup_score × always_on = hr_prob
    Floor: 2%, Cap: 15%
    Transparent math — every output is fully traceable.
    """
    return round(min(max(raw_score, 2.0), 15.0), 1)

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

def get_bullpen_hr9(home_team):
    """Get bullpen HR/9 for home team from cache."""
    bullpen = _cache.get("team_bullpen", {})
    return bullpen.get(home_team, {}).get("hr9", 1.2)

def get_bat_platoon(bat_hand, pit_hand):
    """Platoon multiplier for batter vs pitcher hand."""
    # L vs R and R vs L are favorable platoon matchups
    if bat_hand == "L" and pit_hand == "R": return 1.08
    if bat_hand == "R" and pit_hand == "L": return 1.08
    return 0.94  # same hand = unfavorable

def get_pit_platoon(pit_hand, bat_hand):
    """Platoon multiplier for pitcher vs batter hand."""
    if pit_hand == "R" and bat_hand == "L": return 0.94
    if pit_hand == "L" and bat_hand == "R": return 0.94
    return 1.06  # same hand = pitcher advantage


def compute_hr_prob_multiplicative(
        name, bat_hand, opp_p_name, opp_p_hand, park_factor, weather_mult, home_team=""):
    """
    Decision Tree HR probability.
    Builds a feature vector for this batter+matchup and runs it through
    the trained tree. Falls back to a simple rule-based estimate if tree
    not trained yet.
    Returns: (hr_prob, breakdown, archetype, trend, reasons, platoon_tag, conf)
    """
    import numpy as np

    # ── Gather all stats ──
    bc   = get_batter_stats(name, 2026)
    b8d  = get_batter_8d(name)
    pc   = get_pitcher_stats(opp_p_name, 2026)
    b_split = get_batter_split(name, opp_p_hand)
    p_split = get_pitcher_split(opp_p_name, bat_hand)

    pa_26       = bc.get("pa", 0)
    has_8d      = b8d.get("pa", 0) >= 5
    pit_ip_s    = pc.get("ip", 0) or 0
    pit_ip_vh   = p_split.get("ip", 0) or 0

    # ── Build feature vector ──
    def fv(stat):
        """Get clean feature value for this batter+matchup."""
        # Map stat name to data source
        batter_season = {
            "barrel_pct_season": bc.get("barrel_pct", 0),
            "hard_hit_season":   bc.get("hard_hit_pct", 0),
            "ev_season":         bc.get("exit_velo", 0),
            "iso_season":        bc.get("iso", 0),
            "la_season":         bc.get("launch_angle", 0),
            "pull_pct_season":   bc.get("pull_pct", 0),
        }
        batter_l8d = {
            "barrel_pct_l8d":  b8d.get("barrel_pct", 0),
            "hard_hit_l8d":    b8d.get("hard_hit_pct", 0),
            "ev_l8d":          b8d.get("exit_velo", 0),
            "bat_speed_l8d":   b8d.get("bat_speed", 0),
            "xwoba_l8d":       b8d.get("xwoba", 0),
            "xslg_l8d":        b8d.get("xslg", 0),
            "slg_l8d":         b8d.get("slg", 0),
            "iso_l8d":         b8d.get("iso", 0),
            "k_pct_l8d":       b8d.get("k_pct", 0),
            "la_l8d":          b8d.get("launch_angle", 0),
        }
        batter_splits = {
            "iso_vs_hand":  b_split.get("iso", 0),
            "slg_vs_hand":  b_split.get("slg", 0),
        }
        pitcher = {
            "pit_hr9_season":      pc.get("hr9", 0),
            "pit_era_season":      pc.get("era", 4.2),
            "pit_hard_hit_season": pc.get("hard_hit_pct", 0),
            "pit_k9_season":       pc.get("k9", 0),
            "pit_hr9_vs_hand":     p_split.get("hr9", 0),
            "pit_slg_vs_hand":     p_split.get("slg", 0),
        }
        context = {
            "park_factor":       park_factor,
            "weather_mult":      weather_mult,
            "bullpen_hr9":       get_bullpen_hr9(home_team),
            "bat_platoon_mult":  get_bat_platoon(bat_hand, opp_p_hand),
            "pit_platoon_mult":  get_pit_platoon(opp_p_hand, bat_hand),
            "pitch_matchup_score": compute_pitch_matchup(opp_p_name, name)[0],
        }
        all_vals = {**batter_season, **batter_l8d, **batter_splits,
                    **pitcher, **context}
        raw = all_vals.get(stat, 0)

        # Zero disambiguation
        if stat in ZERO_DOMINANT:
            ip_field_map = {
                "pit_hr9_season":  pit_ip_s,
                "pit_hr9_vs_hand": pit_ip_vh,
                "pit_slg_vs_hand": pit_ip_vh,
            }
            ip = ip_field_map.get(stat, 0)
            min_ip = 5 if "vs_hand" in stat else 1
            if ip >= min_ip:
                return float(raw) if raw is not None else 0.0
            else:
                fallback = pc.get("hr9", _dt_medians.get(stat, 1.1))
                return float(fallback)

        # Regular stats — zero means missing, use median
        if raw is None or raw == 0:
            return _dt_medians.get(stat, 0.0)
        return float(raw)

    # ── Run Decision Tree if trained ──
    if _dt_model is not None and _dt_features:
        feat_vec = np.array([[fv(stat) for stat in _dt_features]])
        proba = _dt_model.predict_proba(feat_vec)[0]
        raw_prob = proba[1] * 100

        # Apply calibration factor — corrects for biased training population
        # Factor auto-removes as dataset HR rate approaches true league rate (~2.8%)
        cal_factor = _model_weights.get("calibration_factor", 1.0)
        calibrated = raw_prob * cal_factor

        # Floor 2%, cap 15%
        hr_prob = round(min(max(calibrated, 2.0), 15.0), 1)

        print(f"DT: raw={raw_prob:.1f}% × cal={cal_factor:.3f} → {hr_prob}%") if raw_prob > 10 else None
    else:
        # ── Fallback: rule-based estimate while tree is training on startup ──
        # Uses same stats tree found most important: hard_hit, iso_vs_hand,
        # pit_hr9_vs_hand, la_season, iso_l8d, pit_hr9_season, barrel_season
        barrel_s  = fv("barrel_pct_season")
        hh_s      = fv("hard_hit_season")
        iso_hand  = fv("iso_vs_hand")
        la_s      = fv("la_season")
        iso_l8d_v = fv("iso_l8d")
        pit_hr9_vh = fv("pit_hr9_vs_hand")
        pit_hr9_s  = fv("pit_hr9_season")

        # Start at league avg and adjust based on top tree features
        base = 2.8

        # Batter quality (hard_hit_season most important per tree)
        hh_med = _dt_medians.get("hard_hit_season", 46.8)
        if hh_s > hh_med * 1.3:    base *= 2.0   # elite hard hit
        elif hh_s > hh_med * 1.1:  base *= 1.5
        elif hh_s < hh_med * 0.8:  base *= 0.7

        # ISO vs hand
        iso_med = _dt_medians.get("iso_vs_hand", 0.239)
        if iso_hand > iso_med * 1.4:   base *= 1.6
        elif iso_hand > iso_med * 1.1: base *= 1.2
        elif iso_hand < iso_med * 0.6: base *= 0.8

        # Pitcher vulnerability (pit_hr9_vs_hand)
        pit_med = _dt_medians.get("pit_hr9_vs_hand", 1.08)
        if pit_hr9_vh > pit_med * 1.5:   base *= 1.5
        elif pit_hr9_vh > pit_med * 1.1: base *= 1.2
        elif pit_hr9_vh < pit_med * 0.5: base *= 0.6
        elif pit_hr9_vh < pit_med * 0.8: base *= 0.8

        # Apply calibration factor even to fallback
        cal_factor = _model_weights.get("calibration_factor", 0.163)
        hr_prob = round(min(max(base * park_factor * weather_mult, 2.0), 15.0), 1)
        print(f"FALLBACK (tree not ready): {name} base={base:.1f} → {hr_prob}%")

    # ── Build breakdown for display ──
    importances = _model_weights.get("feature_importances", {})
    top_stats = list(importances.items())[:6]

    # Archetype
    barrel_s = fv("barrel_pct_season")
    iso_s    = fv("iso_season")
    ev_s     = fv("ev_season")
    if barrel_s > 15 and iso_s > 0.250:   archetype = "Power Hitter"
    elif barrel_s > 10 and ev_s > 90:     archetype = "Line Drive Power"
    elif iso_s < 0.120:                   archetype = "Contact Hitter"
    else:                                 archetype = "Balanced"

    # Trend
    barrel_l8d = fv("barrel_pct_l8d")
    iso_l8d    = fv("iso_l8d")
    if barrel_l8d > barrel_s * 1.2 or iso_l8d > iso_s * 1.2: trend = "hot"
    elif barrel_l8d < barrel_s * 0.8:                          trend = "cold"
    else:                                                       trend = "neutral"

    # Reasons
    reasons = []
    if barrel_s > 15:   reasons.append(f"Barrel% {barrel_s:.1f}%")
    if iso_s > 0.250:   reasons.append(f"ISO .{int(iso_s*1000):03d}")
    pit_hr9 = fv("pit_hr9_season")
    if pit_hr9 > 1.4:   reasons.append(f"Pit HR/9 {pit_hr9:.2f}")
    if trend == "hot":  reasons.append("Hot streak")

    # Platoon tag
    bpm = fv("bat_platoon_mult")
    ppm = fv("pit_platoon_mult")
    platoon_tag = None
    if bpm >= 1.20: platoon_tag = f"Batter strong vs {opp_p_hand}HP"
    if ppm >= 1.20: platoon_tag = (platoon_tag + " + " if platoon_tag else "") + f"SP weak vs {bat_hand}HB"

    conf = "High" if pa_26 >= 50 and has_8d else "Medium" if pa_26 >= 20 else "Low"

    breakdown = {
        "model_type": "decision_tree",
        "tree_trained": _dt_model is not None,
        "feature_importances": importances,
        "top_features": top_stats,
        "barrel_season": barrel_s,
        "iso_season":    iso_s,
        "ev_season":     ev_s,
        "pit_hr9":       pit_hr9,
        "park_factor":   park_factor,
        "weather_mult":  weather_mult,
        "pa_season":     pa_26,
        "has_8d":        has_8d,
        "bat_platoon_mult": bpm,
        "pit_platoon_mult": ppm,
        "pitch_bonus":   fv("pitch_matchup_score"),
        "batter_score":  round(barrel_s / 10, 2) if barrel_s > 0 else 1.0,
        "matchup_score": round(pit_hr9 / 1.1, 2) if pit_hr9 > 0 else 1.0,
        "hr_season":     int(bc.get("hr", 0)),
        "k_mult":        1.0,
        "bullpen_hr9":   fv("bullpen_hr9"),
        "iso_vs_hand":   fv("iso_vs_hand"),
        "xwoba_l8d":     fv("xwoba_l8d"),
        "xslg_l8d":      fv("xslg_l8d"),
        "split_pa":      b_split.get("pa", 0),
        "split_hr":      int(b_split.get("hr", 0)),
        "split_slg":     round(b_split.get("slg", 0), 3),
        "split_iso":     round(b_split.get("iso", 0), 3),
        "split_woba":    round(b_split.get("woba", 0), 3),
        "split_k_pct":   round(b_split.get("k_pct", 0), 1),
        "hr9_season":    round(pc.get("hr9", 0), 2),
        "hr9_split":     round(p_split.get("hr9", 0), 2),
        "pit_hard":      round(pc.get("hard_hit_pct", 0), 1),
        "pit_blend_note": f"2026 ({pit_ip_s:.0f} IP)",
        "blend_note":    f"2026 ({pa_26} PA)" + (" + L8D" if has_8d else ""),
        "active_stats":  _dt_features[:8] if _dt_features else [],
        "stat_mults":    {"batter_score": 1.0, "matchup_score": 1.0},
        "base_rate":     2.8,
        "la_use":        fv("la_season"),
        "barrel_use":    barrel_s,
        "barrel_mult":   1.0,
        "la_mult":       1.0,
        "pit_vuln_mult": 1.0,
        "hot_cold_mult": 1.0,
        "pit_modifier":  1.0,
        "pull_s":        fv("pull_pct_season"),
        "hh_season":     fv("hard_hit_season"),
        "split_ip_vs_bat": pit_ip_vh,
        "slg_vs_bat":    round(p_split.get("slg", 0), 3) if pit_ip_vh >= 5 else 0,
        "pitch_breakdown": [],
        "data_conf": {
            "barrel": 1 if barrel_s > 0 and pa_26 >= 20 else 0,
            "hard_hit": 1 if fv("hard_hit_season") > 0 and pa_26 >= 20 else 0,
        },
        "n_pit_components": 1 if pit_hr9 > 0 else 0,
    }

    return hr_prob, breakdown, archetype, trend, reasons, platoon_tag, conf


def compute_hr_probability(name, bat_hand, opp_p_name, opp_p_hand, park_factor, weather_mult, home_team=""):
    return compute_hr_prob_multiplicative(name, bat_hand, opp_p_name, opp_p_hand, park_factor, weather_mult, home_team)

async def fetch_weather(lat, lon, game_time_utc):
    try:
        # Default to 1pm local (most common day game time)
        local_hour = 13
        if game_time_utc:
            try:
                dt = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
                # Convert UTC to ET (UTC-4 during EDT, UTC-5 during EST)
                # April-November = EDT = UTC-4
                et_offset = -4
                local_hour = (dt.hour + et_offset) % 24
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
        # Find the index matching the local game hour
        idx = 0
        for i, t in enumerate(times):
            if f"T{local_hour:02d}:" in t:
                idx = i
                break
        if idx == 0 and len(temps) > local_hour:
            idx = local_hour  # fallback
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
    ip_26 = pc.get("ip", 0)
    is_new = pc.get("is_new", False)
    top_pitches = get_pitcher_top_pitches(p_name)
    vs_L = get_pitcher_split(p_name, "L")
    vs_R = get_pitcher_split(p_name, "R")
    nl = p_name.lower().strip()
    ip_data = _cache["player_ip"].get(nl, {})
    if not ip_data:
        last = nl.split()[-1]
        for k, v in _cache["player_ip"].items():
            if last in k: ip_data = v; break
    k9_val  = pc.get("k9", 0)
    avg_ip  = ip_data.get("avg_ip", 5.0) or 5.0
    gs_val  = ip_data.get("gs", 0)
    return {
        "name": p_name, "hand": p_hand,
        "era": round(pc.get("era", 0), 2) or None,
        "hr9": round(pc.get("hr9", 0), 2) or None,
        "hard_hit_pct": round(pc.get("hard_hit_pct", 0), 1) or None,
        "barrel_pct": round(pc.get("barrel_pct_allowed", 0), 1) or None,
        "ip_2026": round(ip_26, 1),
        "is_new": is_new,
        "blend_note": f"2026 season ({ip_26:.0f} IP)" + (" — NEW" if is_new else ""),
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

@app.get("/debug-contact")
async def debug_contact(player: str = "Yordan Alvarez"):
    """Debug contact log — check cache size, sample keys, and player lookup"""
    nl = player.lower().strip()
    last = nl.split()[-1]
    # Direct match
    direct = _contact_log.get(nl)
    # Partial match
    partial_keys = [k for k in _contact_log if last in k]
    # Sample of what's in cache
    sample_keys = list(_contact_log.keys())[:10]
    return {
        "cache_size": len(_contact_log),
        "player_searched": player,
        "direct_match": direct is not None,
        "direct_events": len(direct) if direct else 0,
        "partial_matches": partial_keys[:5],
        "sample_cache_keys": sample_keys,
        "contact_url": savant_contact_log_url(),
        "events": direct or (list(_contact_log.values())[0] if _contact_log else []),
    }

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

@app.get("/recalibrate")
async def manual_recalibrate():
    """Manually trigger model recalibration — requires 50+ completed records"""
    result = await recalibrate_model()
    return result

@app.get("/model-weights")
async def get_model_weights():
    """Return current model weights and calibration status"""
    season_start = date(2026, 4, 13)
    days_since_start = (date.today() - season_start).days
    return {
        "weights": _model_weights,
        "calibration": {
            "round": get_rotation_round(),
            "last_calibrated": _model_weights.get("last_calibrated"),
            "records_used": _model_weights.get("records_used", 0),
            "days_since_season_start": days_since_start,
            "auto_recalibrate": "Every Sunday 3am ET",
        },
        "league_constants": LEAGUE_CONSTANTS,
    }

@app.get("/save-predictions")
async def manual_save_predictions():
    """Manually trigger saving today's predictions"""
    await save_daily_predictions()
    return {"status": "done", "date": date.today().isoformat()}


@app.get("/resave-today")
async def resave_today_predictions():
    """
    Re-run today's predictions with current model weights.
    Preserves any hit_hr results already recorded (HRs that already happened).
    Only updates model_hr_pct and model breakdown fields — never touches hit_hr.
    Use this after pushing a new model/LR calibration mid-day.
    """
    import json
    if not _cache["ready"]:
        return {"error": "Cache not ready"}
    today = date.today().isoformat()
    path = f"data/predictions/{today}.json"

    # Load existing file so we can preserve hit_hr values
    existing_content, sha = await github_get_file(path)
    existing_results = {}  # name+team -> hit_hr
    if existing_content:
        try:
            ex_recs = json.loads(existing_content)
            for r in ex_recs:
                key = f"{r.get('name','')}|{r.get('team','')}"
                existing_results[key] = r.get("hit_hr")  # preserve hit_hr (0, 1, DNP, or None)
        except Exception as e:
            print(f"resave-today: could not parse existing file: {e}")

    # Run full save — this computes fresh model_hr_pct with current LR weights
    await save_daily_predictions(force=True)

    # Now re-load what was just saved and patch hit_hr values back in
    new_content, new_sha = await github_get_file(path)
    if not new_content:
        return {"error": "Save ran but file not found after write"}

    try:
        new_recs = json.loads(new_content)
        patched = 0
        for r in new_recs:
            key = f"{r.get('name','')}|{r.get('team','')}"
            if key in existing_results and existing_results[key] is not None:
                r["hit_hr"] = existing_results[key]
                patched += 1
        # Write patched file back
        final_content = json.dumps(new_recs, indent=2)
        await github_put_file(path, final_content, f"resave+patch: {today} ({patched} results preserved)", new_sha)
        preserved = {k: v for k, v in existing_results.items() if v is not None}
        return {
            "status": "done",
            "date": today,
            "total_records": len(new_recs),
            "results_preserved": patched,
            "preserved_detail": preserved
        }
    except Exception as e:
        return {"error": f"Patch step failed: {e}"}

@app.get("/record-results")
async def manual_record_results(target_date: str = None, force: bool = True):
    """Manually trigger recording results for a date. force=True re-checks DNP records."""
    d = target_date or (date.today() - timedelta(days=1)).isoformat()
    await record_results(d, force=force)
    return {"status": "done", "date": d, "force": force}

@app.get("/rerecord-all")
async def rerecord_all_results():
    """Re-run record_results for all prediction files — fixes bad DNP records"""
    import json
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"https://api.github.com/repos/{GITHUB_REPO}/contents/data/predictions",
                headers={"Authorization": f"token {GITHUB_TOKEN}"}
            )
            files = r.json()
        results = []
        for f in sorted(files, key=lambda x: x["name"]):
            fname = f["name"]
            if not fname.endswith(".json"): continue
            d = fname.replace(".json", "")
            # Skip today — games not finished
            if d == date.today().isoformat(): continue
            content, _ = await github_get_file(f"data/predictions/{fname}")
            if not content: continue
            try:
                recs = json.loads(content)
                total = len(recs)
                bad_dnp = sum(1 for r in recs if r.get("hit_hr") == "DNP")
                nulls   = sum(1 for r in recs if r.get("hit_hr") is None)
                results.append({"date": d, "total": total, "dnp": bad_dnp, "null": nulls})
                # Re-record with force=True to fix bad DNP records
                if bad_dnp > 0 or nulls > 0:
                    await record_results(d, force=True)
                    print(f"Re-recorded {d} (force=True): {bad_dnp} DNP re-checked, {nulls} nulls filled")
            except Exception as e:
                results.append({"date": d, "error": str(e)})
        return {"status": "complete", "dates_processed": results}
    except Exception as e:
        return {"error": str(e)}

@app.get("/verify-results")
async def verify_results():
    """Show DNP/HR/miss breakdown for all prediction dates with tiered hit rates"""
    import json
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"https://api.github.com/repos/{GITHUB_REPO}/contents/data/predictions",
                headers={"Authorization": f"token {GITHUB_TOKEN}"}
            )
            files = r.json()
        summary = []
        # Aggregate across all dates for overall tier analysis
        all_recs = []
        for f in sorted(files, key=lambda x: x["name"], reverse=True):
            fname = f["name"]
            if not fname.endswith(".json"): continue
            d = fname.replace(".json", "")
            content, _ = await github_get_file(f"data/predictions/{fname}")
            if not content: continue
            try:
                recs = json.loads(content)
                total  = len(recs)
                hrs    = sum(1 for r in recs if r.get("hit_hr") == 1)
                misses = sum(1 for r in recs if r.get("hit_hr") == 0)
                dnps   = sum(1 for r in recs if r.get("hit_hr") == "DNP")
                nulls  = sum(1 for r in recs if r.get("hit_hr") is None)

                # Tier breakdown — only count non-DNP, non-null
                def tier_hr(recs, lo, hi):
                    t = [r for r in recs if r.get("hit_hr") in (0,1)
                         and lo <= (r.get("model_hr_pct") or 0) < hi]
                    h = sum(1 for r in t if r.get("hit_hr") == 1)
                    return h, len(t)

                h20, n20 = tier_hr(recs, 20, 999)
                h15, n15 = tier_hr(recs, 15, 20)
                h10, n10 = tier_hr(recs, 10, 15)
                h8,  n8  = tier_hr(recs, 8,  10)
                h5,  n5  = tier_hr(recs, 5,   8)

                suspicious = dnps > total * 0.3
                summary.append({
                    "date": d,
                    "total": total,
                    "hrs": hrs,
                    "misses": misses,
                    "dnp": dnps,
                    "null": nulls,
                    "overall_hr_rate": f"{round(hrs/max(hrs+misses,1)*100,1)}%",
                    "dnp_pct": f"{round(dnps/max(total,1)*100,1)}%",
                    "suspicious": suspicious,
                    "tiers": {
                        "20pct_plus":  {"hrs": h20, "total": n20, "hit_rate": f"{round(h20/max(n20,1)*100,1)}%"},
                        "15_to_20":    {"hrs": h15, "total": n15, "hit_rate": f"{round(h15/max(n15,1)*100,1)}%"},
                        "10_to_15":    {"hrs": h10, "total": n10, "hit_rate": f"{round(h10/max(n10,1)*100,1)}%"},
                        "8_to_10":     {"hrs": h8,  "total": n8,  "hit_rate": f"{round(h8/max(n8,1)*100,1)}%"},
                        "5_to_8":      {"hrs": h5,  "total": n5,  "hit_rate": f"{round(h5/max(n5,1)*100,1)}%"},
                    }
                })
                all_recs.extend([r for r in recs if r.get("hit_hr") in (0,1)])
            except Exception:
                continue

        # Overall cumulative tier analysis
        def cum_tier(recs, lo, hi):
            t = [r for r in recs if lo <= (r.get("model_hr_pct") or 0) < hi]
            h = sum(1 for r in t if r.get("hit_hr") == 1)
            return h, len(t), round(h/max(len(t),1)*100,1)

        ch20, cn20, cr20 = cum_tier(all_recs, 20, 999)
        ch15, cn15, cr15 = cum_tier(all_recs, 15, 20)
        ch10, cn10, cr10 = cum_tier(all_recs, 10, 15)
        ch8,  cn8,  cr8  = cum_tier(all_recs, 8,  10)
        ch5,  cn5,  cr5  = cum_tier(all_recs, 5,   8)

        return {
            "cumulative_tier_analysis": {
                "20pct_plus": {"hrs": ch20, "total": cn20, "hit_rate": f"{cr20}%",
                               "note": "Your top betting tier"},
                "15_to_20":  {"hrs": ch15, "total": cn15, "hit_rate": f"{cr15}%",
                               "note": "Strong plays"},
                "10_to_15":  {"hrs": ch10, "total": cn10, "hit_rate": f"{cr10}%",
                               "note": "Watchlist"},
                "8_to_10":   {"hrs": ch8,  "total": cn8,  "hit_rate": f"{cr8}%",
                               "note": "Marginal"},
                "5_to_8":    {"hrs": ch5,  "total": cn5,  "hit_rate": f"{cr5}%",
                               "note": "Data only — never bet"},
            },
            "dates": summary
        }
    except Exception as e:
        return {"error": str(e)}
async def debug_results(target_date: str = None):
    """Show what the MLB API returned for a date vs what we predicted"""
    d = target_date or (date.today() - timedelta(days=1)).isoformat()
    try:
        url = f"{MLB_API}/schedule?sportId=1&date={d}&gameType=R&hydrate=boxscore"
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url)
            sched = r.json()
        hr_hitters = {}
        all_players = {}
        for game_date in sched.get("dates", []):
            for game in game_date.get("games", []):
                if game.get("status", {}).get("abstractGameState") != "Final": continue
                gid = game["gamePk"]
                away = game["teams"]["away"]["team"]["name"]
                home = game["teams"]["home"]["team"]["name"]
                async with httpx.AsyncClient(timeout=10) as client:
                    r2 = await client.get(f"{MLB_API}/game/{gid}/boxscore")
                    box = r2.json()
                for side in ["away", "home"]:
                    for _, p in box.get("teams", {}).get(side, {}).get("players", {}).items():
                        stats = p.get("stats", {}).get("batting", {})
                        name = p.get("person", {}).get("fullName", "")
                        if not name: continue
                        ab = int(stats.get("atBats", 0) or 0)
                        hrs = int(stats.get("homeRuns", 0) or 0)
                        all_players[name] = {"ab": ab, "hr": hrs, "game": f"{away}@{home}"}
                        if hrs > 0:
                            hr_hitters[name] = hrs
        # Load predictions for that date
        path = f"data/predictions/{d}.json"
        import json as _json
        raw, _ = await github_get_file(path)
        records = _json.loads(raw) if raw else []
        pred_names = [r["name"] for r in records]
        # Match predictions to API results
        matches = []
        misses = []
        for name in pred_names:
            nl = name.lower()
            found = next((k for k in all_players if k.lower() == nl), None)
            if not found:
                last = nl.split()[-1]
                found = next((k for k in all_players if last in k.lower()), None)
            if found:
                matches.append({"predicted": name, "api_name": found, "ab": all_players[found]["ab"], "hr": all_players[found]["hr"]})
            else:
                misses.append(name)
        return {
            "date": d,
            "hr_hitters_in_api": hr_hitters,
            "predicted_players": len(pred_names),
            "matched": len(matches),
            "missed": misses,
            "matches": matches
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/model-state")
async def debug_model_state():
    """
    Show exactly what the model is doing after recalibration:
    - Which 8 stats are active and their weights
    - What data source each stat reads from
    - What changed from defaults
    - Sample sizes used in last calibration
    - Correlation values per stat
    """
    import json

    # ── Data source map — where each stat comes from ──
    DATA_SOURCES = {
        "barrel_season":     "Baseball Savant — season Statcast (barrel_batted_rate)",
        "barrel_l8d":        "Baseball Savant — rolling 8-day Statcast window",
        "la_season":         "Baseball Savant — season launch_angle_avg",
        "la_l8d":            "Baseball Savant — rolling 8-day launch angle",
        "ev_season":         "Baseball Savant — season exit_velocity_avg",
        "ev_l8d":            "Baseball Savant — rolling 8-day exit velocity",
        "iso_season":        "Baseball Savant — season SLG minus AVG",
        "iso_vs_hand":       "Baseball Savant — batter split vs pitcher hand (ISO)",
        "hard_hit_season":   "Baseball Savant — season hard_hit_percent",
        "hard_hit_l8d":      "Baseball Savant — rolling 8-day hard hit%",
        "pit_hr9_season":    "MLB Stats API — pitcher HR/9 from game logs",
        "pit_hr9_vs_hand":   "Baseball Savant — pitcher splits vs batter hand (HR/9)",
        "pit_slg_season":    "Baseball Savant — pitcher season SLG allowed",
        "pit_slg_vs_hand":   "Baseball Savant — pitcher splits vs batter hand (SLG)",
        "park":              "Static park HR factors table (hand-adjusted)",
        "weather":           "Open-Meteo API — wind speed/direction/temperature",
        "bullpen":           "MLB Stats API — team bullpen HR/9",
        "bat_platoon":       "Baseball Savant — batter ISO vs hand / ISO overall ratio",
        "pit_platoon":       "Baseball Savant — pitcher SLG vs hand / SLG overall ratio",
        "pitch_delta":       "Baseball Savant — batter + pitcher pitch run values (arsenal)",
        "k_pct":             "Baseball Savant — batter K% (penalty gate)",
        "fb_pct_season":     "Baseball Savant — batter fly ball%",
        "pull_pct_season":   "Baseball Savant — batter pull%",
        "pit_fb_pct_allowed":"Baseball Savant — pitcher fly ball% allowed",
        "hard_hit_l8d":      "Baseball Savant — rolling 8-day hard hit%",
        "k_pct_l8d":         "Baseball Savant — rolling 8-day K%",
        "pit_era_diff":      "MLB Stats API — pitcher ERA vs league avg",
        "pit_k9_season":     "MLB Stats API — pitcher K/9 season",
        "xwoba_l8d":         "Baseball Savant — rolling 8-day xwOBA",
        "xslg_l8d":          "Baseball Savant — rolling 8-day xSLG",
        "bat_speed_l8d":     "Baseball Savant — rolling 8-day bat speed",
        "slg_l8d":           "Baseball Savant — rolling 8-day SLG",
    }

    weights = _model_weights
    active_stats = weights.get("active_stats", DEFAULT_WEIGHTS["active_stats"])
    default_stats = DEFAULT_WEIGHTS["active_stats"]

    # ── Build active stat details ──
    active_details = []
    for stat in active_stats:
        w_key = stat + "_w"
        current_w = round(float(weights.get(w_key, 1.0)), 3)
        default_w = round(float(DEFAULT_WEIGHTS.get(w_key, 1.0)), 3)
        changed = abs(current_w - default_w) >= 0.05
        was_default = stat in default_stats
        source = DATA_SOURCES.get(stat, "Unknown source")
        active_details.append({
            "stat": stat,
            "weight": current_w,
            "default_weight": default_w,
            "weight_changed": changed,
            "direction": "up" if current_w > default_w else "down" if current_w < default_w else "unchanged",
            "was_in_default_model": was_default,
            "promoted": not was_default,
            "data_source": source,
        })

    # Sort by weight descending
    active_details.sort(key=lambda x: x["weight"], reverse=True)

    # ── Stats that were dropped vs defaults ──
    dropped_from_default = [s for s in default_stats if s not in active_stats]
    newly_promoted = [s for s in active_stats if s not in default_stats]

    # ── All weight changes from defaults ──
    all_changes = []
    for key in DEFAULT_WEIGHTS:
        if not key.endswith("_w"): continue
        old_w = round(float(DEFAULT_WEIGHTS[key]), 3)
        new_w = round(float(weights.get(key, 1.0)), 3)
        if abs(new_w - old_w) >= 0.05:
            stat_name = key.replace("_w", "")
            all_changes.append({
                "stat": stat_name,
                "before": old_w,
                "after": new_w,
                "delta": round(new_w - old_w, 3),
                "direction": "up" if new_w > old_w else "down",
                "data_source": DATA_SOURCES.get(stat_name, "Unknown"),
            })
    all_changes.sort(key=lambda x: abs(x["delta"]), reverse=True)

    # ── Calibration metadata ──
    meta = {
        "last_calibrated": weights.get("last_calibrated"),
        "records_used": weights.get("records_used", 0),
        "calibration_round": weights.get("calibration_round", 0),
        "promoted_stats": weights.get("promoted_stats", []),
        "dropped_stats": weights.get("dropped_stats", []),
        "recent_changes": weights.get("recent_changes", []),
    }

    # ── Sigmoid status ──
    sigmoid_status = {
        "status": "REMOVED — April 2026",
        "reason": "Scores clustered 14-47, never reaching sigmoid center of 50. All outputs compressed to 5-14% band.",
        "current_output": "direct: running * 100, floor 2%, cap 28%",
        "previous_formula": "0.02 + sigmoid((raw-50)/18) * 0.25",
    }

    return {
        "calibration_meta": meta,
        "active_model": {
            "count": len(active_details),
            "stats": active_details,
        },
        "changes_from_default": {
            "dropped": dropped_from_default,
            "promoted": newly_promoted,
            "weight_changes": all_changes,
        },
        "sigmoid": sigmoid_status,
        "calibration": {
            "round": get_rotation_round(),
            "last_calibrated": _model_weights.get("last_calibrated"),
            "records_used": _model_weights.get("records_used", 0),
            "auto_recalibrate": "Every Sunday 3am ET",
        },
        "data_source_legend": DATA_SOURCES,
    }


@app.get("/debug/scores")
async def debug_score_distribution():
    """
    Return model% distribution for today's predictions.
    Sigmoid removed April 2026 — model_hr_pct is now the direct output.
    """
    import json, statistics

    today = date.today().isoformat()
    path = f"data/predictions/{today}.json"
    content, _ = await github_get_file(path)

    if not content:
        return {"error": f"No predictions found for {today} — run /save-predictions first"}

    try:
        records = json.loads(content)
    except Exception as e:
        return {"error": f"Failed to parse predictions: {e}"}

    pcts = [r["model_hr_pct"] for r in records if r.get("model_hr_pct") is not None]
    if not pcts:
        return {"error": "No model_hr_pct values found"}

    pcts_sorted = sorted(pcts)
    n = len(pcts)

    def percentile(lst, p):
        idx = int(len(lst) * p / 100)
        return lst[min(idx, len(lst)-1)]

    mean_pct   = round(statistics.mean(pcts), 2)
    median_pct = round(statistics.median(pcts), 2)
    std_pct    = round(statistics.stdev(pcts) if n > 1 else 0, 2)

    diagnosis = []
    above_8  = len([p for p in pcts if p >= 8])
    above_10 = len([p for p in pcts if p >= 10])
    above_15 = len([p for p in pcts if p >= 15])

    if mean_pct < 6:
        diagnosis.append(f"SKEW LOW: mean={mean_pct}% — base rates too conservative, most batters below 8% threshold")
    elif mean_pct > 12:
        diagnosis.append(f"SKEW HIGH: mean={mean_pct}% — outputs may be inflated")
    else:
        diagnosis.append(f"HEALTHY: mean={mean_pct}% — reasonable distribution")

    diagnosis.append(f"Above 8%: {above_8}/{n} batters showing on board")
    diagnosis.append(f"Above 10%: {above_10}/{n} | Above 15%: {above_15}/{n}")

    player_scores = sorted([
        {"name": r.get("name"), "team": r.get("team"),
         "pitcher": r.get("opp_pitcher"), "model_pct": r.get("model_hr_pct")}
        for r in records if r.get("model_hr_pct") is not None
    ], key=lambda x: x["model_pct"], reverse=True)

    return {
        "date": today,
        "n_predictions": n,
        "distribution": {
            "min": pcts_sorted[0],
            "max": pcts_sorted[-1],
            "mean": mean_pct,
            "median": median_pct,
            "std_dev": std_pct,
            "p25": round(percentile(pcts_sorted, 25), 2),
            "p75": round(percentile(pcts_sorted, 75), 2),
            "p90": round(percentile(pcts_sorted, 90), 2),
        },
        "threshold_counts": {
            "above_5pct": len([p for p in pcts if p >= 5]),
            "above_8pct": above_8,
            "above_10pct": above_10,
            "above_15pct": above_15,
        },
        "diagnosis": diagnosis,
        "all_players": player_scores,
    }


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
        "bat_8d": len(_cache["bat_8d"]),
        "bat_l5g": len(_cache["bat_l5g"]),
        "bat_vs_lhp": len(_cache["bat_vs_lhp"]),
        "bat_vs_rhp": len(_cache["bat_vs_rhp"]),
        # Pitcher data
        "pit_2026": len(_cache["pit_2026"]),
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
        # Decision Tree status
        "tree_trained": _dt_model is not None,
        "tree_depth": _model_weights.get("tree_depth"),
        "tree_records": _model_weights.get("records_used", 0),
        "calibration_factor": _model_weights.get("calibration_factor", 1.0),
        "next_depth_upgrade": _model_weights.get("next_depth_upgrade"),
    }

@app.get("/reload-contact")
async def reload_contact_get():
    """GET endpoint to trigger contact log reload — browser friendly"""
    asyncio.create_task(reload_contact_log())
    return {"status": "Contact log reloading — check back in 60 seconds"}

@app.get("/clear-cache")
async def clear_cache():
    """Clear games cache — forces fresh rebuild on next /games call"""
    _games_cache.clear()
    # Spot check — does Yordan have contact data right now?
    test = get_contact_log("Yordan Alvarez")
    return {
        "status": "Games cache cleared",
        "contact_log_size": len(_contact_log),
        "yordan_events": len(test),
        "yordan_sample": test[0] if test else None,
    }

@app.get("/debug-contact-fetch")
async def debug_contact_fetch():
    """Test whether the contact log CSV URL is reachable from Railway"""
    url = savant_contact_log_url()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            r = await client.get(url, headers=headers, follow_redirects=True)
            text = r.text[:500] if r.text else ""
            return {
                "status_code": r.status_code,
                "content_length": len(r.text),
                "first_500_chars": text,
                "is_csv": text.startswith("pitch_type") or "player_name" in text[:200],
                "url": url,
                "deny_reason": r.headers.get("x-deny-reason", "none"),
            }
    except Exception as e:
        return {"error": str(e), "url": url}

@app.post("/reload")
async def reload_data():
    _games_cache.clear()
    threading.Thread(target=run_async, args=(load_all_savant_data(),), daemon=True).start()
    asyncio.create_task(reload_contact_log())
    return {"status": "Reloading data from Baseball Savant"}

async def reload_contact_log():
    """Reload contact log — 9MB file needs 180s timeout"""
    await asyncio.sleep(5)
    async with httpx.AsyncClient(timeout=200) as client:
        df = await fetch_savant_csv(savant_contact_log_url(), client, timeout=180)
        if not df.empty:
            _build_contact_log(df)
            _games_cache.clear()
            print(f"contact_log reloaded: {len(_contact_log)} players")
        else:
            print("contact_log reload failed — CSV empty")

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
                    "barrel": round(bc.get("barrel_pct", 0), 1),
                    "ev":     round(bc.get("exit_velo", 0), 1),
                    "la":     round(bc.get("launch_angle", 0), 1),
                    "hh":     round(bc.get("hard_hit_pct", 0), 1),
                    "iso":    round(bc.get("iso", 0), 3),
                    "slg":    round(bc.get("slg_percent", 0), 3),
                    "avg":    round(bc.get("batting_avg", 0), 3),
                    "k":      round(bc.get("k_pct", 0), 1),
                    "pull":   round(bc.get("pull_pct", 0), 1),
                    "hr":     int(bc.get("hr", 0)),
                },
                "l8d": {
                    "pa":       int(b8d.get("pa", 0)),
                    "barrel":   round(b8d.get("barrel_pct", 0), 1),
                    "ev":       round(b8d.get("exit_velo", 0), 1),
                    "la":       round(b8d.get("launch_angle", 0), 1),
                    "hh":       round(b8d.get("hard_hit_pct", 0), 1),
                    "iso":      round(b8d.get("iso", 0), 3),
                    "slg":      round(b8d.get("slg", 0), 3),
                    "avg":      round(b8d.get("avg", 0), 3),
                    "pull":     round(b8d.get("pull_pct", 0), 1),
                    "k_pct":    round(b8d.get("k_pct", 0), 1),
                    "xwoba":    round(b8d.get("xwoba", 0), 3),
                    "xslg":     round(b8d.get("xslg", 0), 3),
                    "bat_speed":round(b8d.get("bat_speed", 0), 1),
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
                "contact_log": get_contact_log(name),
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



@app.get("/coverage-check")
async def coverage_check():
    """
    Check coverage and build the clean stat list for Decision Tree.
    Distinguishes between true zeros vs missing data for each stat.
    """
    import json
    if not GITHUB_TOKEN:
        return {"error": "No GitHub token"}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/data/predictions",
                headers={"Authorization": f"token {GITHUB_TOKEN}"}
            )
            files = r.json()
    except Exception as e:
        return {"error": str(e)}

    all_records = []
    for f in sorted(files, key=lambda x: x["name"]):
        if not f["name"].endswith(".json"): continue
        content_raw, _ = await github_get_file(f"data/predictions/{f['name']}")
        if not content_raw: continue
        try:
            recs = json.loads(content_raw)
            completed = [r for r in recs if r.get("hit_hr") in [0, 1]]
            all_records.extend(completed)
        except:
            continue

    if not all_records:
        return {"error": "No completed records found"}

    total = len(all_records)
    hr_count = sum(1 for r in all_records if r.get("hit_hr") == 1)

    # Stats where zero = truly missing (player/pitcher not in Savant yet)
    ZERO_MEANS_MISSING = {
        "barrel_pct_season", "hard_hit_season", "ev_season", "iso_season",
        "la_season", "pull_pct_season",
        "barrel_pct_l8d", "hard_hit_l8d", "ev_l8d", "bat_speed_l8d",
        "xwoba_l8d", "xslg_l8d", "slg_l8d", "iso_l8d", "k_pct_l8d",
        "la_l8d", "pa_l8d",
    }

    # Stats where zero = real data (pitcher genuinely hasn't given up HR)
    # BUT only if IP companion field is > 0, otherwise still missing
    ZERO_MEANS_DOMINANT = {
        "pit_hr9_season":   "pit_ip_season",     # zero HR/9 only real if IP > 0
        "pit_hr9_vs_hand":  "pit_ip_vs_hand",    # zero HR/9 vs hand only real if IP > 5
        "pit_slg_vs_hand":  "pit_ip_vs_hand",    # zero SLG vs hand only real if IP > 5
    }

    # Stats that are always real (computed values, never truly missing)
    ALWAYS_REAL = {
        "park_factor", "weather_mult", "bullpen_hr9",
        "bat_platoon_mult", "pit_platoon_mult",
        "pitch_matchup_score",
        "pit_era_season", "pit_hard_hit_season", "pit_k9_season",
        "pit_ip_season", "pit_ip_vs_hand",
        "iso_vs_hand", "slg_vs_hand", "pa_vs_hand",
    }

    # For each record build a clean feature vector
    # filling zeros with the right value based on stat type
    def clean_record(rec):
        cleaned = {}
        cleaned["hit_hr"] = rec["hit_hr"]

        # Season averages for fallback
        pit_hr9_season = rec.get("pit_hr9_season", 0)
        pit_ip_season  = rec.get("pit_ip_season", 0)
        pit_ip_vs_hand = rec.get("pit_ip_vs_hand", 0)

        for stat in ZERO_MEANS_MISSING:
            v = rec.get(stat)
            if v is None or v == 0:
                cleaned[stat] = None  # genuinely missing
            else:
                cleaned[stat] = float(v)

        for stat, ip_field in ZERO_MEANS_DOMINANT.items():
            v = rec.get(stat)
            ip = rec.get(ip_field, 0)
            min_ip = 5 if "vs_hand" in stat else 1
            if ip >= min_ip:
                # Real data — zero means dominant, keep it
                cleaned[stat] = float(v) if v is not None else 0.0
            else:
                # No IP = no data — fall back to season rate
                fallback = pit_hr9_season if "hr9" in stat else None
                cleaned[stat] = fallback

        for stat in ALWAYS_REAL:
            v = rec.get(stat)
            cleaned[stat] = float(v) if v is not None else None

        return cleaned

    cleaned_records = [clean_record(r) for r in all_records]

    # Now check coverage of cleaned records
    ALL_STATS = list(ZERO_MEANS_MISSING | set(ZERO_MEANS_DOMINANT.keys()) | ALWAYS_REAL)

    coverage = {}
    for stat in ALL_STATS:
        real = [r for r in cleaned_records if r.get(stat) is not None]
        pct = round(len(real) / total * 100, 1)
        hr_rate = round(
            sum(1 for r in real if r.get("hit_hr") == 1) / max(len(real), 1) * 100, 1
        )
        coverage[stat] = {
            "coverage_pct": pct,
            "real_records": len(real),
            "hr_rate": hr_rate,
            "zero_treatment": (
                "dominant_if_ip>0" if stat in ZERO_MEANS_DOMINANT
                else "missing" if stat in ZERO_MEANS_MISSING
                else "always_real"
            ),
            "usable": pct >= 60
        }

    usable   = sorted([s for s, v in coverage.items() if v["usable"]],
                      key=lambda s: coverage[s]["coverage_pct"], reverse=True)
    unusable = [s for s, v in coverage.items() if not v["usable"]]

    return {
        "total_records": total,
        "hr_records": hr_count,
        "hr_rate": round(hr_count / total * 100, 1),
        "usable_for_tree": usable,
        "unusable_drop": unusable,
        "coverage_detail": dict(sorted(
            coverage.items(),
            key=lambda x: x[1]["coverage_pct"], reverse=True
        ))
    }


@app.get("/debug-score")
async def debug_score(batter: str, pitcher: str = "TBD", bat_hand: str = None, pit_hand: str = "R", home_team: str = ""):
    """
    Show exactly how batter score and matchup score are computed for a player.
    Example: /debug-score?batter=Munetaka+Murakami&pitcher=Michael+Soroka&pit_hand=R
    """
    if not _cache["ready"]:
        return {"error": "Cache not ready"}

    # Resolve bat hand if not provided
    if not bat_hand:
        cached = _cache.get("player_hands", {}).get(batter.lower().strip(), {})
        bat_hand = cached.get("bat_side", "R")
        if bat_hand == "S": bat_hand = "L" if pit_hand == "R" else "R"

    park_factor = get_park_hr_factor(home_team or "Unknown", bat_hand)
    wx_mult = 1.0

    hr_prob, breakdown, archetype, trend, reasons, platoon_tag, conf = compute_hr_probability(
        batter, bat_hand, pitcher, pit_hand, park_factor, wx_mult, home_team
    )

    sm = breakdown.get("stat_mults", {})
    active_stats = breakdown.get("active_stats", [])

    # Pool definitions (mirrors scoring function)
    BATTER_POOL = {
        "barrel_season", "hard_hit_season", "ev_season", "iso_season",
        "la_season", "pull_pct_season", "fb_pct_season", "chase_rate_season",
        "barrel_l8d", "hard_hit_l8d", "ev_l8d", "bat_speed_l8d",
        "xwoba_l8d", "xslg_l8d", "slg_l8d", "xslg_gap_l8d",
        "iso_l8d", "k_pct_l8d", "la_l8d",
    }
    MATCHUP_POOL = {
        "park", "weather", "iso_vs_hand", "bat_platoon", "pit_platoon",
        "bullpen", "pitch_delta",
        "pit_hr9_season", "pit_hr9_vs_hand", "pit_slg_vs_hand",
        "pit_hard_hit_season", "pit_era_diff", "pit_k9_season",
        "pit_slg_season", "pit_fb_pct", "pit_stuff_plus",
    }

    # Build per-stat breakdown
    # stat_mults stores individual stats OR group keys (group_{stat}) for correlated groups
    # Need to check both to find the right multiplier
    batter_stats = []
    matchup_stats = []

    for stat in active_stats:
        # Check direct key first, then group key, then raw_mults via group scan
        mult = sm.get(stat)
        if mult is None:
            # Check if this stat was absorbed into a correlated group
            # Groups are stored as group_{first_stat_in_group}
            for key, val in sm.items():
                if key.startswith("group_") and key != "group_batter_score" and key != "group_matchup_score":
                    mult = val
                    break
            if mult is None:
                mult = 1.0
        entry = {
            "stat": stat,
            "multiplier": round(float(mult), 3),
            "direction": "boost" if float(mult) > 1.05 else "suppress" if float(mult) < 0.95 else "neutral"
        }
        if stat in BATTER_POOL:
            batter_stats.append(entry)
        elif stat in MATCHUP_POOL:
            matchup_stats.append(entry)

    # Get actual scores computed by the model
    batter_score = sm.get("batter_score") or breakdown.get("batter_score") or 1.0
    matchup_score = sm.get("matchup_score") or breakdown.get("matchup_score") or 1.0

    # Always-on multipliers (not in active_stats)
    always_on = {
        "park_factor": round(breakdown.get("park_factor", 1.0), 3),
        "weather_mult": round(breakdown.get("weather_mult", 1.0), 3),
        "k_penalty": round(breakdown.get("k_mult", 1.0), 3),
        "pit_platoon": round(breakdown.get("pit_platoon_mult", 1.0), 3),
        "bat_platoon": round(breakdown.get("bat_platoon_mult", 1.0), 3),
    }

    base = breakdown.get("base_rate", 0)
    k_pen = always_on["k_penalty"]

    return {
        "batter": batter,
        "pitcher": pitcher,
        "bat_hand": bat_hand,
        "pit_hand": pit_hand,
        "hr_prob": round(hr_prob, 1),
        "batter_score": {
            "total": round(float(batter_score), 3),
            "active_stats": batter_stats,
            "formula": " × ".join([f"{s['stat']} {s['multiplier']}x" for s in batter_stats])
        },
        "matchup_score": {
            "total": round(float(matchup_score), 3),
            "active_stats": matchup_stats,
            "formula": " × ".join([f"{s['stat']} {s['multiplier']}x" for s in matchup_stats])
        },
        "always_on_multipliers": always_on,
        "active_stats_all": active_stats,
        "final_math": {
            "base_rate": round(base, 2),
            "× batter_score": round(float(batter_score), 3),
            "× matchup_score": round(float(matchup_score), 3),
            "× k_penalty": round(k_pen, 3),
            "× park": always_on["park_factor"],
            "× weather": always_on["weather_mult"],
            "= expected": round(base * float(batter_score) * float(matchup_score) * k_pen, 1),
            "= actual_hr_prob": round(hr_prob, 1),
            "note": "expected vs actual differ if sigmoid or always-on caps applied"
        }
    }

@app.get("/research")
async def research(player: str, date: str = None):
    from datetime import date as date_cls
    today = date if date else date_cls.today().isoformat()
    date = None  # clear to avoid shadowing
    if not _cache["ready"]:
        return {"error": "Data loading — try again in 30 seconds"}

    bc = get_batter_stats(player, 2026)
    b8d = get_batter_8d(player)
    bl5g = get_batter_l5g(player)
    pa_26 = bc.get("pa", 0)
    has_8d = b8d.get("pa", 0) >= 3
    w_s, w_8 = (0.70, 0.30) if has_8d else (1.0, 0.0)

    def blend3(s26, d8):
        return round(s26 * w_s + d8 * w_8, 3) if (has_8d and d8 > 0) else round(s26, 3)

    stats = {
        "name": player, "pa_2026": pa_26,
        "blend_note": f"2026 season ({pa_26} PA)" + (" + L8D" if has_8d else ""),
        "season_2026": bc,
        "last_8d": b8d,
        "last_5g": bl5g,
        "blended": {
            "barrel_pct":   blend3(bc.get("barrel_pct", 0), b8d.get("barrel_pct", 0)),
            "iso":          blend3(bc.get("iso", 0), b8d.get("iso", 0)),
            "pull_pct":     blend3(bc.get("pull_pct", 0), b8d.get("pull_pct", 0)),
            "launch_angle": blend3(bc.get("launch_angle", 0), b8d.get("launch_angle", 0)),
            "hard_hit_pct": blend3(bc.get("hard_hit_pct", 0), b8d.get("hard_hit_pct", 0)),
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
                    ip_26 = pc.get("ip", 0)
                    pitch_bonus, pitch_details = compute_pitch_matchup(opp_p_name, player)
                    matchup = {
                        "home_team": game["teams"]["home"]["team"]["name"],
                        "away_team": game["teams"]["away"]["team"]["name"],
                        "game_time": game.get("gameDate", ""),
                        "pitcher_name": opp_p_name,
                        "pitcher_hand": opp_p_hand,
                        "pitcher_stats": {
                            "era": round(pc.get("era", 0), 2),
                            "hr9": round(pc.get("hr9", 0), 2),
                            "hard_hit_pct": round(pc.get("hard_hit_pct", 0), 1),
                            "barrel_pct_allowed": round(pc.get("barrel_pct_allowed", 0), 1),
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
