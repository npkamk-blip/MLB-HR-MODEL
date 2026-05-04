# main.py — MLB HR Model by Nick
# Version: 2026-05-04
# Model: Random Forest (adaptive depth/trees by record count)
# Features: RF adaptive params, save_parlay_combinations, record_parlay_results,
#           /refresh-8d, /debug-arsenal, /parlay-results, /version, bullpen_w_blend fix
# DO NOT overwrite with an older file — fetch from GitHub before editing.

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
import json
import statistics
import itertools
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

# ── Random Forest model globals ──
_rf_model    = None   # trained sklearn RandomForestClassifier
_rf_features = []     # ordered feature list used at training time
_rf_medians  = {}     # per-feature medians for missing value imputation
_rf_trained  = False  # True once model is fitted and ready

# ── XGBoost model globals ──
_xgb_model    = None  # trained XGBClassifier
_xgb_features = []    # feature list (same as RF + day_of_season)
_xgb_medians  = {}    # per-feature medians
_xgb_trained  = False # True once fitted
_xgb_oob      = 0.0   # cross-val score for comparison vs RF

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

async def recalibrate_model(save_to_github: bool = True):
    """
    Train Random Forest on all completed prediction records.
    Locked params: 200 estimators, max_depth=5, min_samples_leaf=10,
    max_features='sqrt', bootstrap=True, random_state=42, n_jobs=-1.
    Missing values filled with per-feature medians.
    Feature importances saved to model_weights.json for frontend display.
    """
    global _model_weights, _rf_model, _rf_features, _rf_medians, _rf_trained
    import json

    # ── Load all prediction records from GitHub ──
    all_records = []
    try:
        if not GITHUB_TOKEN: return {"error": "No GitHub token"}
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(
                f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/data/predictions",
                headers={"Authorization": f"token {GITHUB_TOKEN}"}
            )
            files = r.json() if r.is_success else []
        for f in files:
            if not f.get("name", "").endswith(".json"): continue
            content, _ = await github_get_file(f"data/predictions/{f['name']}")
            if content:
                try:
                    recs = json.loads(content)
                    all_records.extend(recs)
                except: pass
    except Exception as e:
        print(f"RF data load error: {e}")
        return {"error": str(e)}

    # ── Filter to completed records only ──
    completed = [r for r in all_records if r.get("hit_hr") in [0, 1]]
    n = len(completed)
    if n < 50:
        return {"error": f"Not enough data — need 50+, have {n}"}

    print(f"Training Random Forest on {n} completed records")

    # ── Feature set — all numeric fields stored in prediction records ──
    FEATURES = [
        "barrel_pct_season", "barrel_pct_l8d",
        "la_season", "la_l8d",
        "ev_season", "ev_l8d",
        "iso_season", "iso_vs_hand",
        "hard_hit_season", "hard_hit_l8d",
        "k_pct_season", "k_pct_l8d",
        "fb_pct_season", "pull_pct_season",
        "pit_hr9_season", "pit_hr9_vs_hand",
        "pit_hard_hit_season", "pit_era_season",
        "pit_k9_season", "pit_era_diff",
        "pit_slg_vs_hand", "pit_fb_pct_allowed",
        "park_factor", "weather_mult",
        "bat_platoon_mult", "pit_platoon_mult",
        "bullpen_vuln", "pitch_matchup_score",
        "combined_pitch_delta", "xslg_l8d",
        "xwoba_l8d", "xslg_gap_l8d",
        "bat_speed_l8d",
    ]

    # ── Build X, y — impute missing with per-feature median ──
    import statistics
    medians = {}
    for feat in FEATURES:
        vals = [float(r[feat]) for r in completed if r.get(feat) not in (None, 0, "") and r.get(feat) == r.get(feat)]
        medians[feat] = statistics.median(vals) if vals else 0.0

    def build_row(rec):
        return [float(rec.get(feat) or medians.get(feat, 0.0)) for feat in FEATURES]

    X = [build_row(r) for r in completed]
    y = [int(r["hit_hr"]) for r in completed]

    # ── RF params — scale with record count, never hardcoded ──
    # More data = deeper trees + smaller leaves = model learns more nuance
    # n_estimators grows too: more records benefit from more trees
    if n < 200:
        max_depth, min_leaf, n_trees = 4, 15, 100
    elif n < 500:
        max_depth, min_leaf, n_trees = 5, 12, 150
    elif n < 1000:
        max_depth, min_leaf, n_trees = 6, 10, 200
    elif n < 2500:
        max_depth, min_leaf, n_trees = 7, 8,  250
    elif n < 5000:
        max_depth, min_leaf, n_trees = 8, 6,  300
    elif n < 10000:
        max_depth, min_leaf, n_trees = 10, 4, 400
    else:
        max_depth, min_leaf, n_trees = 12, 3, 500

    print(f"RF params: n={n} → depth={max_depth}, min_leaf={min_leaf}, trees={n_trees}")

    # ── Train Random Forest ──
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return {"error": "sklearn not installed — add scikit-learn to requirements.txt"}

    rf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        max_features="sqrt",   # core RF principle — keeps trees decorrelated
        bootstrap=True,
        random_state=42,       # reproducibility only
        n_jobs=-1,             # use all CPU cores
    )
    rf.fit(X, y)

    # ── Feature importances ──
    importances = {feat: round(float(imp), 4)
                   for feat, imp in zip(FEATURES, rf.feature_importances_)}
    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    # ── OOB score (free accuracy estimate using held-out samples) ──
    rf_oob = RandomForestClassifier(
        n_estimators=n_trees, max_depth=max_depth, min_samples_leaf=min_leaf,
        max_features="sqrt", bootstrap=True, oob_score=True,
        random_state=42, n_jobs=-1,
    )
    rf_oob.fit(X, y)
    oob_score = round(float(rf_oob.oob_score_), 4)

    # ── HR rate in training data (calibration reference) ──
    hr_rate = round(sum(y) / len(y) * 100, 2)

    # ── Persist model state ──
    _rf_model    = rf
    _rf_features = FEATURES
    _rf_medians  = medians
    _rf_trained  = True

    # ── Save weights/metadata to GitHub ──
    new_weights = _model_weights.copy()
    new_weights["last_calibrated"]   = date.today().isoformat()
    new_weights["records_used"]      = n
    new_weights["calibration_round"] = get_rotation_round()
    new_weights["model_type"]        = "random_forest"
    new_weights["oob_score"]         = oob_score
    new_weights["hr_rate"]           = hr_rate
    new_weights["feature_importances"] = dict(ranked)
    new_weights["top_features"]      = [k for k, _ in ranked[:8]]
    new_weights["recent_changes"]    = [f"{k}: {v:.4f}" for k, v in ranked[:10]]
    # Save actual params used — so frontend/status always shows what's running
    new_weights["rf_params"] = {
        "n_estimators": n_trees,
        "max_depth":    max_depth,
        "min_samples_leaf": min_leaf,
        "max_features": "sqrt",
        "records_used": n,
    }
    _model_weights = new_weights

    if save_to_github:
        await save_model_weights(new_weights)
        await save_model_log(new_weights)
    else:
        print("RF trained (startup — skipping GitHub write to prevent deploy loop)")

    print(f"RF trained — {n} records, OOB={oob_score:.3f}, HR rate={hr_rate}%")
    print(f"Top features: {[k for k,_ in ranked[:5]]}")
    return {
        "status":       "done",
        "records_used": n,
        "oob_score":    oob_score,
        "hr_rate":      hr_rate,
        "top_features": [k for k, _ in ranked[:8]],
        "feature_importances": dict(ranked[:10]),
    }


async def startup_train_rf():
    """Train RF on startup — no GitHub write so we don't trigger infinite redeploy loop."""
    global _rf_trained
    await asyncio.sleep(20)  # let Savant data load first
    try:
        print("Startup: training Random Forest from saved records...")
        result = await recalibrate_model(save_to_github=False)
        if isinstance(result, dict) and result.get("status") == "done":
            print(f"Startup RF trained — {result.get('records_used')} records, "
                  f"OOB={result.get('oob_score')}, top={result.get('top_features',[])[0] if result.get('top_features') else '?'}")
        else:
            print(f"Startup RF result: {result}")
    except Exception as e:
        print(f"Startup RF error (non-fatal): {e}")


async def train_xgboost(save_to_github: bool = True):
    """
    Train XGBoost in parallel with RF — same records, same features + day_of_season.
    Runs silently. Does not affect predictions until it outperforms RF.
    Uses cross-validation score (not OOB) for honest comparison vs RF.
    """
    global _xgb_model, _xgb_features, _xgb_medians, _xgb_trained, _xgb_oob
    import json

    # ── Load records (same as RF) ──
    all_records = []
    try:
        if not GITHUB_TOKEN: return {"error": "No GitHub token"}
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(
                f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/data/predictions",
                headers={"Authorization": f"token {GITHUB_TOKEN}"}
            )
            files = r.json() if r.is_success else []
        for f in files:
            if not f.get("name", "").endswith(".json"): continue
            content, _ = await github_get_file(f"data/predictions/{f['name']}")
            if content:
                try: all_records.extend(json.loads(content))
                except: pass
    except Exception as e:
        return {"error": str(e)}

    completed = [r for r in all_records if r.get("hit_hr") in [0, 1]]
    n = len(completed)
    if n < 50:
        return {"error": f"Not enough data — need 50+, have {n}"}

    # ── Feature set — RF features + day_of_season ──
    FEATURES = [
        "barrel_pct_season", "barrel_pct_l8d",
        "la_season", "la_l8d",
        "ev_season", "ev_l8d",
        "iso_season", "iso_vs_hand",
        "hard_hit_season", "hard_hit_l8d",
        "k_pct_season", "k_pct_l8d",
        "fb_pct_season", "pull_pct_season",
        "pit_hr9_season", "pit_hr9_vs_hand",
        "pit_hard_hit_season", "pit_era_season",
        "pit_k9_season", "pit_era_diff",
        "pit_slg_vs_hand", "pit_fb_pct_allowed",
        "park_factor", "weather_mult",
        "bat_platoon_mult", "pit_platoon_mult",
        "bullpen_vuln", "pitch_matchup_score",
        "combined_pitch_delta", "xslg_l8d",
        "xwoba_l8d", "xslg_gap_l8d",
        "bat_speed_l8d",
        "day_of_season",   # XGBoost-specific — captures seasonal patterns
    ]

    import statistics
    medians = {}
    for feat in FEATURES:
        vals = [float(r[feat]) for r in completed
                if r.get(feat) not in (None, "", 0) and r.get(feat) == r.get(feat)]
        medians[feat] = statistics.median(vals) if vals else 0.0

    def build_row(rec):
        return [float(rec.get(feat) or medians.get(feat, 0.0)) for feat in FEATURES]

    X = [build_row(r) for r in completed]
    y = [int(r["hit_hr"]) for r in completed]

    # ── Train XGBoost ──
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return {"error": "xgboost not installed — add xgboost to requirements.txt"}

    # scale_pos_weight handles class imbalance properly
    # = count(negative) / count(positive)
    n_pos = sum(y)
    n_neg = n - n_pos
    spw   = round(n_neg / max(n_pos, 1), 2)

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X, y)

    # ── Cross-val score for honest RF comparison ──
    try:
        from sklearn.model_selection import cross_val_score
        import numpy as np
        cv_scores = cross_val_score(xgb, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
        xgb_cv = round(float(np.mean(cv_scores)), 4)
    except Exception:
        xgb_cv = 0.0

    # ── Feature importances ──
    importances = {feat: round(float(imp), 4)
                   for feat, imp in zip(FEATURES, xgb.feature_importances_)}
    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    _xgb_model    = xgb
    _xgb_features = FEATURES
    _xgb_medians  = medians
    _xgb_trained  = True
    _xgb_oob      = xgb_cv

    print(f"XGBoost trained — {n} records, CV AUC={xgb_cv:.3f}, "
          f"scale_pos_weight={spw}, top={ranked[0][0] if ranked else '?'}")

    if save_to_github:
        # Save XGBoost metadata alongside RF weights
        import json as _json
        xgb_meta = {
            "model_type": "xgboost",
            "last_trained": date.today().isoformat(),
            "records_used": n,
            "cv_auc": xgb_cv,
            "scale_pos_weight": spw,
            "top_features": [k for k, _ in ranked[:8]],
            "feature_importances": dict(ranked),
        }
        existing, sha = await github_get_file("data/xgb_meta.json")
        await github_put_file("data/xgb_meta.json",
                              _json.dumps(xgb_meta, indent=2),
                              f"xgb: {n} records, AUC={xgb_cv}", sha)

    return {
        "status":       "done",
        "records_used": n,
        "cv_auc":       xgb_cv,
        "rf_oob":       _model_weights.get("oob_score", 0),
        "scale_pos_weight": spw,
        "top_features": [k for k, _ in ranked[:8]],
        "winning_model": "xgboost" if xgb_cv > _model_weights.get("oob_score", 0) else "random_forest",
    }


async def startup_train_xgb():
    """Train XGBoost on startup — silent, no GitHub write, doesn't affect predictions yet."""
    await asyncio.sleep(45)  # after RF finishes
    try:
        print("Startup: training XGBoost silently...")
        result = await train_xgboost(save_to_github=False)
        if isinstance(result, dict) and result.get("status") == "done":
            print(f"Startup XGBoost trained — CV AUC={result.get('cv_auc')}, "
                  f"top={result.get('top_features',[])[0] if result.get('top_features') else '?'}, "
                  f"winning={result.get('winning_model')}")
        else:
            print(f"Startup XGBoost result: {result}")
    except Exception as e:
        print(f"Startup XGBoost error (non-fatal): {e}")


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
],
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

def savant_contact_log_url():
    """Pitch-by-pitch URL — only batted ball contact events, minimal columns for speed."""
    cutoff = (date.today() - timedelta(days=8)).isoformat()
    today_str = (date.today() + timedelta(days=1)).isoformat()
    # hfAB=54 filters to balls in play only (no strikes/balls) — much smaller dataset
    # Only request columns we need for contact log
    # hfAB=54 = "in_play" filter — returns only batted ball events (~8k rows vs 25k)
    # This reduces download from 16MB to ~5MB, preventing Railway timeout
    return (f"{SAVANT_BASE}/statcast_search/csv?all=true"
            f"&hfPT=&hfAB=54%7C&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull="
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
    # cf_bearing = compass degrees from home plate toward CF (verified Andrew Clem database)
    # hr_bearing_R = LF bearing (RHB pull) = (cf+270)%360
    # hr_bearing_L = RF bearing (LHB pull) = (cf+90)%360
    # open_factor = wind exposure (1.0=Wrigley fully open)
    "Arizona Diamondbacks":  {"lat":33.4453,"lon":-112.0667,"dome":True},
    "Atlanta Braves":        {"lat":33.8907,"lon":-84.4677, "dome":False,"cf_bearing":45, "hr_bearing_R":315,"hr_bearing_L":135,"open_factor":0.5},  # NE
    "Baltimore Orioles":     {"lat":39.2838,"lon":-76.6217, "dome":False,"cf_bearing":67, "hr_bearing_R":337,"hr_bearing_L":157,"open_factor":0.6},  # ENE
    "Boston Red Sox":        {"lat":42.3467,"lon":-71.0972, "dome":False,"cf_bearing":45, "hr_bearing_R":315,"hr_bearing_L":135,"open_factor":0.8},  # NE
    "Chicago Cubs":          {"lat":41.9484,"lon":-87.6553, "dome":False,"cf_bearing":45, "hr_bearing_R":315,"hr_bearing_L":135,"open_factor":1.0},  # NE
    "Chicago White Sox":     {"lat":41.8299,"lon":-87.6338, "dome":False,"cf_bearing":135,"hr_bearing_R":45, "hr_bearing_L":225,"open_factor":0.5},  # SE inverted
    "Cincinnati Reds":       {"lat":39.0979,"lon":-84.5082, "dome":False,"cf_bearing":67, "hr_bearing_R":337,"hr_bearing_L":157,"open_factor":0.6},  # ENE
    "Cleveland Guardians":   {"lat":41.4954,"lon":-81.6854, "dome":False,"cf_bearing":67, "hr_bearing_R":337,"hr_bearing_L":157,"open_factor":0.6},  # ENE
    "Colorado Rockies":      {"lat":39.7559,"lon":-104.9942,"dome":False,"cf_bearing":67, "hr_bearing_R":337,"hr_bearing_L":157,"open_factor":0.8},  # ENE
    "Detroit Tigers":        {"lat":42.3390,"lon":-83.0485, "dome":False,"cf_bearing":135,"hr_bearing_R":45, "hr_bearing_L":225,"open_factor":0.5},  # SE inverted
    "Houston Astros":        {"lat":29.7573,"lon":-95.3555, "dome":True},
    "Kansas City Royals":    {"lat":39.0517,"lon":-94.4803, "dome":False,"cf_bearing":45, "hr_bearing_R":315,"hr_bearing_L":135,"open_factor":0.7},  # NE
    "Los Angeles Angels":    {"lat":33.8003,"lon":-117.8827,"dome":False,"cf_bearing":45, "hr_bearing_R":315,"hr_bearing_L":135,"open_factor":0.5},  # NE
    "Los Angeles Dodgers":   {"lat":34.0739,"lon":-118.2400,"dome":False,"cf_bearing":22, "hr_bearing_R":292,"hr_bearing_L":112,"open_factor":0.5},  # NNE
    "Miami Marlins":         {"lat":25.7781,"lon":-80.2197, "dome":True},
    "Milwaukee Brewers":     {"lat":43.0282,"lon":-87.9712, "dome":True},
    "Minnesota Twins":       {"lat":44.9817,"lon":-93.2778, "dome":False,"cf_bearing":67, "hr_bearing_R":337,"hr_bearing_L":157,"open_factor":0.6},  # ENE
    "New York Mets":         {"lat":40.7571,"lon":-73.8458, "dome":False,"cf_bearing":67, "hr_bearing_R":337,"hr_bearing_L":157,"open_factor":0.5},  # ENE
    "New York Yankees":      {"lat":40.8296,"lon":-73.9262, "dome":False,"cf_bearing":90, "hr_bearing_R":0,  "hr_bearing_L":180,"open_factor":0.6},  # E
    "Oakland Athletics":     {"lat":38.5726,"lon":-121.5088,"dome":False,"cf_bearing":45, "hr_bearing_R":315,"hr_bearing_L":135,"open_factor":0.5},  # NE
    "Philadelphia Phillies": {"lat":39.9056,"lon":-75.1665, "dome":False,"cf_bearing":67, "hr_bearing_R":337,"hr_bearing_L":157,"open_factor":0.5},  # ENE
    "Pittsburgh Pirates":    {"lat":40.4469,"lon":-80.0057, "dome":False,"cf_bearing":67, "hr_bearing_R":337,"hr_bearing_L":157,"open_factor":0.7},  # ENE
    "San Diego Padres":      {"lat":32.7076,"lon":-117.1570,"dome":False,"cf_bearing":315,"hr_bearing_R":225,"hr_bearing_L":45, "open_factor":0.8},  # NW faces downtown
    "San Francisco Giants":  {"lat":37.7786,"lon":-122.3893,"dome":False,"cf_bearing":90, "hr_bearing_R":0,  "hr_bearing_L":180,"open_factor":0.9},  # E bay wind blows IN
    "Seattle Mariners":      {"lat":47.5914,"lon":-122.3325,"dome":True},
    "St. Louis Cardinals":   {"lat":38.6226,"lon":-90.1928, "dome":False,"cf_bearing":67, "hr_bearing_R":337,"hr_bearing_L":157,"open_factor":0.5},  # ENE
    "Tampa Bay Rays":        {"lat":27.7683,"lon":-82.6534, "dome":True},
    "Texas Rangers":         {"lat":32.7473,"lon":-97.0825, "dome":True},
    "Toronto Blue Jays":     {"lat":43.6414,"lon":-79.3894, "dome":True},
    "Washington Nationals":  {"lat":38.8730,"lon":-77.0074, "dome":False,"cf_bearing":67, "hr_bearing_R":337,"hr_bearing_L":157,"open_factor":0.5},  # ENE
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

        # Batter 2025
        # bat_2025 removed — 2026-only model

        # Batter 8d — aggregated stats per player (group_by=name)
        df = await fetch_savant_csv(savant_8d_url(), client)
        if not df.empty:
            _cache["bat_8d"] = calc_statcast_8d(df)
            print(f"bat_8d: {len(_cache['bat_8d'])} rows")
        else:
            print("bat_8d: 0 rows")

        # Contact log fetched separately in refresh_8d to avoid startup timeout

        # Pitcher 2026
        df = await fetch_savant_csv(savant_pitcher_url(min_pa=5), client)
        if not df.empty:
            _cache["pit_2026"] = calc_pitcher_stats(df)
            print(f"pit_2026: {len(_cache['pit_2026'])} rows")

        # Pitcher 2025
        # pit_2025 removed — 2026-only model

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
    async with httpx.AsyncClient(timeout=60) as client:
        df = await fetch_savant_csv(savant_8d_url(), client)
        if not df.empty:
            agg = calc_statcast_8d(df)
            if not agg.empty:
                _cache["bat_8d"] = agg
                print(f"bat_8d refreshed: {len(agg)} players")
        await asyncio.sleep(2)
        df_contact = await fetch_savant_csv(savant_contact_log_url(), client)
        if not df_contact.empty:
            _build_contact_log(df_contact)
            print(f"contact_log refreshed: {len(_contact_log)} players")
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
    """Run in background — check every hour for scheduled tasks"""
    while True:
        now = datetime.now()
        await asyncio.sleep(3600)  # check every hour
        if now.hour == 7:
            try:
                await load_all_savant_data()
            except Exception as e:
                print(f"Daily refresh error: {e}")
        # Save predictions at 1pm ET (18:00 UTC) — lineups mostly confirmed
        if now.hour == 11:
            try:
                await save_daily_predictions()
                await save_parlay_combinations()
            except Exception as e:
                print(f"Prediction save error: {e}")
        # Record results at 2am ET — all games finished
        if now.hour == 2:
            try:
                yesterday = (date.today() - timedelta(days=1)).isoformat()
                await record_results(yesterday)
                await record_parlay_results(yesterday)
            except Exception as e:
                print(f"Result recording error: {e}")
        # Retrain RF + XGBoost at 3am ET daily (after results are in)
        if now.hour == 3:
            try:
                print(f"Nightly retrain — Round {get_rotation_round()} Day {get_rotation_day()}")
                await recalibrate_model()
                await train_xgboost()
            except Exception as e:
                print(f"Nightly retrain error: {e}")
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
                        pa26 = bc2.get("pa", 0); pa25 = 0
                        bwc2 = get_batter_blend_weights(pa26, pa25)
                        barrel_s = round(blend(bc2.get("barrel_pct",0), 0, bwc2), 1)
                        la_s     = round(blend(bc2.get("launch_angle",0), 0, bwc2), 1)
                        ev_s     = round(blend(bc2.get("exit_velo",0), 0, bwc2), 1)
                        iso_s    = round(blend(bc2.get("iso",0), 0, bwc2), 3)
                        hh_s     = round(blend(bc2.get("hard_hit_pct",0), 0, bwc2), 1)
                        k_s      = round(blend(bc2.get("k_pct",0), 0, bwc2), 1)
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
                        # Raw pitcher stats
                        pc2  = get_pitcher_stats(opp_p_name, 2026)
                        ip26 = pc2.get("ip",0)
                        pwc2 = get_pitcher_blend_weights(ip26, 0)
                        pit_hr9_s   = round(blend(pc2.get("hr9",0), 0), 2)
                        pit_era_s   = round(blend(pc2.get("era",0), 0), 2)
                        pit_hh_s    = round(blend(pc2.get("hard_hit_pct",0), 0), 1)
                        pit_k9_s    = round(blend(pc2.get("k9",0), 0), 1)
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
                            "fb_pct_season": round(blend(bc2.get("fb_pct",0), 0, bwc2), 1),
                            "pull_pct_season": round(blend(bc2.get("pull_pct",0), 0, bwc2), 1),
                            "pit_fb_pct_allowed": round(blend(pc2.get("fb_pct",0), 0), 1),
                            "hard_hit_l8d": hh_l8d,
                            "k_pct_l8d": round(b8d2.get("k_pct",0), 1),
                            # ── ROUND 3 CANDIDATES ──
                            "pit_era_season": pit_era_s,
                            "pit_era_diff": round(pit_era_s - 4.20, 2) if pit_era_s > 0 else 0,
                            "pit_hr_fb_pct": round(blend(pc2.get("hr_fb_pct",0), 0), 1),
                            "lineup_k_pct": 0,  # populated at game time in future
                            # ── ROUND 4 CANDIDATES ──
                            "pit_k9_season": pit_k9_s,
                            # ── XGBOOST FEATURES ──
                            "day_of_season": (date.today() - date(2026, 3, 20)).days,
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
                    rec["hit_hr"] = "DNP"
                    rec["actual_ab"] = ab
                    dnp_count += 1
                else:
                    # Exact match first, then partial last name match for foreign players
                    hit = nl in hr_hitters
                    if not hit:
                        last = nl.split()[-1]
                        hit = any(last in k for k in hr_hitters)
                    rec["hit_hr"] = 1 if hit else 0
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
    asyncio.create_task(startup_catchup())
    asyncio.create_task(startup_train_rf())
    asyncio.create_task(startup_train_xgb())

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
    """2026-only. year param kept for call-site compatibility."""
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
    """2026-only. year param kept for call-site compatibility."""
    df = _cache["pit_2026"]
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
def blend(v1, v2, w1=1.0, w2=0.0):
    """Returns v1 only — 2025 data removed, w2 always 0."""
    return float(v1 or 0)

def get_batter_blend_weights(pa_2026, pa_2025=0):
    """2026-only. pa_2025 kept as unused param for call-site compatibility."""
    return 1.0, 0.0

def get_pitcher_blend_weights(ip_2026, ip_2025=0):
    """2026-only. ip_2025 kept as unused param for call-site compatibility."""
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
    # Direction label — uses cf_bearing for precise field direction labels
    cf_bearing = stadium.get("cf_bearing", 67)  # default ENE
    hr_bear_r  = stadium.get("hr_bearing_R", (cf_bearing + 270) % 360)
    hr_bear_l  = stadium.get("hr_bearing_L", (cf_bearing + 90)  % 360)

    if wind_speed < 5:
        direction_label = "Calm"
    else:
        diff_cf = abs(angle_diff(wind_toward, cf_bearing))
        diff_lf = abs(angle_diff(wind_toward, hr_bear_r))
        diff_rf = abs(angle_diff(wind_toward, hr_bear_l))
        if alignment > 0.5:
            if diff_cf <= 25:       direction_label = "Out to CF"
            elif diff_lf < diff_rf: direction_label = "Out to LF"
            else:                   direction_label = "Out to RF"
        elif alignment > 0.15:
            if diff_cf <= 35:       direction_label = "Blowing Out"
            elif diff_lf < diff_rf: direction_label = "Toward LF"
            else:                   direction_label = "Toward RF"
        elif alignment < -0.5:      direction_label = "Blowing In"
        elif alignment < -0.15:     direction_label = "Slightly In"
        else:                       direction_label = "Crosswind"
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
    b8d = get_batter_8d(name)
    b_split_vs_hand = get_batter_split(name, opp_p_hand)   # batter vs pitcher hand
    b_split_opp     = get_batter_split(name, "R" if opp_p_hand == "L" else "L")  # vs opposite hand
    p_split_vs_bat  = get_pitcher_split(opp_p_name, bat_hand)  # pitcher vs batter hand

    pa_26 = bc.get("pa", 0); pa_25 = 0
    bwc = 1.0
    # For brand new MLB players (under 60 PA, no 2025 data), Savant's game_date_gt
    # filter sometimes returns full season stats instead of true L8D window.
    # If L8D PA matches season PA exactly, it's bogus — clear it.

    has_8d = b8d.get("pa", 0) >= 3
    total_pa = pa_26

    # ── Step 1: Base HR rate (per-PA, relative ranking model) ──
    # base_rate = HR/PA blended between 2026 season and 2025 career
    # Output is a relative score — higher = more likely than others today
    # Not a literal per-game probability. Rankings matter more than absolute values.
    hr_season = bc.get("hr", 0)
    hr_career  = blend(bc.get("hr", 0), 0, bwc)
    pa_season  = max(pa_26, 1)

    hr_per_pa_season = hr_season / pa_season if pa_season > 0 else 0
    hr_per_pa_career = hr_career / max(pa_26, 1) if pa_26 > 0 else 0.028

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
        base_rate = hr_per_pa_career

    # Floor: league avg HR/PA ~2.8%
    if base_rate <= 0:
        base_rate = 0.028
    base_rate = min(base_rate, 0.12)

    # Small sample confidence gate
    if total_pa < 30:   base_rate = base_rate * 0.55 + 0.028 * 0.45
    elif total_pa < 60: base_rate = base_rate * 0.75 + 0.028 * 0.25

    running = base_rate

    # ── Step 2: Barrel% — season + L8D weighted separately via safe_mult ──
    LG_BARREL = LEAGUE_CONSTANTS["lg_barrel_pct"]
    barrel_season = blend(bc.get("barrel_pct", 0), 0, bwc)
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
    la_season = blend(bc.get("launch_angle", 0), 0, bwc)
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
    ip_26 = pc.get("ip", 0)
    pwc = 1.0

    LG_HR9 = LEAGUE_CONSTANTS["lg_hr9"]
    LG_HH  = LEAGUE_CONSTANTS["lg_hard_hit"]
    pit_hr9_season  = blend(pc.get("hr9", 0), 0)
    pit_hr9_vs_hand = p_split_vs_bat.get("hr9", 0)
    pit_hard        = blend(pc.get("hard_hit_pct", 0), 0)
    pit_ip_vs_hand  = p_split_vs_bat.get("ip", 0)
    total_ip        = ip_26 + 0

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
    iso_overall   = blend(bc.get("iso", 0), 0, bwc)
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
    k_season = blend(bc.get("k_pct", 0), 0, bwc)
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
    # Bullpen blend is always 25% — bullpen_w is an exponent applied in safe_mult above,
    # NOT a blend fraction. Using it as a fraction would break math when bullpen_w > 1.0.
    bullpen_w_blend = 0.25
    running = (running * (1 - bullpen_w_blend)) + (bullpen_component * bullpen_w_blend)

    hr_prob = round(min(running * 100, LEAGUE_CONSTANTS["hr_prob_cap"]), 1)

    # ── Build breakdown for frontend ──
    pitch_bonus, pitch_details = compute_pitch_matchup(opp_p_name, name)
    archetype = get_archetype(barrel_season, k_season,
                              blend(bc.get("fb_pct", 0), 0, bwc),
                              iso_overall if iso_overall else blend(bc.get("iso",0), 0, bwc))
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
    blend_note = "100% 2026" + (" + 8d" if has_8d else "")

    breakdown = {
        "base_rate": round(base_rate * 100, 2),
        "barrel_mult": round(barrel_mult, 3), "la_mult": round(la_mult, 3),
        "pit_vuln_mult": round(pit_vuln_mult, 3),
        "bat_platoon_mult": round(bat_platoon_mult, 3), "pit_platoon_mult": round(pit_platoon_mult, 3),
        "park_factor": round(park_factor, 3), "weather_mult": round(weather_mult, 3),
        "hot_cold_mult": round(hot_cold_mult, 3), "k_mult": round(k_mult, 3),
        "iso_vs_hand": round(iso_vs_hand, 3), "iso_overall": round(iso_overall, 3),
        "split_pa": split_pa, "split_ip_vs_bat": round(split_ip_vs_bat, 1),
        "slg_vs_bat": round(slg_vs_bat, 3) if split_ip_vs_bat >= 5 else 0,
        "pit_slg_overall": round(slg_overall_pit, 3),
        "split_hr": int(b_split_vs_hand.get("hr", 0)),
        "split_slg": round(b_split_vs_hand.get("slg", 0), 3),
        "split_woba": round(b_split_vs_hand.get("woba", 0), 3),
        "split_k_pct": round(b_split_vs_hand.get("k_pct", 0), 1),
        "split_brl": round(b_split_vs_hand.get("barrel_pct", 0), 1),
        "split_iso": round(iso_vs_hand, 3), "split_ip": round(split_ip_vs_bat, 1),
        "hr9_split": round(p_split_vs_bat.get("hr9", 0), 2),
        "hr9_season": round(pit_hr9_season, 2), "pit_hard": round(pit_hard, 1),
        "n_pit_components": n_components,
        "pit_blend_note": "100% 2026",
        "barrel_use": round(barrel_use, 1), "barrel_season": round(barrel_season, 1),
        "la_use": round(la_use, 1), "la_season": round(la_season, 1),
        "la_8d_raw": round(la_l8d, 1), "barrel_8d_raw": round(b8d.get("barrel_pct", 0), 1),
        "hr_season": int(bc.get("hr", 0)), "pa_season": int(pa_26), "pa_8d": int(b8d.get("pa", 0)),
        "has_8d": has_8d, "blend_note": blend_note, "k_season": round(k_season, 1),
        "pitch_bonus": pitch_bonus, "pitch_breakdown": pitch_details,
        "data_conf": {
            "barrel": 1 if barrel_season > 0 and pa_26 >= 20 else 0,
            "la": 1 if la_season > 0 and pa_26 >= 20 else 0,
            "pit_hr9": 1 if pit_hr9_season > 0 and total_ip >= 10 else 0,
            "pit_hr9_hand": 1 if pit_hr9_vs_hand > 0 and pit_ip_vs_hand >= 5 else 0,
            "iso_vs_hand": 1 if iso_vs_hand > 0 and split_pa >= 30 else 0,
            "park": 1 if park_factor != 1.0 else 0,
            "pitch_delta": 1 if pitch_bonus != 0 else 0,
            "bat_platoon": 1 if bat_platoon_mult != 1.0 else 0,
        },
        "bullpen_hr9": round(bullpen_hr9, 2), "bullpen_vuln": round(bullpen_vuln, 3),
        "iso_use": round(iso_vs_hand if iso_vs_hand > 0 else iso_overall, 3),
        "pull_s": round(blend(bc.get("pull_pct", 0), 0, bwc), 1),
        "pit_modifier": round(pit_vuln_mult, 3),
        "hr_rate_8d": round(b8d.get("hr", 0) / max(b8d.get("pa", 1), 1) * 600, 1) if has_8d else 0,
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
    """
    Main prediction entry point.
    When RF is trained: uses pure Random Forest probability.
    Falls back to multiplicative model when RF not yet trained.
    """
    # Always compute multiplicative — used as fallback and for breakdown data
    mult_prob, breakdown, archetype, trend, reasons, platoon_tag, conf = \
        compute_hr_prob_multiplicative(name, bat_hand, opp_p_name, opp_p_hand, park_factor, weather_mult, home_team)

    if not _rf_trained or _rf_model is None:
        return mult_prob, breakdown, archetype, trend, reasons, platoon_tag, conf

    # ── Build RF feature vector ──
    try:
        bc  = get_batter_stats(name, 2026)
        b8d = get_batter_8d(name)
        b_split = get_batter_split(name, opp_p_hand)
        pc  = get_pitcher_stats(opp_p_name, 2026)
        p_split = get_pitcher_split(opp_p_name, bat_hand)
        pa_26 = bc.get("pa", 0); pa_25 = 0
        bwc = 1.0
        ip_26 = pc.get("ip", 0)
        pwc = 1.0
        pitch_score, _ = compute_pitch_matchup(opp_p_name, name)

        def bv(k): return blend(bc.get(k, 0), 0, bwc)
        def pv(k): return blend(pc.get(k, 0), 0)

        feat_vals = {
            "barrel_pct_season":    bv("barrel_pct"),
            "barrel_pct_l8d":       b8d.get("barrel_pct", 0),
            "la_season":            bv("launch_angle"),
            "la_l8d":               b8d.get("launch_angle", 0),
            "ev_season":            bv("exit_velo"),
            "ev_l8d":               b8d.get("exit_velo", 0),
            "iso_season":           bv("iso"),
            "iso_vs_hand":          b_split.get("iso", 0),
            "hard_hit_season":      bv("hard_hit_pct"),
            "hard_hit_l8d":         b8d.get("hard_hit_pct", 0),
            "k_pct_season":         bv("k_pct"),
            "k_pct_l8d":            b8d.get("k_pct", 0),
            "fb_pct_season":        bv("fb_pct"),
            "pull_pct_season":      bv("pull_pct"),
            "pit_hr9_season":       pv("hr9"),
            "pit_hr9_vs_hand":      p_split.get("hr9", 0),
            "pit_hard_hit_season":  pv("hard_hit_pct"),
            "pit_era_season":       pv("era"),
            "pit_k9_season":        pv("k9"),
            "pit_era_diff":         round(pv("era") - 4.20, 2) if pv("era") > 0 else 0,
            "pit_slg_vs_hand":      p_split.get("slg", 0),
            "pit_fb_pct_allowed":   pv("fb_pct"),
            "park_factor":          park_factor,
            "weather_mult":         weather_mult,
            "bat_platoon_mult":     breakdown.get("bat_platoon_mult", 1.0),
            "pit_platoon_mult":     breakdown.get("pit_platoon_mult", 1.0),
            "bullpen_vuln":         breakdown.get("bullpen_vuln", 1.0),
            "pitch_matchup_score":  pitch_score,
            "combined_pitch_delta": breakdown.get("combined_pitch_delta", 0),
            "xslg_l8d":             b8d.get("xslg", 0),
            "xwoba_l8d":            b8d.get("xwoba", 0),
            "xslg_gap_l8d":         round(b8d.get("xslg", 0) - b8d.get("slg", 0), 3) if b8d.get("xslg", 0) > 0 else 0,
            "bat_speed_l8d":        b8d.get("bat_speed", 0),
        }

        row = [float(feat_vals.get(f) or _rf_medians.get(f, 0.0)) for f in _rf_features]
        proba = _rf_model.predict_proba([row])[0]
        rf_prob = round(min(float(proba[1]) * 100, LEAGUE_CONSTANTS["hr_prob_cap"]), 1)

        breakdown["rf_prob"]   = rf_prob
        breakdown["mult_prob"] = mult_prob  # kept for debugging only

        return rf_prob, breakdown, archetype, trend, reasons, platoon_tag, conf

    except Exception as e:
        print(f"RF predict error for {name}: {e} — falling back to multiplicative")
        return mult_prob, breakdown, archetype, trend, reasons, platoon_tag, conf


def predict_xgb(name, bat_hand, opp_p_name, opp_p_hand, park_factor, weather_mult, breakdown):
    """
    Get XGBoost probability for a batter — runs silently alongside RF.
    Returns float probability or None if XGBoost not trained yet.
    Uses same feature vector as RF + day_of_season.
    """
    if not _xgb_trained or _xgb_model is None:
        return None
    try:
        bc      = get_batter_stats(name, 2026)
        b8d     = get_batter_8d(name)
        b_split = get_batter_split(name, opp_p_hand)
        pc      = get_pitcher_stats(opp_p_name, 2026)
        p_split = get_pitcher_split(opp_p_name, bat_hand)
        bwc     = 1.0

        def bv(k): return float(bc.get(k, 0) or 0)
        def pv(k): return float(pc.get(k, 0) or 0)

        feat_vals = {
            "barrel_pct_season":    bv("barrel_pct"),
            "barrel_pct_l8d":       b8d.get("barrel_pct", 0),
            "la_season":            bv("launch_angle"),
            "la_l8d":               b8d.get("launch_angle", 0),
            "ev_season":            bv("exit_velo"),
            "ev_l8d":               b8d.get("exit_velo", 0),
            "iso_season":           bv("iso"),
            "iso_vs_hand":          b_split.get("iso", 0),
            "hard_hit_season":      bv("hard_hit_pct"),
            "hard_hit_l8d":         b8d.get("hard_hit_pct", 0),
            "k_pct_season":         bv("k_pct"),
            "k_pct_l8d":            b8d.get("k_pct", 0),
            "fb_pct_season":        bv("fb_pct"),
            "pull_pct_season":      bv("pull_pct"),
            "pit_hr9_season":       pv("hr9"),
            "pit_hr9_vs_hand":      p_split.get("hr9", 0),
            "pit_hard_hit_season":  pv("hard_hit_pct"),
            "pit_era_season":       pv("era"),
            "pit_k9_season":        pv("k9"),
            "pit_era_diff":         round(pv("era") - 4.20, 2) if pv("era") > 0 else 0,
            "pit_slg_vs_hand":      p_split.get("slg", 0),
            "pit_fb_pct_allowed":   pv("fb_pct"),
            "park_factor":          park_factor,
            "weather_mult":         weather_mult,
            "bat_platoon_mult":     breakdown.get("bat_platoon_mult", 1.0),
            "pit_platoon_mult":     breakdown.get("pit_platoon_mult", 1.0),
            "bullpen_vuln":         breakdown.get("bullpen_vuln", 1.0),
            "pitch_matchup_score":  breakdown.get("pitch_matchup_score", 0),
            "combined_pitch_delta": breakdown.get("combined_pitch_delta", 0),
            "xslg_l8d":             b8d.get("xslg", 0),
            "xwoba_l8d":            b8d.get("xwoba", 0),
            "xslg_gap_l8d":         round(b8d.get("xslg", 0) - b8d.get("slg", 0), 3) if b8d.get("xslg", 0) > 0 else 0,
            "bat_speed_l8d":        b8d.get("bat_speed", 0),
            "day_of_season":        (date.today() - date(2026, 3, 20)).days,
        }

        row   = [float(feat_vals.get(f) or _xgb_medians.get(f, 0.0)) for f in _xgb_features]
        proba = _xgb_model.predict_proba([row])[0]
        return round(min(float(proba[1]) * 100, LEAGUE_CONSTANTS["hr_prob_cap"]), 1)
    except Exception as e:
        print(f"XGB predict error for {name}: {e}")
        return None

def _compute_hr_probability_legacy(name, bat_hand, opp_p_name, opp_p_hand, park_factor, weather_mult):
    bc = get_batter_stats(name, 2026)
    b8d = get_batter_8d(name)
    b_split = get_batter_split(name, opp_p_hand)

    pa_26 = bc.get("pa", 0); pa_25 = 0
    bwc = 1.0
    has_8d = b8d.get("pa", 0) >= 3
    w_s = 0.70 if has_8d else 1.0
    w_8 = 0.30 if has_8d else 0.0

    def blend3(s26, s25, d8):
        s = blend(s26, s25, bwc)
        return round(s * w_s + d8 * w_8, 2) if (has_8d and d8 > 0) else round(s, 2)

    barrel_s = blend3(bc.get("barrel_pct", 0), 0, b8d.get("barrel_pct", 0))
    fb_s     = blend3(bc.get("fb_pct", 0), 0, 0)
    pull_s   = blend3(bc.get("pull_pct", 0), 0, b8d.get("pull_pct", 0))
    la_s     = blend3(bc.get("launch_angle", 0), 0, b8d.get("launch_angle", 0))
    k_s      = blend3(bc.get("k_pct", 0), 0, 0)
    hr_fb_s  = 0

    iso_split  = b_split.get("iso", 0) if b_split.get("pa", 0) >= 20 else 0
    iso_season = blend(bc.get("iso", 0), 0, bwc)
    iso_base   = iso_split if iso_split > 0 else iso_season
    iso_8d     = b8d.get("iso", 0) if has_8d else 0
    iso_use    = round(iso_base * w_s + iso_8d * w_8, 3) if iso_8d > 0 else iso_base

    hr_rate_8d    = b8d.get("hr_rate", 0)
    barrel_season = blend(bc.get("barrel_pct", 0), 0, bwc)
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

    total_pa = pa_26
    if total_pa < 30: batter_score *= 0.55
    elif total_pa < 60: batter_score *= 0.75
    elif pa_26 < 10: batter_score *= 0.80

    archetype = get_archetype(barrel_season, k_s, fb_s, iso_season)

    # Step 2 — Pitcher modifier
    pc = get_pitcher_stats(opp_p_name, 2026)
    pp = get_pitcher_stats(opp_p_name)
    ip_26 = pc.get("ip", 0)
    pwc = 1.0

    p_split  = get_pitcher_split(opp_p_name, bat_hand)
    split_ip = p_split.get("ip", 0)

    hr9_season = blend(pc.get("hr9", 0), pp.get("hr9", 0))
    hr9_split  = p_split.get("hr9", 0) if split_ip >= 5 else 0

    if hr9_split > 0 and hr9_season > 0:
        split_ratio  = hr9_split / hr9_season
        split_weight = min(split_ip / 30.0, 0.80)
        pit_hr9      = hr9_season * (1 - split_weight) + hr9_split * split_weight
        platoon_magnitude = round((split_ratio - 1.0) * 100, 1)
    else:
        pit_hr9 = hr9_season
        platoon_magnitude = 0.0

    pit_hard = blend(pc.get("hard_hit_pct", 0), pp.get("hard_hit_pct", 0))
    pit_brl  = blend(pc.get("barrel_pct_allowed", 0), pp.get("barrel_pct_allowed", 0))
    pit_fb   = blend(pc.get("fb_pct", 0), pp.get("fb_pct", 0))

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
    blend_note = "100% 2026" + (" + 30% 8d" if has_8d else "")

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
        "pit_blend_note": "100% 2026",
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
    pp = get_pitcher_stats(p_name)
    ip_26 = pc.get("ip", 0)
    pwc = 1.0
    top_pitches = get_pitcher_top_pitches(p_name)
    vs_L = get_pitcher_split(p_name, "L")
    vs_R = get_pitcher_split(p_name, "R")
    nl = p_name.lower().strip()
    ip_data = _cache["player_ip"].get(nl, {})
    if not ip_data:
        last = nl.split()[-1]
        for k, v in _cache["player_ip"].items():
            if last in k: ip_data = v; break
    k9_val  = blend(pc.get("k9", 0), pp.get("k9", 0))
    avg_ip  = ip_data.get("avg_ip", 5.0) or 5.0
    gs_val  = ip_data.get("gs", 0)
    return {
        "name": p_name, "hand": p_hand,
        "era": round(blend(pc.get("era", 0), pp.get("era", 0)), 2) or None,
        "hr9": round(blend(pc.get("hr9", 0), pp.get("hr9", 0)), 2) or None,
        "hard_hit_pct": round(blend(pc.get("hard_hit_pct", 0), pp.get("hard_hit_pct", 0)), 1) or None,
        "barrel_pct": round(blend(pc.get("barrel_pct_allowed", 0), pp.get("barrel_pct_allowed", 0)), 1) or None,
        "ip_2026": round(ip_26, 1),
        "blend_note": "100% 2026",
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
@app.get("/dashboard")
async def get_dashboard():
    """
    Dashboard data — top 8 today + running hit rate stats.
    Calculates hit rates for top 8, top 4, and overall from last 30 days.
    """
    if not GITHUB_TOKEN:
        return {"error": "GitHub not configured"}
    import json
    try:
        # ── Load last 30 days of prediction records ──
        url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/data/predictions"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, headers=headers)
        files = r.json() if r.is_success else []
        all_records = []
        dates = []
        for f in sorted(files, key=lambda x: x["name"], reverse=True)[:30]:
            if not f["name"].endswith(".json"): continue
            d = f["name"].replace(".json","")
            content, _ = await github_get_file(f"data/predictions/{f['name']}")
            if not content: continue
            try:
                recs = json.loads(content)
                for rec in recs: rec["_date"] = d
                all_records.extend(recs)
                dates.append(d)
            except: continue

        # ── Hit rate calculations ──
        # Group by date, rank by model_hr_pct, calculate hit rates by rank tier
        from collections import defaultdict
        by_date = defaultdict(list)
        for rec in all_records:
            if rec.get("hit_hr") in [0, 1]:
                by_date[rec["_date"]].append(rec)

        top8_hits = 0; top8_total = 0
        top4_hits = 0; top4_total = 0
        overall_hits = 0; overall_total = 0
        # 2-of-top4 rate (parlay signal)
        two_of_top4 = 0; two_of_top4_days = 0
        # 2-of-top8 rate
        two_of_top8 = 0; two_of_top8_days = 0

        slate_days = []
        for d, recs in sorted(by_date.items(), reverse=True):
            ranked = sorted(recs, key=lambda x: x.get("model_hr_pct",0), reverse=True)
            t8 = ranked[:8]
            t4 = ranked[:4]
            t8_hr = sum(1 for r in t8 if r.get("hit_hr")==1)
            t4_hr = sum(1 for r in t4 if r.get("hit_hr")==1)
            all_hr = sum(1 for r in ranked if r.get("hit_hr")==1)

            top8_hits  += t8_hr;  top8_total  += len(t8)
            top4_hits  += t4_hr;  top4_total  += len(t4)
            overall_hits += all_hr; overall_total += len(ranked)

            if len(t4) >= 4:
                two_of_top4_days += 1
                if t4_hr >= 2: two_of_top4 += 1
            if len(t8) >= 8:
                two_of_top8_days += 1
                if t8_hr >= 2: two_of_top8 += 1

            slate_days.append({
                "date":         d,
                "top8_hits":    t8_hr,
                "top8_total":   len(t8),
                "top4_hits":    t4_hr,
                "slate_total":  len(ranked),
                "slate_hrs":    all_hr,
                "top8": [{"name": r["name"], "team": r.get("team",""),
                          "pct": r.get("model_hr_pct",0), "hit": r.get("hit_hr")} for r in t8],
            })

        def pct(hits, total):
            return round(hits/total*100, 1) if total > 0 else 0

        stats = {
            "days_tracked":       len(slate_days),
            "top8_hit_rate":      pct(top8_hits, top8_total),
            "top4_hit_rate":      pct(top4_hits, top4_total),
            "overall_hit_rate":   pct(overall_hits, overall_total),
            "two_of_top4_rate":   pct(two_of_top4, two_of_top4_days),
            "two_of_top8_rate":   pct(two_of_top8, two_of_top8_days),
            "top8_total_picks":   top8_total,
            "top8_total_hits":    top8_hits,
        }

        # ── Today's top 8 ──
        today = date.today().isoformat()
        today_content, _ = await github_get_file(f"data/predictions/{today}.json")
        top8_today = []
        if today_content:
            try:
                today_recs = json.loads(today_content)
                ranked_today = sorted(
                    [r for r in today_recs if r.get("model_hr_pct") is not None],
                    key=lambda x: x.get("model_hr_pct", 0), reverse=True
                )[:8]
                top8_today = ranked_today
            except: pass

        return {
            "stats":      stats,
            "top8_today": top8_today,
            "slate_days": slate_days[:14],  # last 14 days for history display
            "model_type": "random_forest" if _rf_trained else "multiplicative",
            "xgb_trained": _xgb_trained,
            "xgb_cv":     _xgb_oob,
            "rf_oob":     _model_weights.get("oob_score", 0),
        }
    except Exception as e:
        print(f"Dashboard error: {e}")
        import traceback; traceback.print_exc()
        return {"error": str(e)}


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

@app.get("/debug-results")
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
        raw, _ = await github_get_file(path)
        records = json.loads(raw) if raw else []
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
        # Model state
        "tree_trained": _rf_trained,
        "model_type": "random_forest" if _rf_trained else "multiplicative",
        "model_version": f"round-{get_rotation_round()}-day-{get_rotation_day()}",
        "records_used": _model_weights.get("records_used", 0),
        "rf_threshold": 50,
        "oob_score": _model_weights.get("oob_score"),
        "rf_params": _model_weights.get("rf_params"),
        "is_retraining": False,
    }

@app.get("/version")
def version():
    """Quick check — confirms what's deployed and model status for both RF and XGBoost."""
    rf_oob  = _model_weights.get("oob_score", 0)
    xgb_cv  = _xgb_oob
    winning = "xgboost" if (_xgb_trained and xgb_cv > rf_oob) else "random_forest"
    return {
        "file_version":    "2026-05-04",
        "active_model":    winning,
        "rf": {
            "trained":      _rf_trained,
            "records_used": _model_weights.get("records_used", 0),
            "oob_score":    rf_oob,
            "params":       _model_weights.get("rf_params"),
            "top_features": _model_weights.get("top_features", []),
        },
        "xgboost": {
            "trained":      _xgb_trained,
            "cv_auc":       xgb_cv,
            "beats_rf":     _xgb_trained and xgb_cv > rf_oob,
            "top_features": [k for k, _ in sorted(
                ({f: float(v) for f, v in _model_weights.get("feature_importances", {}).items()}).items(),
                key=lambda x: x[1], reverse=True
            )[:5]] if _xgb_trained else [],
        },
        "features": [
            "XGBoost training silently in parallel with RF",
            "day_of_season added to all prediction records",
            "Parlay tracking active",
            "2026-only model — no 2025 blending",
        ]
    }


@app.get("/xgboost-status")
async def xgboost_status():
    """Full XGBoost training status — pull latest metadata from GitHub."""
    meta = {"trained": _xgb_trained, "cv_auc": _xgb_oob}
    if GITHUB_TOKEN:
        content, _ = await github_get_file("data/xgb_meta.json")
        if content:
            import json
            try: meta.update(json.loads(content))
            except: pass
    rf_oob = _model_weights.get("oob_score", 0)
    meta["rf_oob_comparison"] = rf_oob
    meta["xgb_beats_rf"] = _xgb_trained and _xgb_oob > rf_oob
    meta["recommendation"] = (
        "XGBoost ready to go live — flip compute_hr_probability to use XGBoost"
        if meta.get("xgb_beats_rf") else
        f"RF still winning — XGBoost CV AUC {_xgb_oob:.3f} vs RF OOB {rf_oob:.3f}. Keep collecting data."
    )
    return meta

@app.post("/reload")
async def reload_data():
    _games_cache.clear()
    threading.Thread(target=run_async, args=(load_all_savant_data(),), daemon=True).start()
    asyncio.create_task(reload_contact_log())
    return {"status": "Reloading data from Baseball Savant"}

async def reload_contact_log():
    """Fetch contact log separately after a short delay so main data loads first"""
    await asyncio.sleep(45)
    async with httpx.AsyncClient(timeout=120) as client:
        df = await fetch_savant_csv(savant_contact_log_url(), client)
        if not df.empty:
            _build_contact_log(df)
            print(f"contact_log reloaded: {len(_contact_log)} players")

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
            pa_26 = bc.get("pa", 0); pa_25 = 0
            bwc = 1.0
            b8d = get_batter_8d(name)

            bl5g = get_batter_l5g(name)

            all_batters.append({
                "name": name, "team": team, "hr_prob": hr_prob,
                "xgb_prob": predict_xgb(name, bat_hand, opp_p_name, opp_p_hand, park_factor, batter_wx_mult, breakdown),
                "archetype": archetype, "trend": trend, "confidence": conf,
                "reasons": reasons, "opp_pitcher": opp_p_name,
                "bat_hand": bat_hand, "opp_p_hand": opp_p_hand,
                "park_factor": round(park_factor, 2),
                "l8d_hr_count": get_l8d_hr(name),
                "season": {
                    "barrel": round(blend(bc.get("barrel_pct", 0), 0, bwc), 1),
                    "ev":     round(blend(bc.get("exit_velo", 0), 0, bwc), 1),
                    "la":     round(blend(bc.get("launch_angle", 0), 0, bwc), 1),
                    "hh":     round(blend(bc.get("hard_hit_pct", 0), 0, bwc), 1),
                    "iso":    round(blend(bc.get("iso", 0), 0, bwc), 3),
                    "slg":    round(blend(bc.get("slg_percent", 0), 0, bwc), 3),
                    "avg":    round(blend(bc.get("batting_avg", 0), 0, bwc), 3),
                    "k":      round(blend(bc.get("k_pct", 0), 0, bwc), 1),
                    "pull":   round(blend(bc.get("pull_pct", 0), 0, bwc), 1),
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
                    "xslg":     round(b8d.get("xslg", 0), 3),
                    "xwoba":    round(b8d.get("xwoba", 0), 3),
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

@app.get("/debug-weather")
async def debug_weather(team: str = "Pittsburgh Pirates", wind_deg: int = 315, wind_speed: int = 12, temp: int = 56):
    """Debug weather calculation for any stadium.
    Example: /debug-weather?team=Pittsburgh+Pirates&wind_deg=315&wind_speed=12&temp=56
    wind_deg = meteorological direction wind comes FROM (0=N, 90=E, 180=S, 270=W)
    """
    stadium = STADIUMS.get(team)
    if not stadium:
        return {"error": f"Team not found: {team}", "available": list(STADIUMS.keys())}
    if stadium.get("dome"):
        return {"team": team, "dome": True, "result": "No wind effect in dome"}

    cf_bearing = stadium.get("cf_bearing", 67)
    hr_r = stadium.get("hr_bearing_R", (cf_bearing + 270) % 360)
    hr_l = stadium.get("hr_bearing_L", (cf_bearing + 90) % 360)
    wind_toward = (wind_deg + 180) % 360

    COMPASS = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    def compass(deg): return COMPASS[round(((deg % 360) + 360) % 360 / 22.5) % 16]

    mult_r, label_r = calc_weather_multiplier(team, wind_speed, wind_deg, temp, "R")
    mult_l, label_l = calc_weather_multiplier(team, wind_speed, wind_deg, temp, "L")

    return {
        "team": team,
        "stadium_orientation": {
            "cf_bearing": cf_bearing,
            "cf_direction": compass(cf_bearing),
            "lf_bearing": hr_r,
            "lf_direction": compass(hr_r),
            "rf_bearing": hr_l,
            "rf_direction": compass(hr_l),
        },
        "wind": {
            "from_deg": wind_deg,
            "from_direction": compass(wind_deg),
            "toward_deg": wind_toward,
            "toward_direction": compass(wind_toward),
            "speed_mph": wind_speed,
            "temp_f": temp,
        },
        "result_rhb": {"mult": mult_r, "label": label_r},
        "result_lhb": {"mult": mult_l, "label": label_l},
        "interpretation": f"Wind FROM {compass(wind_deg)} TOWARD {compass(wind_toward)}. CF at {compass(cf_bearing)}. LF at {compass(hr_r)}, RF at {compass(hr_l)}.",
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
    pa_26 = bc.get("pa", 0); pa_25 = 0
    bwc = 1.0
    has_8d = b8d.get("pa", 0) >= 3
    w_s, w_8 = (0.70, 0.30) if has_8d else (1.0, 0.0)

    def blend3(s26, s25, d8):
        s = blend(s26, s25, bwc)
        return round(s * w_s + d8 * w_8, 3) if (has_8d and d8 > 0) else round(s, 3)

    stats = {
        "name": player, "pa_2026": pa_26, "pa_2025": pa_25,
        "blend_note": "100% 2026",
        "season_2026": bc,
        "last_8d": b8d,
        "last_5g": bl5g,
        "blended": {
            "barrel_pct":   blend3(bc.get("barrel_pct", 0), 0, b8d.get("barrel_pct", 0)),
            "iso":          blend3(bc.get("iso", 0), 0, b8d.get("iso", 0)),
            "pull_pct":     blend3(bc.get("pull_pct", 0), 0, b8d.get("pull_pct", 0)),
            "launch_angle": blend3(bc.get("launch_angle", 0), 0, b8d.get("launch_angle", 0)),
            "hard_hit_pct": blend3(bc.get("hard_hit_pct", 0), 0, b8d.get("hard_hit_pct", 0)),
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
                    pwc = 1.0
                    pitch_bonus, pitch_details = compute_pitch_matchup(opp_p_name, player)
                    matchup = {
                        "home_team": game["teams"]["home"]["team"]["name"],
                        "away_team": game["teams"]["away"]["team"]["name"],
                        "game_time": game.get("gameDate", ""),
                        "pitcher_name": opp_p_name,
                        "pitcher_hand": opp_p_hand,
                        "pitcher_stats": {
                            "era": round(blend(pc.get("era", 0), 0), 2),
                            "hr9": round(blend(pc.get("hr9", 0), 0), 2),
                            "hard_hit_pct": round(blend(pc.get("hard_hit_pct", 0), 0), 1),
                            "barrel_pct_allowed": round(blend(pc.get("barrel_pct_allowed", 0), 0), 1),
                            "ip_2026": ip_26,
                            "blend_note": "100% 2026",
                        },
                        "pitch_matchup": pitch_details,
                        "pitch_bonus": pitch_bonus,
                    }
                    break
            if matchup: break
    except Exception as e:
        print(f"Research matchup error: {e}")

    return {"player": stats, "matchup": matchup, "date": today}


# ── /refresh-8d ──────────────────────────────────────────────────────────────
@app.get("/refresh-8d")
async def manual_refresh_8d():
    """Manually trigger 8-day rolling data refresh (Statcast + MLB API)."""
    asyncio.create_task(refresh_8d())
    _cache["last_8d_update"] = datetime.now().isoformat()
    return {"status": "8d refresh triggered", "ts": _cache["last_8d_update"]}


# ── /debug-arsenal ───────────────────────────────────────────────────────────
@app.get("/debug-arsenal")
async def debug_arsenal(player: str = "Freddie Freeman", pitcher: str = "Logan Webb"):
    """Debug bat_arsenal and pit_arsenal column names and matchup values."""
    bat_df = _cache.get("bat_arsenal", pd.DataFrame())
    pit_df = _cache.get("pit_arsenal", pd.DataFrame())
    bat_cols = list(bat_df.columns) if not bat_df.empty else []
    pit_cols = list(pit_df.columns) if not pit_df.empty else []
    bat_row = fuzzy_match(player, bat_df)
    pit_row = fuzzy_match(pitcher, pit_df)
    score, details = compute_pitch_matchup(pitcher, player)
    return {
        "bat_arsenal_columns": bat_cols,
        "pit_arsenal_columns": pit_cols,
        "bat_row_sample": dict(bat_row) if bat_row is not None else None,
        "pit_row_sample": dict(pit_row) if pit_row is not None else None,
        "pitch_matchup_score": score,
        "pitch_matchup_details": details,
    }


# ── Parlay combination tracking ───────────────────────────────────────────────
async def save_parlay_combinations():
    """
    Called at 12pm daily after save_daily_predictions().
    Takes top picks (≥7%), generates all 2-leg and 3-leg combos,
    saves to data/parlays/{date}.json on GitHub.
    """
    if not GITHUB_TOKEN: return
    today = date.today().isoformat()
    path  = f"data/parlays/{today}.json"
    existing, sha = await github_get_file(path)
    if existing:
        print(f"Parlay combos already saved for {today}")
        return
    try:
        import json, itertools
        pred_content, _ = await github_get_file(f"data/predictions/{today}.json")
        if not pred_content: return
        preds = json.loads(pred_content)
        top   = sorted([p for p in preds if (p.get("model_hr_pct") or 0) >= 7],
                       key=lambda x: x.get("model_hr_pct", 0), reverse=True)[:15]
        top3  = top[:10]  # 3-leg combos from top 10 only

        def combo_obj(legs):
            return {
                "legs":     [{"name": p["name"], "team": p.get("team",""), "pct": p.get("model_hr_pct",0)} for p in legs],
                "n_legs":   len(legs),
                "both_hit": None,  # patched at 2am by record_parlay_results
            }

        combos  = [combo_obj(list(c)) for c in itertools.combinations(top,  2)]
        combos += [combo_obj(list(c)) for c in itertools.combinations(top3, 3)]
        content = json.dumps(combos, indent=2)
        await github_put_file(path, content, f"parlays: {today} ({len(combos)} combos)", sha)
        print(f"Saved {len(combos)} parlay combos for {today}")
    except Exception as e:
        print(f"save_parlay_combinations error: {e}")


async def record_parlay_results(target_date: str):
    """
    Called at 2am inside record_results() after individual outcomes are patched.
    Reads today's prediction file for hit_hr outcomes, patches both_hit into combos.
    """
    if not GITHUB_TOKEN: return
    path = f"data/parlays/{target_date}.json"
    existing, sha = await github_get_file(path)
    if not existing: return
    try:
        import json
        combos = json.loads(existing)
        pred_content, _ = await github_get_file(f"data/predictions/{target_date}.json")
        if not pred_content: return
        preds    = json.loads(pred_content)
        hit_map  = {(r.get("name") or ""): r.get("hit_hr") for r in preds if r.get("name")}
        updated  = 0
        for combo in combos:
            outcomes = [hit_map.get(leg["name"]) for leg in combo.get("legs", [])]
            if all(o is not None for o in outcomes):
                combo["both_hit"] = int(all(o == 1 for o in outcomes))
                updated += 1
        content = json.dumps(combos, indent=2)
        await github_put_file(path, content, f"parlay results: {target_date}", sha)
        print(f"Patched {updated}/{len(combos)} parlay combos for {target_date}")
    except Exception as e:
        print(f"record_parlay_results error: {e}")


@app.get("/parlay-results")
async def parlay_results_endpoint(days: int = 30):
    """
    Aggregate parlay combination hit rates across tracked days.
    Returns today's combos, all-time hit rates, and best winning combos.
    """
    if not GITHUB_TOKEN:
        return {"error": "No GitHub token"}
    try:
        import json
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(
                f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/data/parlays",
                headers={"Authorization": f"token {GITHUB_TOKEN}"}
            )
            files = r.json() if r.is_success else []
        all_combos = []
        dates_loaded = []
        for f in sorted(files, key=lambda x: x.get("name",""), reverse=True)[:days]:
            if not f.get("name","").endswith(".json"): continue
            content, _ = await github_get_file(f"data/parlays/{f['name']}")
            if content:
                try:
                    day_combos = json.loads(content)
                    for c in day_combos:
                        c["_date"] = f["name"].replace(".json","")
                    all_combos.extend(day_combos)
                    dates_loaded.append(f["name"].replace(".json",""))
                except: pass

        completed = [c for c in all_combos if c.get("both_hit") is not None]
        two_leg   = [c for c in completed if c.get("n_legs") == 2]
        three_leg = [c for c in completed if c.get("n_legs") == 3]
        two_hits  = [c for c in two_leg   if c.get("both_hit") == 1]
        three_hits= [c for c in three_leg if c.get("both_hit") == 1]

        # Best combos by hit count
        combo_counts = {}
        for c in two_hits + three_hits:
            key = tuple(sorted(leg["name"] for leg in c.get("legs", [])))
            combo_counts[key] = combo_counts.get(key, 0) + 1
        best = sorted(combo_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        today_str   = date.today().isoformat()
        today_combos = [c for c in all_combos if c.get("_date") == today_str]

        return {
            "today": {
                "date":         today_str,
                "total_combos": len(today_combos),
                "two_leg":      len([c for c in today_combos if c.get("n_legs") == 2]),
                "three_leg":    len([c for c in today_combos if c.get("n_legs") == 3]),
            },
            "all_time": {
                "days_tracked":        len(dates_loaded),
                "two_leg_combos":      len(two_leg),
                "three_leg_combos":    len(three_leg),
                "two_leg_hit_rate":    f"{len(two_hits)/len(two_leg)*100:.1f}%" if two_leg else "--",
                "three_leg_hit_rate":  f"{len(three_hits)/len(three_leg)*100:.1f}%" if three_leg else "--",
            },
            "best_combos": [{"players": list(k), "hits": v} for k, v in best],
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
