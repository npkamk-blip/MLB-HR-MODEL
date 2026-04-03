from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import pandas as pd
from datetime import date, timedelta
import threading
import uvicorn
import math
import os
import io
from collections import defaultdict

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SEASON_CURRENT = 2026
SEASON_PRIOR = 2025
MLB_API = "https://statsapi.mlb.com/api/v1"
SAVANT_BASE = "https://baseballsavant.mlb.com"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

_cache = {
    "batting_current": pd.DataFrame(),
    "batting_prior": pd.DataFrame(),
    "pit_savant_current": pd.DataFrame(),   # pitcher batted ball stats from Savant
    "pit_savant_prior": pd.DataFrame(),
    "pit_arsenal_current": pd.DataFrame(),
    "pit_arsenal_prior": pd.DataFrame(),
    "bat_arsenal_current": pd.DataFrame(),
    "player_hands": {},
    "ready": False
}

PITCH_NAMES = {
    "FF":"Fastball","FA":"Fastball","SI":"Sinker","SL":"Slider",
    "CH":"Changeup","CU":"Curve","FC":"Cutter","FS":"Splitter",
    "ST":"Sweeper","KC":"Knuckle-Curve","KN":"Knuckleball","SV":"Sweeper",
}

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

BAT_COL_MAP = {
    "Name":"name","PA":"pa","HR":"hr","AVG":"avg","ISO":"iso",
    "Barrel%":"barrel_pct","EV":"exit_velo","LA":"launch_angle",
    "Hard%":"hard_hit_pct","FB%":"fb_pct","HR/FB":"hr_fb_pct",
    "Pull%":"pull_pct","K%":"k_pct",
}

async def fetch_savant_csv(url, timeout=30):
    """Fetch a CSV from Baseball Savant and return as DataFrame"""
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent":"Mozilla/5.0"})
            if not r.is_success:
                print(f"Savant CSV error {r.status_code}: {url}")
                return pd.DataFrame()
            df = pd.read_csv(io.StringIO(r.text))
            print(f"Savant CSV loaded: {len(df)} rows from {url[:80]}")
            return df
    except Exception as e:
        print(f"Savant CSV fetch error: {e}")
        return pd.DataFrame()

def fetch_savant_csv_sync(url, timeout=30):
    """Synchronous version for use in startup thread"""
    import requests
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"}, allow_redirects=True)
        if not r.ok:
            print(f"Savant CSV error {r.status_code}: {url[:80]}")
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        print(f"Savant loaded: {len(df)} rows, cols: {list(df.columns[:10])}")
        return df
    except Exception as e:
        print(f"Savant fetch error: {e}")
        return pd.DataFrame()

def load_savant_data():
    print("Loading all data...")

    # ── Batter season stats from Baseball Savant (FanGraphs blocks Railway) ──
    for season, kb in [(SEASON_CURRENT,"batting_current"),(SEASON_PRIOR,"batting_prior")]:
        # Savant custom leaderboard — batter power/contact stats
        url = (f"{SAVANT_BASE}/leaderboard/custom?year={season}&type=batter&filter=&min=5"
               f"&selections=player_id,player_name,pa,barrel_batted_rate,hard_hit_percent,"
               f"exit_velocity_avg,launch_angle_avg,sweet_spot_percent&csv=true")
        bat = fetch_savant_csv_sync(url)
        if bat.empty:
            print(f"{season} batter Savant failed")
            continue
        bat.columns = [c.lower().strip() for c in bat.columns]
        # Also fetch ISO/pull/FB from Savant sprint speed + batted ball leaderboard
        url2 = (f"{SAVANT_BASE}/leaderboard/custom?year={season}&type=batter&filter=&min=5"
                f"&selections=player_id,player_name,pa,iz_contact_percent,"
                f"oz_swing_percent,whiff_percent&csv=true")
        bat2 = fetch_savant_csv_sync(url2)

        # Build unified batter dataframe with standardized column names
        bat_clean = pd.DataFrame()
        bat_clean["name"] = bat.get("player_name", bat.get("last_name, first_name", pd.Series()))
        bat_clean["player_id"] = bat.get("player_id", pd.Series(dtype=int))
        bat_clean["pa"] = pd.to_numeric(bat.get("pa", 0), errors="coerce").fillna(0)

        # Barrel rate — Savant stores as 0-100
        brl = bat.get("barrel_batted_rate", pd.Series(dtype=float))
        bat_clean["barrel_pct"] = pd.to_numeric(brl, errors="coerce").fillna(0)

        # Hard hit %
        hh = bat.get("hard_hit_percent", pd.Series(dtype=float))
        bat_clean["hard_hit_pct"] = pd.to_numeric(hh, errors="coerce").fillna(0)

        # Exit velo
        ev = bat.get("exit_velocity_avg", pd.Series(dtype=float))
        bat_clean["exit_velo"] = pd.to_numeric(ev, errors="coerce").fillna(0)

        # Launch angle
        la = bat.get("launch_angle_avg", pd.Series(dtype=float))
        bat_clean["launch_angle"] = pd.to_numeric(la, errors="coerce").fillna(0)

        _cache[kb] = bat_clean
        print(f"{season} batting (Savant): {len(bat_clean)} players")
        for col in ["barrel_pct","exit_velo","hard_hit_pct"]:
            vals = bat_clean[col].dropna()
            vals = vals[vals > 0]
            if len(vals) > 0:
                print(f"  {col} median: {vals.median():.2f}")

    # ── Also get ISO, pull%, FB%, K% from MLB Stats API per player ──
    # These will be populated per-player at game time from batter splits
    # For now Savant gives us the core power metrics we need

    # ── Pitcher batted ball stats from Baseball Savant ──
    for season, kp in [(SEASON_CURRENT,"pit_savant_current"),(SEASON_PRIOR,"pit_savant_prior")]:
        url = (f"{SAVANT_BASE}/leaderboard/custom?year={season}&type=pitcher&filter=&min=1"
               f"&selections=player_id,player_name,pa,barrel_batted_rate,hard_hit_percent,"
               f"exit_velocity_avg,launch_angle_avg,fb_percent,gb_percent,hr_per_pa&csv=true")
        df = fetch_savant_csv_sync(url)
        if not df.empty:
            # Normalize column names
            df.columns = [c.lower().strip() for c in df.columns]
            _cache[kp] = df
            print(f"{season} pitcher Savant: {len(df)} rows, cols: {list(df.columns[:12])}")

    # Pitch mix coming soon — placeholder
    print("Pitch mix disabled for now — coming soon")

    _cache["ready"] = True
    print("All data ready.")

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=load_savant_data, daemon=True).start()

def fuzzy_match_name(name, df, name_col="player_name"):
    """Match player name in a dataframe"""
    if df is None or df.empty or name_col not in df.columns:
        return None
    name_lower = name.lower().strip()
    # Exact match
    matches = df[df[name_col].str.lower().str.strip() == name_lower]
    if not matches.empty:
        return matches.iloc[0]
    # Contains match
    for _, row in df.iterrows():
        rn = str(row.get(name_col,"")).lower().strip()
        if rn in name_lower or name_lower in rn:
            return row
    # Last name match
    parts = name_lower.split()
    if len(parts) >= 2:
        last = parts[-1]
        for _, row in df.iterrows():
            if last in str(row.get(name_col,"")).lower():
                return row
    return None

def fuzzy_match(name, df):
    """Match by name column — works with both pybaseball and Savant formats"""
    if df is None or df.empty: return None
    # Try 'name' column first, then 'player_name'
    for col in ["name", "player_name"]:
        if col in df.columns:
            result = fuzzy_match_name(name, df, name_col=col)
            if result is not None:
                return result
    return None

def get_pitcher_savant_stats(pitcher_name, season="current"):
    """Get pitcher batted ball stats from Savant cache"""
    df = _cache["pit_savant_current"] if season == "current" else _cache["pit_savant_prior"]
    row = fuzzy_match_name(pitcher_name, df, "player_name")
    if row is None and season == "current":
        row = fuzzy_match_name(pitcher_name, _cache["pit_savant_prior"], "player_name")
    if row is None:
        return {}
    result = {}
    col_map = {
        "barrel_batted_rate":"barrel_pct_allowed",
        "hard_hit_percent":"hard_hit_pct_allowed",
        "exit_velocity_avg":"exit_velo_allowed",
        "launch_angle_avg":"la_allowed",
        "fb_percent":"fb_pct",
        "gb_percent":"gb_pct",
        "hr_per_pa":"hr_per_pa",
        "pa":"pa",
    }
    for src, dst in col_map.items():
        if src in row.index:
            val = row.get(src, 0)
            try:
                result[dst] = float(val) if val and str(val) != 'nan' else 0
            except:
                result[dst] = 0
    return result

def get_pitcher_top_pitches(pitcher_name):
    """Pitch mix coming soon"""
    return []

def get_batter_barrel_vs_pitches(player_name, pitch_codes):
    """Get batter's barrel rate vs specific pitch types"""
    df = _cache["bat_arsenal_current"]
    if df.empty or not pitch_codes:
        return {}
    # Find all rows for this batter
    name_col = next((c for c in ["player_name","last_name, first_name"] if c in df.columns), None)
    if not name_col:
        return {}
    batter_rows = df[df[name_col].str.lower().str.strip().str.contains(player_name.split()[-1].lower(), na=False)]
    if batter_rows.empty:
        return {}
    pitch_col = next((c for c in df.columns if "pitch_type" in c), None)
    barrel_col = next((c for c in df.columns if "barrel" in c), None)
    if not pitch_col or not barrel_col:
        return {}
    result = {}
    for _, row in batter_rows.iterrows():
        pt = str(row.get(pitch_col,"")).upper()
        if pt in pitch_codes:
            try:
                result[pt] = round(float(row.get(barrel_col, 0) or 0), 1)
            except:
                pass
    return result

def get_batter_blend_weights(pa_current):
    pa = float(pa_current or 0)
    if pa >= 150: return 1.0, 0.0
    elif pa >= 50: w=(pa-50)/100.0; return w, 1.0-w
    else: w=0.20+(pa/50.0)*0.30; return w, 1.0-w

def get_pitcher_blend_weights(ip_current):
    ip = float(ip_current or 0)
    if ip >= 30: return 1.0, 0.0
    elif ip >= 10: w=(ip-10)/20.0; return 0.5+w*0.5, 0.5-w*0.5
    elif ip > 0: return 0.5, 0.5
    else: return 0.0, 1.0

def blend_stat(cv, pv, wc, wp):
    c, p = float(cv or 0), float(pv or 0)
    if c==0 and p==0: return 0
    if c==0: return p
    if p==0: return c
    return c*wc + p*wp

def get_park_hr_factor(home_team, batter_hand):
    pf = PARK_HR_FACTORS.get(home_team, {"L":1.0,"R":1.0})
    return pf.get(batter_hand if batter_hand in ("L","R") else "R", 1.0)

def angle_diff(a, b):
    diff = abs(a-b) % 360
    return diff if diff <= 180 else 360-diff

def calc_weather_multiplier(home_team, wind_speed, wind_direction, temperature, batter_hand="R"):
    stadium = STADIUMS.get(home_team)
    if not stadium: return 1.0, "Unknown"
    if stadium.get("dome"): return 1.0, "Dome"
    hr_bearing = stadium.get("hr_bearing", 225)
    open_factor = stadium.get("open_factor", 0.5)
    diff = angle_diff(wind_direction, hr_bearing)
    alignment = math.cos(math.radians(diff))
    if wind_speed < 5: speed_factor = 0
    elif wind_speed < 10: speed_factor = 0.3
    elif wind_speed < 16: speed_factor = 0.7
    else: speed_factor = 1.0
    wind_mult = 1.0 + (alignment * speed_factor * 0.12 * open_factor)
    temp_mult = 1.06 if temperature >= 80 else 1.02 if temperature >= 70 else 0.91 if temperature < 50 else 0.96 if temperature < 60 else 1.0
    if abs(alignment) <= 0.5 and wind_speed >= 10:
        cross_diff = (wind_direction - hr_bearing) % 360
        if 45 < cross_diff < 225:
            direction_label = "Favors Lefties"
            if batter_hand != "L": wind_mult *= 0.97
        else:
            direction_label = "Favors Righties"
            if batter_hand != "R": wind_mult *= 0.97
    elif alignment > 0.5 and wind_speed >= 10: direction_label = "Blowing Out"
    elif alignment < -0.5 and wind_speed >= 10: direction_label = "Blowing In"
    elif wind_speed < 5: direction_label = "Calm"
    else: direction_label = "Crosswind"
    return round(wind_mult * temp_mult, 3), direction_label

def sigmoid_to_prob(raw_score):
    centered = (raw_score - 50) / 18.0
    sigmoid = 1 / (1 + math.exp(-centered))
    prob = 0.02 + sigmoid * 0.25
    return round(min(max(prob, 0.02), 0.25) * 100, 1)

def get_archetype(barrel_pct, k_pct, fb_pct, iso):
    if barrel_pct >= 10 and k_pct >= 28: return "Boom/Bust"
    elif barrel_pct >= 10 and k_pct < 22: return "Pure Power"
    elif barrel_pct >= 7 and fb_pct >= 38: return "Power"
    elif iso >= 0.180 and k_pct < 20: return "Balanced"
    elif k_pct >= 28: return "High K"
    else: return "Contact"

def get_trend(hr_rate_14d, barrel_season):
    if barrel_season == 0: return "Steady"
    if hr_rate_14d > 30: return "Heating Up"
    elif hr_rate_14d < 5: return "Cooling Off"
    return "Steady"

def get_confidence(pa_current, ip_pitcher, is_projected, bat_splits_pa):
    score = 0
    if pa_current >= 150: score += 3
    elif pa_current >= 50: score += 2
    elif pa_current >= 20: score += 1
    if ip_pitcher >= 20: score += 2
    elif ip_pitcher >= 10: score += 1
    if not is_projected: score += 2
    if bat_splits_pa and bat_splits_pa >= 30: score += 1
    if score >= 7: return "High"
    elif score >= 4: return "Medium"
    else: return "Low"

async def fetch_weather(lat, lon, game_time_utc):
    try:
        hour = 18
        if game_time_utc:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(game_time_utc.replace("Z","+00:00"))
                hour = dt.hour
            except: pass
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&hourly=temperature_2m,windspeed_10m,winddirection_10m"
               f"&temperature_unit=fahrenheit&windspeed_unit=mph&forecast_days=1&timezone=auto")
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url); d = r.json()
        hourly = d.get("hourly",{})
        times = hourly.get("time",[])
        temps = hourly.get("temperature_2m",[])
        speeds = hourly.get("windspeed_10m",[])
        directions = hourly.get("winddirection_10m",[])
        idx = min(hour, len(temps)-1)
        for i,t in enumerate(times):
            if f"T{hour:02d}:" in t: idx=i; break
        return (round(temps[idx]) if idx<len(temps) else 70,
                round(speeds[idx]) if idx<len(speeds) else 0,
                round(directions[idx]) if idx<len(directions) else 0)
    except: return 70, 0, 0

async def fetch_player_hand(player_id):
    if player_id in _cache["player_hands"]:
        return _cache["player_hands"][player_id]
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(f"{MLB_API}/people/{player_id}")
            d = r.json()
        person = d.get("people",[{}])[0]
        bat = person.get("batSide",{}).get("code","") or "R"
        pit = person.get("pitchHand",{}).get("code","") or "R"
        result = {"bat_side":bat,"pitch_hand":pit,"name":person.get("fullName","")}
        _cache["player_hands"][player_id] = result
        return result
    except:
        return {"bat_side":"R","pitch_hand":"R","name":""}

async def fetch_batter_splits(player_id, pit_hand, season=2026):
    try:
        split_code = "vsl" if pit_hand == "L" else "vsr"
        url = (f"{MLB_API}/people/{player_id}/stats"
               f"?stats=statSplits&group=hitting&season={season}&sitCodes={split_code}")
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url); d = r.json()
        for stat_group in d.get("stats",[]):
            for split in stat_group.get("splits",[]):
                if split.get("split",{}).get("code","") != split_code: continue
                s = split.get("stat",{})
                slg = float(s.get("sluggingPercentage",0) or 0)
                avg = float(s.get("avg",0) or 0)
                pa = int(s.get("plateAppearances",0))
                hr = int(s.get("homeRuns",0))
                return {"hr":hr,"pa":pa,"slg":slg,"avg":avg,"iso":slg-avg,
                        "hr_rate":(hr/max(pa,1))*600}
        return None
    except: return None

async def fetch_pitcher_splits(player_id, season=2026):
    try:
        url = (f"{MLB_API}/people/{player_id}/stats"
               f"?stats=statSplits&group=pitching&season={season}&sitCodes=vsl,vsr")
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url); d = r.json()
        result = {}
        for stat_group in d.get("stats",[]):
            for split in stat_group.get("splits",[]):
                code = split.get("split",{}).get("code","")
                if code not in ("vsl","vsr"): continue
                s = split.get("stat",{})
                ip = float(s.get("inningsPitched",0) or 0)
                hr = int(s.get("homeRuns",0))
                result[code] = {
                    "hr9":round((hr/ip)*9,2) if ip>0 else 0,
                    "era":float(s.get("era",0) or 0),
                    "ip":ip,"pa":int(s.get("battersFaced",0)),
                }
        return result
    except: return {}

async def fetch_pitcher_season_stats(player_id, season):
    try:
        url = f"{MLB_API}/people/{player_id}/stats?stats=season&group=pitching&season={season}"
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(url); d = r.json()
        splits = d.get("stats",[{}])[0].get("splits",[])
        if not splits: return {}
        s = splits[0].get("stat",{})
        ip = float(s.get("inningsPitched",0) or 0)
        hr = int(s.get("homeRuns",0))
        so = int(s.get("strikeOuts",0))
        tbf = int(s.get("battersFaced",1))
        return {
            "era":float(s.get("era",0) or 0),
            "whip":float(s.get("whip",0) or 0),
            "hr9":round((hr/ip)*9,2) if ip>0 else 0,
            "k9":round((so/ip)*9,2) if ip>0 else 0,
            "k_pct":round((so/tbf)*100,1) if tbf>0 else 0,
            "ip":ip,
        }
    except: return {}

async def fetch_batter_season_stats(player_id, season=2026):
    """Fetch batter season stats from MLB Stats API — ISO, K%, OPS etc"""
    try:
        url = f"{MLB_API}/people/{player_id}/stats?stats=season&group=hitting&season={season}"
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(url); d = r.json()
        splits = d.get("stats",[{}])[0].get("splits",[])
        if not splits: return {}
        s = splits[0].get("stat",{})
        slg = float(s.get("sluggingPercentage",0) or 0)
        avg = float(s.get("avg",0) or 0)
        obp = float(s.get("obp",0) or 0)
        so  = int(s.get("strikeOuts",0))
        pa  = int(s.get("plateAppearances",1))
        hr  = int(s.get("homeRuns",0))
        fb_raw = int(s.get("flyBalls",0)) if "flyBalls" in s else 0
        ab  = int(s.get("atBats",1))
        return {
            "iso": round(slg - avg, 3),
            "slg": slg,
            "avg": avg,
            "obp": obp,
            "ops": round(slg + obp, 3),
            "k_pct": round((so/pa)*100, 1) if pa > 0 else 0,
            "hr": hr,
            "pa": pa,
            "hr_rate": round((hr/pa)*600, 1) if pa > 0 else 0,
        }
    except: return {}

async def fetch_recent_form(player_id, days=14):
    try:
        end, start = date.today(), date.today()-timedelta(days=days)
        url = f"{MLB_API}/people/{player_id}/stats?stats=byDateRange&group=hitting&season=2026&startDate={start}&endDate={end}"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url); d = r.json()
        splits = d.get("stats",[{}])[0].get("splits",[])
        if not splits: return None
        s = splits[0].get("stat",{})
        slg = float(s.get("sluggingPercentage",0) or 0)
        avg = float(s.get("avg",0) or 0)
        pa = int(s.get("plateAppearances",0))
        hr = int(s.get("homeRuns",0))
        return {"hr":hr,"pa":pa,"slg":slg,"iso":slg-avg,
                "hr_rate":(hr/max(pa,1))*600,"ops":float(s.get("ops",0) or 0)}
    except: return None

async def fetch_projected_lineup(team_id, team_name):
    try:
        end, start = date.today(), date.today()-timedelta(days=10)
        url = f"{MLB_API}/schedule?sportId=1&teamId={team_id}&startDate={start}&endDate={end}&hydrate=boxscore"
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url); d = r.json()
        recent_games = []
        for de in reversed(d.get("dates",[])):
            for g in de.get("games",[]):
                if g.get("status",{}).get("abstractGameState") == "Final":
                    recent_games.append(g["gamePk"])
            if len(recent_games) >= 5: break
        player_data = defaultdict(lambda: {"name":"","appearances":0,"orders":[],"id":0})
        for gid in recent_games[:5]:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(f"{MLB_API}/game/{gid}/boxscore"); box = r.json()
                for side in ["away","home"]:
                    td = box.get("teams",{}).get(side,{})
                    if team_name.lower() not in td.get("team",{}).get("name","").lower(): continue
                    for _, p in td.get("players",{}).items():
                        order = p.get("battingOrder")
                        if order and int(order) <= 900:
                            person = p.get("person",{})
                            pid = person.get("id",0)
                            player_data[pid]["name"] = person.get("fullName","")
                            player_data[pid]["id"] = pid
                            player_data[pid]["appearances"] += 1
                            player_data[pid]["orders"].append(int(order)//100)
            except: continue
        projected = [{"id":d["id"],"name":d["name"],"appearances":d["appearances"],
                      "avg_order":sum(d["orders"])/len(d["orders"])}
                     for d in player_data.values() if d["appearances"]>=2 and d["name"]]
        projected.sort(key=lambda x: x["avg_order"])
        return projected[:9], "projected"
    except Exception as e:
        print(f"Projected lineup error {team_name}: {e}"); return [], "projected"

async def fetch_dk_hr_props():
    if not ODDS_API_KEY: return {}
    try:
        for market in ["batter_home_runs","player_home_runs"]:
            url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events?apiKey={ODDS_API_KEY}&dateFormat=iso"
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(url)
                if not r.is_success: continue
                events = r.json()
            props = {}
            for event in events[:15]:
                event_id = event.get("id","")
                try:
                    prop_url = (f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds?"
                               f"apiKey={ODDS_API_KEY}&regions=us&markets={market}&oddsFormat=american&bookmakers=draftkings")
                    async with httpx.AsyncClient(timeout=10) as client:
                        pr = await client.get(prop_url)
                        if not pr.is_success: continue
                        pd_data = pr.json()
                    for bk in pd_data.get("bookmakers",[]):
                        if bk.get("key") != "draftkings": continue
                        for mkt in bk.get("markets",[]):
                            for outcome in mkt.get("outcomes",[]):
                                name = outcome.get("description") or outcome.get("name","")
                                price = outcome.get("price",0)
                                if name and price: props[name.lower()] = price
                except: continue
            if props: print(f"DK HR props: {len(props)}"); return props
        return {}
    except Exception as e:
        print(f"DK props error: {e}"); return {}

def match_dk_odds(player_name, props):
    if not props: return None
    nl = player_name.lower()
    if nl in props: return props[nl]
    parts = nl.split()
    if len(parts) >= 2:
        last = parts[-1]
        for k,v in props.items():
            if last in k: return v
    return None

def fmt_odds(o):
    if o is None: return None
    return f"+{int(o)}" if o > 0 else str(int(o))

def compute_hr_probability(bc, bp, recent, bat_hand, opp_hand,
                           bat_splits, barrel_vs_pitches, pitcher_top_pitches,
                           park_factor, weather_mult,
                           pit_mlb_cur, pit_mlb_pri,
                           pit_savant, pit_splits, ip_current, pa_current):
    wc, wp = get_batter_blend_weights(pa_current)
    pit_wc, pit_wp = get_pitcher_blend_weights(ip_current)

    # ── STEP 1: Batter HR Score ──
    barrel_s = blend_stat(bc.get("barrel_pct"), bp.get("barrel_pct"), wc, wp)
    hard_s   = blend_stat(bc.get("hard_hit_pct"), bp.get("hard_hit_pct"), wc, wp)
    fb_s     = blend_stat(bc.get("fb_pct"), bp.get("fb_pct"), wc, wp)
    pull_s   = blend_stat(bc.get("pull_pct"), bp.get("pull_pct"), wc, wp)
    iso_s    = blend_stat(bc.get("iso"), bp.get("iso"), wc, wp)
    k_s      = blend_stat(bc.get("k_pct"), bp.get("k_pct"), wc, wp)
    la_s     = float(bc.get("launch_angle",0) or 0)

    # Use split ISO if available
    iso_use = iso_s
    if bat_splits and bat_splits.get("pa",0) >= 30:
        iso_use = bat_splits.get("iso", iso_s) or iso_s

    s1_barrel = round(min(barrel_s/15.0,1.0)*20, 2)
    s1_hard   = round(min(hard_s/45.0,1.0)*10, 2)
    s1_iso    = round(min(iso_use/0.280,1.0)*10, 2)
    s1_fb     = round(min(fb_s/45.0,1.0)*8, 2)
    s1_pull   = round(min(pull_s/50.0,1.0)*7, 2)
    s1_la     = round(min(max(la_s-10,0)/20.0,1.0)*5, 2) if la_s > 0 else 0

    batter_score = s1_barrel + s1_hard + s1_iso + s1_fb + s1_pull + s1_la

    # Recent form trend
    hr_rate_14d = 0.0
    if recent and recent.get("pa",0) >= 5:
        hr_rate_14d = recent.get("hr_rate", 0)
        if hr_rate_14d > 30: batter_score = min(batter_score * 1.15, 60)
        elif hr_rate_14d > 15: batter_score = min(batter_score * 1.07, 60)
        elif hr_rate_14d < 3 and recent.get("pa",0) >= 15: batter_score = batter_score * 0.95

    archetype = get_archetype(barrel_s, k_s, fb_s, iso_s)
    trend_label = get_trend(hr_rate_14d, barrel_s)

    # Pitch matchup bonus
    pitch_bonus = 0.0
    pitch_breakdown = []
    # League avg wOBA ~.320, strong batter vs pitch ~.400+, weak ~.240
    if barrel_vs_pitches and pitcher_top_pitches:
        for pitch in pitcher_top_pitches:
            code = pitch["code"]
            usage = pitch["pct"] / 100.0
            vs_data = barrel_vs_pitches.get(code)
            if vs_data and vs_data.get("woba") is not None:
                woba = vs_data["woba"]
                diff = woba - 0.320  # vs league avg
                pts = diff * usage * 25  # scale to meaningful points
                pitch_bonus += pts
                pitch_breakdown.append({
                    "name":pitch["name"],"pct":pitch["pct"],
                    "woba_vs":woba,"diff":round(diff,3),"bonus":round(pts,1)
                })
        pitch_bonus = max(min(pitch_bonus, 8), -8)

    after_pitch = min(batter_score + pitch_bonus, 60)

    # ── STEP 2: Pitcher HR Modifier ──
    # HR/9 from MLB Stats API (most reliable)
    pit_hr9_cur = pit_mlb_cur.get("hr9", 0)
    pit_hr9_pri = pit_mlb_pri.get("hr9", 0)
    pit_hr9 = blend_stat(pit_hr9_cur, pit_hr9_pri, pit_wc, pit_wp)

    # K% from MLB Stats API
    pit_k_cur = pit_mlb_cur.get("k_pct", 0)
    pit_k_pri = pit_mlb_pri.get("k_pct", 0)
    pit_k = blend_stat(pit_k_cur, pit_k_pri, pit_wc, pit_wp)

    # Batted ball stats from Baseball Savant
    pit_hrfb  = float(pit_savant.get("hr_per_pa", 0) or 0) * 600  # convert to per 600 PA rate
    pit_hard  = float(pit_savant.get("hard_hit_pct_allowed", 0) or 0)
    pit_brl   = float(pit_savant.get("barrel_pct_allowed", 0) or 0)
    pit_fb    = float(pit_savant.get("fb_pct", 0) or 0)

    # Use pitcher splits if available
    vs_hand_code = "vsl" if bat_hand == "L" else "vsr"
    if pit_splits and vs_hand_code in pit_splits:
        split = pit_splits[vs_hand_code]
        if split.get("ip",0) >= 10:
            pit_hr9 = split.get("hr9", pit_hr9)

    # Build pitcher modifier
    # League avg: HR/9~1.15, hard hit~32%, barrel allowed~7.5%, FB%~36%
    m_hr9  = 1.0 + (pit_hr9 - 1.15) / 1.15 * 0.40  if pit_hr9 > 0 else 1.0
    m_hard = 1.0 + (pit_hard - 32.0) / 32.0 * 0.25 if pit_hard > 0 else 1.0
    m_brl  = 1.0 + (pit_brl - 7.5) / 7.5 * 0.20    if pit_brl > 0 else 1.0
    m_fb   = 1.0 + (pit_fb - 36.0) / 36.0 * 0.15   if pit_fb > 0 else 1.0

    # Count how many components have data
    n_components = sum([pit_hr9>0, pit_hard>0, pit_brl>0, pit_fb>0])
    if n_components == 0:
        pit_modifier = 1.0
    elif n_components == 1:
        pit_modifier = m_hr9 if pit_hr9 > 0 else 1.0
    else:
        weights = []
        mods = []
        if pit_hr9 > 0: weights.append(0.40); mods.append(m_hr9)
        if pit_hard > 0: weights.append(0.25); mods.append(m_hard)
        if pit_brl > 0: weights.append(0.20); mods.append(m_brl)
        if pit_fb > 0: weights.append(0.15); mods.append(m_fb)
        # Renormalize weights
        total_w = sum(weights)
        pit_modifier = sum(m*w/total_w for m,w in zip(mods,weights))

    pit_modifier = round(max(min(pit_modifier, 1.6), 0.6), 3)

    # ── STEP 3: K% cap ──
    k_cap = 1.0
    if k_s >= 35: k_cap = 0.75
    elif k_s >= 30: k_cap = 0.88
    elif k_s >= 28: k_cap = 0.94
    after_k = after_pitch * k_cap

    # ── STEP 4: Context ──
    after_context = round(after_k * pit_modifier * park_factor * weather_mult, 1)

    # ── STEP 5: Sigmoid ──
    hr_prob = sigmoid_to_prob(after_context)

    reasons = []
    if barrel_s > 10: reasons.append(f"Barrel {barrel_s:.1f}%")
    if iso_use > 0.200: reasons.append(f"ISO .{int(iso_use*1000):03d}")
    if pit_hr9 > 1.3: reasons.append(f"SP {pit_hr9:.1f} HR/9")
    if pit_hard > 38: reasons.append(f"SP {pit_hard:.0f}% hard contact")
    if park_factor >= 1.15: reasons.append("HR-friendly park")
    elif park_factor <= 0.90: reasons.append("Pitcher-friendly park")

    breakdown = {
        "barrel_s":round(barrel_s,1),"s1_barrel":s1_barrel,
        "hard_s":round(hard_s,1),"s1_hard":s1_hard,
        "iso_use":round(iso_use,3),"s1_iso":s1_iso,
        "fb_s":round(fb_s,1),"s1_fb":s1_fb,
        "pull_s":round(pull_s,1),"s1_pull":s1_pull,
        "la_s":round(la_s,1),"s1_la":s1_la,
        "batter_score":round(batter_score,1),
        "k_s":round(k_s,1),"k_cap":k_cap,
        "pit_hr9":round(pit_hr9,2),"pit_hard":round(pit_hard,1),
        "pit_brl":round(pit_brl,1),"pit_fb":round(pit_fb,1),
        "pit_modifier":pit_modifier,
        "pitch_bonus":round(pitch_bonus,1),
        "pitch_breakdown":pitch_breakdown,
        "after_pitch":round(after_pitch,1),
        "after_k":round(after_k,1),
        "park_factor":park_factor,
        "weather_mult":weather_mult,
        "after_context":after_context,
        "hr_prob":hr_prob,
        "blend_note":f"{int(wc*100)}% 2026 / {int(wp*100)}% 2025 ({int(pa_current)} PA)",
        "pit_blend_note":f"{int(pit_wc*100)}% 2026 / {int(pit_wp*100)}% 2025 ({ip_current:.0f} IP)",
        "n_pit_components":n_components,
    }

    return hr_prob, breakdown, archetype, trend_label, reasons

@app.get("/")
def root():
    return {"status":"Sharp MLB HR Model","data_ready":_cache["ready"],"season":SEASON_CURRENT}

@app.get("/status")
def status():
    return {
        "ready":_cache["ready"],
        "batters_2026":len(_cache["batting_current"]),  # from Savant
        "batters_2025":len(_cache["batting_prior"]),
        "pit_savant_2026":len(_cache["pit_savant_current"]),
        "pit_savant_2025":len(_cache["pit_savant_prior"]),
        "pit_arsenal_2026":len(_cache["pit_arsenal_current"]),
        "bat_arsenal_2026":len(_cache["bat_arsenal_current"]),
    }

@app.get("/debug-savant")
def debug_savant():
    result = {}
    for key in ["pit_savant_current","pit_savant_prior","pit_arsenal_current","bat_arsenal_current"]:
        df = _cache[key]
        result[key] = {"rows":len(df),"cols":list(df.columns[:15]) if not df.empty else []}
    return result

@app.get("/games")
async def get_games(form_days: int = 14):
    if not _cache["ready"]:
        return {"games":[],"date":date.today().isoformat(),"loading":True,
                "message":"Data loading — try again in 60 seconds."}
    today = date.today().isoformat()
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{MLB_API}/schedule?sportId=1&date={today}&hydrate=team,probablePitcher")
        data = r.json()
    dk_props = await fetch_dk_hr_props()
    dates = data.get("dates",[])
    if not dates: return {"games":[],"date":today,"loading":False}

    games_out = []
    for game in dates[0].get("games",[]):
        if game.get("status",{}).get("abstractGameState") == "Final": continue
        gid = game["gamePk"]
        away_team    = game["teams"]["away"]["team"]["name"]
        home_team    = game["teams"]["home"]["team"]["name"]
        away_team_id = game["teams"]["away"]["team"]["id"]
        home_team_id = game["teams"]["home"]["team"]["id"]
        away_p       = game["teams"]["away"].get("probablePitcher",{})
        home_p       = game["teams"]["home"].get("probablePitcher",{})
        gtime        = game.get("gameDate","")

        away_p_id = away_p.get("id"); home_p_id = home_p.get("id")
        away_p_hand = home_p_hand = "R"
        away_p_mlb_cur = home_p_mlb_cur = {}
        away_p_mlb_pri = home_p_mlb_pri = {}
        away_p_splits = home_p_splits = {}
        away_p_savant = home_p_savant = {}
        away_p_pitches = home_p_pitches = []

        if away_p_id:
            info = await fetch_player_hand(away_p_id)
            away_p_hand = info.get("pitch_hand","R")
            away_p_mlb_cur = await fetch_pitcher_season_stats(away_p_id, SEASON_CURRENT)
            away_p_mlb_pri = await fetch_pitcher_season_stats(away_p_id, SEASON_PRIOR)
            away_p_splits = await fetch_pitcher_splits(away_p_id, SEASON_CURRENT)
            if not away_p_splits: away_p_splits = await fetch_pitcher_splits(away_p_id, SEASON_PRIOR)
            away_p_savant = get_pitcher_savant_stats(away_p.get("fullName",""))
            away_p_pitches = get_pitcher_top_pitches(away_p.get("fullName",""))

        if home_p_id:
            info = await fetch_player_hand(home_p_id)
            home_p_hand = info.get("pitch_hand","R")
            home_p_mlb_cur = await fetch_pitcher_season_stats(home_p_id, SEASON_CURRENT)
            home_p_mlb_pri = await fetch_pitcher_season_stats(home_p_id, SEASON_PRIOR)
            home_p_splits = await fetch_pitcher_splits(home_p_id, SEASON_CURRENT)
            if not home_p_splits: home_p_splits = await fetch_pitcher_splits(home_p_id, SEASON_PRIOR)
            home_p_savant = get_pitcher_savant_stats(home_p.get("fullName",""))
            home_p_pitches = get_pitcher_top_pitches(home_p.get("fullName",""))

        stadium = STADIUMS.get(home_team,{})
        temp, wind_speed, wind_dir = 70, 0, 0
        if not stadium.get("dome") and stadium.get("lat"):
            temp, wind_speed, wind_dir = await fetch_weather(stadium["lat"], stadium["lon"], gtime)

        lineup_away, lineup_home = [], []
        lineup_away_status = lineup_home_status = "projected"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(f"{MLB_API}/game/{gid}/boxscore"); box = r.json()
            teams = box.get("teams",{})
            def extract(side):
                players = teams.get(side,{}).get("players",{})
                return sorted([p for p in players.values() if p.get("battingOrder") and int(p["battingOrder"])<=900],
                              key=lambda x:int(x["battingOrder"]))[:9]
            ca, ch = extract("away"), extract("home")
            if ca: lineup_away=ca; lineup_away_status="confirmed"
            if ch: lineup_home=ch; lineup_home_status="confirmed"
        except: pass
        if not lineup_away: lineup_away, _ = await fetch_projected_lineup(away_team_id, away_team)
        if not lineup_home: lineup_home, _ = await fetch_projected_lineup(home_team_id, home_team)

        all_batters = []

        async def process(batter, team, opp_p_id, opp_p_hand, opp_p_splits,
                          opp_p_mlb_cur, opp_p_mlb_pri, opp_p_savant,
                          opp_p_name, opp_p_pitches, is_proj):
            if "person" in batter:
                name = batter.get("person",{}).get("fullName","")
                pid  = batter.get("person",{}).get("id")
                bat_hand = batter.get("person",{}).get("batSide",{}).get("code","")
            else:
                name = batter.get("name",""); pid = batter.get("id"); bat_hand = ""

            if pid:
                info = await fetch_player_hand(pid)
                real_hand = info.get("bat_side","")
                if real_hand: bat_hand = real_hand
            if not bat_hand: bat_hand = "R"
            if bat_hand == "S": bat_hand = "L" if opp_p_hand == "R" else "R"

            bc = fuzzy_match(name, _cache["batting_current"])
            bp = fuzzy_match(name, _cache["batting_prior"])
            bc_dict = bc.to_dict() if bc is not None else {}
            bp_dict = bp.to_dict() if bp is not None else {}

            # Supplement Savant stats with MLB API for ISO, K%, pull%
            if pid:
                mlb_stats_cur = await fetch_batter_season_stats(pid, SEASON_CURRENT)
                mlb_stats_pri = await fetch_batter_season_stats(pid, SEASON_PRIOR)
                # Add to bc_dict/bp_dict if not already present
                for key in ["iso","k_pct","hr_rate"]:
                    if key not in bc_dict or float(bc_dict.get(key,0) or 0) == 0:
                        if mlb_stats_cur.get(key,0): bc_dict[key] = mlb_stats_cur[key]
                    if key not in bp_dict or float(bp_dict.get(key,0) or 0) == 0:
                        if mlb_stats_pri.get(key,0): bp_dict[key] = mlb_stats_pri[key]
                # Use MLB API PA count as primary if Savant is 0
                if float(bc_dict.get("pa",0) or 0) == 0:
                    bc_dict["pa"] = mlb_stats_cur.get("pa", 0)

            pa_current = float(bc_dict.get("pa",0) or 0)

            recent = await fetch_recent_form(pid, days=form_days) if pid else None
            bat_splits = None
            if pid:
                bat_splits = await fetch_batter_splits(pid, opp_p_hand, SEASON_CURRENT)
                if not bat_splits or bat_splits.get("pa",0) < 30:
                    prior_splits = await fetch_batter_splits(pid, opp_p_hand, SEASON_PRIOR)
                    if prior_splits and prior_splits.get("pa",0) >= 50:
                        bat_splits = prior_splits

            barrel_vs_pitches = {}
            if opp_p_pitches:
                pitch_codes = [p["code"] for p in opp_p_pitches]
                barrel_vs_pitches = get_batter_vs_pitches(name, pitch_codes)

            park_factor = get_park_hr_factor(home_team, bat_hand)
            weather_mult, weather_label = calc_weather_multiplier(home_team, wind_speed, wind_dir, temp, bat_hand)
            ip_current = float(opp_p_mlb_cur.get("ip",0) or 0)

            hr_prob, breakdown, archetype, trend_label, reasons = compute_hr_probability(
                bc_dict, bp_dict, recent, bat_hand, opp_p_hand,
                bat_splits, barrel_vs_pitches, opp_p_pitches,
                park_factor, weather_mult,
                opp_p_mlb_cur, opp_p_mlb_pri,
                opp_p_savant, opp_p_splits, ip_current, pa_current
            )

            splits_pa = bat_splits.get("pa",0) if bat_splits else 0
            confidence = get_confidence(pa_current, ip_current, is_proj, splits_pa)

            platoon_tag = None
            if opp_p_splits:
                vs_code = "vsl" if bat_hand == "L" else "vsr"
                split = opp_p_splits.get(vs_code,{})
                if split.get("ip",0) >= 10 and split.get("hr9",0) > 1.3:
                    platoon_tag = f"SP weak vs {'LHB' if bat_hand=='L' else 'RHB'} ({split['hr9']:.1f} HR/9)"

            pitch_matchup = []
            for pitch in opp_p_pitches:
                code = pitch["code"]
                vs_data = barrel_vs_pitches.get(code)
                pitch_matchup.append({
                    "name":pitch["name"],"pct":pitch["pct"],
                    "woba_vs": vs_data.get("woba") if vs_data else None,
                    "whiff_vs": vs_data.get("whiff") if vs_data else None,
                    "pa_vs": vs_data.get("pa") if vs_data else None,
                })

            all_batters.append({
                "name":name,"team":team,"hr_prob":hr_prob,
                "archetype":archetype,"trend":trend_label,"confidence":confidence,
                "reasons":reasons,"opp_pitcher":opp_p_name,
                "bat_hand":bat_hand,"opp_p_hand":opp_p_hand,
                "park_factor":round(park_factor,2),
                "statline":{
                    "barrel":round(breakdown["barrel_s"],1),
                    "ev":round(float(bc_dict.get("exit_velo",0) or 0),1),
                    "iso":round(breakdown["iso_use"],3),
                    "hr_fb":round(float(bc_dict.get("hr_fb_pct",0) or 0),1),
                },
                "recent_hr":recent.get("hr",0) if recent else 0,
                "recent_pa":recent.get("pa",0) if recent else 0,
                "recent_slg":round(recent.get("slg",0),3) if recent else 0,
                "dk_odds":fmt_odds(match_dk_odds(name, dk_props)),
                "projected":is_proj,"platoon_tag":platoon_tag,
                "pitch_matchup":[],"breakdown":breakdown,
            })

        away_proj = lineup_away_status == "projected"
        home_proj = lineup_home_status == "projected"
        for b in lineup_away:
            await process(b, away_team, home_p_id, home_p_hand, home_p_splits,
                         home_p_mlb_cur, home_p_mlb_pri, home_p_savant,
                         home_p.get("fullName","TBD"), home_p_pitches, away_proj)
        for b in lineup_home:
            await process(b, home_team, away_p_id, away_p_hand, away_p_splits,
                         away_p_mlb_cur, away_p_mlb_pri, away_p_savant,
                         away_p.get("fullName","TBD"), away_p_pitches, home_proj)

        all_batters.sort(key=lambda x: x["hr_prob"], reverse=True)

        def fmt_pit_display(mlb_cur, mlb_pri, splits, savant, name, hand, pitches):
            ip_cur = mlb_cur.get("ip",0)
            wc, wp = get_pitcher_blend_weights(ip_cur)
            era = mlb_cur.get("era",0) if ip_cur >= 3 else mlb_pri.get("era",0)
            hr9 = mlb_cur.get("hr9",0) if ip_cur >= 3 else mlb_pri.get("hr9",0)
            vs_l = splits.get("vsl",{})
            vs_r = splits.get("vsr",{})
            return {
                "name":name,"hand":hand,
                "era":round(era,2) if era else None,
                "hr9":round(hr9,2) if hr9 else None,
                "hard_hit_pct":round(savant.get("hard_hit_pct_allowed",0),1) or None,
                "barrel_pct":round(savant.get("barrel_pct_allowed",0),1) or None,
                "ip_2026":round(ip_cur,1),
                "blend_note":f"{int(wc*100)}% 2026 / {int(wp*100)}% 2025",
                "vs_L_hr9":round(vs_l.get("hr9",0),2) if vs_l.get("ip",0)>=5 else None,
                "vs_R_hr9":round(vs_r.get("hr9",0),2) if vs_r.get("ip",0)>=5 else None,
                "top_pitches":pitches,
            }

        wx_mult, wx_label = calc_weather_multiplier(home_team, wind_speed, wind_dir, temp)
        games_out.append({
            "game_id":gid,"away":away_team,"home":home_team,"time":gtime,
            "away_pitcher":fmt_pit_display(away_p_mlb_cur, away_p_mlb_pri, away_p_splits,
                                           away_p_savant, away_p.get("fullName","TBD"),
                                           away_p_hand, away_p_pitches),
            "home_pitcher":fmt_pit_display(home_p_mlb_cur, home_p_mlb_pri, home_p_splits,
                                           home_p_savant, home_p.get("fullName","TBD"),
                                           home_p_hand, home_p_pitches),
            "top_hr_candidates":all_batters[:3],
            "lineups_posted":lineup_away_status=="confirmed" or lineup_home_status=="confirmed",
            "lineup_away_status":lineup_away_status,
            "lineup_home_status":lineup_home_status,
            "weather":{"label":wx_label,"temp":temp,"wind_speed":wind_speed,"wind_dir":wind_dir}
        })

    return {"games":games_out,"date":today,"loading":False}

@app.post("/refresh-cache")
def refresh_cache():
    _cache["ready"] = False
    _cache["player_hands"] = {}
    threading.Thread(target=load_savant_data, daemon=True).start()
    return {"status":"Cache refresh started"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
