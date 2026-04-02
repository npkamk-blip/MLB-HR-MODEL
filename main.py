from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import pandas as pd
from pybaseball import batting_stats, pitching_stats, statcast_batter_pitch_arsenal
from datetime import date, timedelta
import threading
import uvicorn
import math
import os
from collections import defaultdict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SEASON_CURRENT = 2026
SEASON_PRIOR = 2025
MLB_API = "https://statsapi.mlb.com/api/v1"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

_cache = {
    "batting_current": pd.DataFrame(),
    "batting_prior": pd.DataFrame(),
    "pitching_current": pd.DataFrame(),
    "pitching_prior": pd.DataFrame(),
    "batter_arsenal": pd.DataFrame(),   # barrel rate vs pitch type 2026
    "player_hands": {},
    "ready": False
}

# Pitch type display names
PITCH_NAMES = {
    "FF": "Fastball", "FA": "Fastball", "SI": "Sinker", "SL": "Slider",
    "CH": "Changeup", "CU": "Curve", "FC": "Cutter", "FS": "Splitter",
    "ST": "Sweeper", "KC": "Knuckle-Curve", "KN": "Knuckleball", "EP": "Eephus",
}

# Park dimensions: {team: {lf, lcf, cf, rcf, rf, lf_wall, rf_wall}}
# Scoring uses: RHB pull to LF, LHB pull to RF
# League avg: LF=331, CF=405, RF=327
PARK_DIMS = {
    "Arizona Diamondbacks":   {"lf":330,"cf":407,"rf":334,"lf_wall":7.5,"rf_wall":7.5,"dome":True},
    "Atlanta Braves":         {"lf":335,"cf":400,"rf":325,"lf_wall":8,"rf_wall":8,"dome":False},
    "Baltimore Orioles":      {"lf":333,"cf":400,"rf":318,"lf_wall":7,"rf_wall":7,"dome":False},
    "Boston Red Sox":         {"lf":310,"cf":389,"rf":302,"lf_wall":37,"rf_wall":5,"dome":False},
    "Chicago Cubs":           {"lf":355,"cf":400,"rf":353,"lf_wall":8,"rf_wall":8,"dome":False},
    "Chicago White Sox":      {"lf":330,"cf":400,"rf":335,"lf_wall":8,"rf_wall":8,"dome":False},
    "Cincinnati Reds":        {"lf":328,"cf":404,"rf":325,"lf_wall":12,"rf_wall":9,"dome":False},
    "Cleveland Guardians":    {"lf":325,"cf":400,"rf":325,"lf_wall":9,"rf_wall":9,"dome":False},
    "Colorado Rockies":       {"lf":347,"cf":415,"rf":350,"lf_wall":8,"rf_wall":8,"dome":False},
    "Detroit Tigers":         {"lf":342,"cf":412,"rf":330,"lf_wall":8,"rf_wall":8,"dome":False},
    "Houston Astros":         {"lf":315,"cf":409,"rf":326,"lf_wall":9,"rf_wall":9,"dome":True},
    "Kansas City Royals":     {"lf":327,"cf":400,"rf":327,"lf_wall":8,"rf_wall":8,"dome":False},  # 2026 updated dims
    "Los Angeles Angels":     {"lf":347,"cf":396,"rf":350,"lf_wall":8,"rf_wall":8,"dome":False},
    "Los Angeles Dodgers":    {"lf":330,"cf":395,"rf":330,"lf_wall":8,"rf_wall":8,"dome":False},
    "Miami Marlins":          {"lf":344,"cf":400,"rf":335,"lf_wall":8,"rf_wall":8,"dome":True},
    "Milwaukee Brewers":      {"lf":342,"cf":400,"rf":337,"lf_wall":8,"rf_wall":8,"dome":True},
    "Minnesota Twins":        {"lf":339,"cf":404,"rf":328,"lf_wall":8,"rf_wall":8,"dome":False},
    "New York Mets":          {"lf":335,"cf":408,"rf":330,"lf_wall":8,"rf_wall":8,"dome":False},
    "New York Yankees":       {"lf":318,"cf":408,"rf":314,"lf_wall":8,"rf_wall":8,"dome":False},
    "Oakland Athletics":      {"lf":330,"cf":400,"rf":300,"lf_wall":8,"rf_wall":8,"dome":False},
    "Philadelphia Phillies":  {"lf":329,"cf":401,"rf":330,"lf_wall":8,"rf_wall":8,"dome":False},
    "Pittsburgh Pirates":     {"lf":325,"cf":399,"rf":320,"lf_wall":8,"rf_wall":8,"dome":False},
    "San Diego Padres":       {"lf":357,"cf":396,"rf":382,"lf_wall":8,"rf_wall":8,"dome":False},
    "San Francisco Giants":   {"lf":399,"cf":391,"rf":365,"lf_wall":8,"rf_wall":8,"dome":False},
    "Seattle Mariners":       {"lf":331,"cf":401,"rf":326,"lf_wall":8,"rf_wall":8,"dome":True},
    "St. Louis Cardinals":    {"lf":336,"cf":400,"rf":335,"lf_wall":8,"rf_wall":8,"dome":False},
    "Tampa Bay Rays":         {"lf":315,"cf":404,"rf":322,"lf_wall":8,"rf_wall":8,"dome":True},
    "Texas Rangers":          {"lf":329,"cf":407,"rf":326,"lf_wall":8,"rf_wall":8,"dome":True},
    "Toronto Blue Jays":      {"lf":328,"cf":400,"rf":328,"lf_wall":8,"rf_wall":8,"dome":True},
    "Washington Nationals":   {"lf":337,"cf":402,"rf":335,"lf_wall":8,"rf_wall":8,"dome":False},
}

# League averages for comparison
LG_AVG_LF = 331
LG_AVG_RF = 327

PARK_FACTORS = {
    "Colorado Rockies":       {"L": 1.35, "R": 1.35},
    "Cincinnati Reds":        {"L": 1.25, "R": 1.20},
    "Baltimore Orioles":      {"L": 1.20, "R": 1.15},
    "Philadelphia Phillies":  {"L": 1.15, "R": 1.10},
    "Boston Red Sox":         {"L": 1.10, "R": 1.12},
    "Chicago Cubs":           {"L": 1.10, "R": 1.08},
    "New York Yankees":       {"L": 1.15, "R": 1.10},
    "Atlanta Braves":         {"L": 1.08, "R": 1.10},
    "Texas Rangers":          {"L": 1.08, "R": 1.05},
    "Milwaukee Brewers":      {"L": 1.05, "R": 1.05},
    "Arizona Diamondbacks":   {"L": 1.05, "R": 1.05},
    "Toronto Blue Jays":      {"L": 1.03, "R": 1.05},
    "Houston Astros":         {"L": 1.02, "R": 1.00},
    "Los Angeles Dodgers":    {"L": 1.00, "R": 1.02},
    "Minnesota Twins":        {"L": 1.00, "R": 1.00},
    "Chicago White Sox":      {"L": 1.00, "R": 1.00},
    "Cleveland Guardians":    {"L": 0.98, "R": 0.97},
    "Kansas City Royals":     {"L": 1.02, "R": 1.02},  # updated for 2026 shorter fences
    "Detroit Tigers":         {"L": 0.97, "R": 0.95},
    "St. Louis Cardinals":    {"L": 0.96, "R": 0.97},
    "Washington Nationals":   {"L": 0.96, "R": 0.95},
    "Pittsburgh Pirates":     {"L": 0.95, "R": 0.97},
    "Tampa Bay Rays":         {"L": 0.93, "R": 0.93},
    "New York Mets":          {"L": 0.93, "R": 0.95},
    "Los Angeles Angels":     {"L": 0.95, "R": 0.92},
    "Seattle Mariners":       {"L": 0.90, "R": 0.92},
    "San Francisco Giants":   {"L": 0.90, "R": 0.88},
    "Miami Marlins":          {"L": 0.88, "R": 0.90},
    "Oakland Athletics":      {"L": 0.87, "R": 0.87},
    "San Diego Padres":       {"L": 0.86, "R": 0.88},
}

STADIUMS = {
    "Arizona Diamondbacks":   {"lat": 33.4453,  "lon": -112.0667, "dome": True},
    "Atlanta Braves":         {"lat": 33.8907,  "lon": -84.4677,  "dome": False, "hr_bearing": 225, "open_factor": 0.5},
    "Baltimore Orioles":      {"lat": 39.2838,  "lon": -76.6217,  "dome": False, "hr_bearing": 180, "open_factor": 0.6},
    "Boston Red Sox":         {"lat": 42.3467,  "lon": -71.0972,  "dome": False, "hr_bearing": 270, "open_factor": 0.7},
    "Chicago Cubs":           {"lat": 41.9484,  "lon": -87.6553,  "dome": False, "hr_bearing": 225, "open_factor": 1.0},
    "Chicago White Sox":      {"lat": 41.8299,  "lon": -87.6338,  "dome": False, "hr_bearing": 315, "open_factor": 0.5},
    "Cincinnati Reds":        {"lat": 39.0979,  "lon": -84.5082,  "dome": False, "hr_bearing": 270, "open_factor": 0.6},
    "Cleveland Guardians":    {"lat": 41.4954,  "lon": -81.6854,  "dome": False, "hr_bearing": 225, "open_factor": 0.6},
    "Colorado Rockies":       {"lat": 39.7559,  "lon": -104.9942, "dome": False, "hr_bearing": 270, "open_factor": 0.7},
    "Detroit Tigers":         {"lat": 42.3390,  "lon": -83.0485,  "dome": False, "hr_bearing": 135, "open_factor": 0.5},
    "Houston Astros":         {"lat": 29.7573,  "lon": -95.3555,  "dome": True},
    "Kansas City Royals":     {"lat": 39.0517,  "lon": -94.4803,  "dome": False, "hr_bearing": 180, "open_factor": 0.6},
    "Los Angeles Angels":     {"lat": 33.8003,  "lon": -117.8827, "dome": False, "hr_bearing": 270, "open_factor": 0.5},
    "Los Angeles Dodgers":    {"lat": 34.0739,  "lon": -118.2400, "dome": False, "hr_bearing": 315, "open_factor": 0.5},
    "Miami Marlins":          {"lat": 25.7781,  "lon": -80.2197,  "dome": True},
    "Milwaukee Brewers":      {"lat": 43.0282,  "lon": -87.9712,  "dome": True},
    "Minnesota Twins":        {"lat": 44.9817,  "lon": -93.2778,  "dome": False, "hr_bearing": 225, "open_factor": 0.6},
    "New York Mets":          {"lat": 40.7571,  "lon": -73.8458,  "dome": False, "hr_bearing": 270, "open_factor": 0.5},
    "New York Yankees":       {"lat": 40.8296,  "lon": -73.9262,  "dome": False, "hr_bearing": 270, "open_factor": 0.6},
    "Oakland Athletics":      {"lat": 38.5726,  "lon": -121.5088, "dome": False, "hr_bearing": 270, "open_factor": 0.5},
    "Philadelphia Phillies":  {"lat": 39.9056,  "lon": -75.1665,  "dome": False, "hr_bearing": 225, "open_factor": 0.5},
    "Pittsburgh Pirates":     {"lat": 40.4469,  "lon": -80.0057,  "dome": False, "hr_bearing": 270, "open_factor": 0.6},
    "San Diego Padres":       {"lat": 32.7076,  "lon": -117.1570, "dome": False, "hr_bearing": 270, "open_factor": 0.8},
    "San Francisco Giants":   {"lat": 37.7786,  "lon": -122.3893, "dome": False, "hr_bearing": 315, "open_factor": 0.9},
    "Seattle Mariners":       {"lat": 47.5914,  "lon": -122.3325, "dome": True},
    "St. Louis Cardinals":    {"lat": 38.6226,  "lon": -90.1928,  "dome": False, "hr_bearing": 225, "open_factor": 0.5},
    "Tampa Bay Rays":         {"lat": 27.7683,  "lon": -82.6534,  "dome": True},
    "Texas Rangers":          {"lat": 32.7473,  "lon": -97.0825,  "dome": True},
    "Toronto Blue Jays":      {"lat": 43.6414,  "lon": -79.3894,  "dome": True},
    "Washington Nationals":   {"lat": 38.8730,  "lon": -77.0074,  "dome": False, "hr_bearing": 180, "open_factor": 0.5},
}

BAT_COL_MAP = {
    "Name":"name","PA":"pa","HR":"hr","AVG":"avg","ISO":"iso",
    "Barrel%":"barrel_pct","EV":"exit_velo","LA":"launch_angle",
    "Hard%":"hard_hit_pct","FB%":"fb_pct","HR/FB":"hr_fb_pct","Pull%":"pull_pct",
}

PIT_COL_MAP = {
    "Name":"name","PA":"pa","TBF":"pa","HR":"hr","ERA":"era",
    "WHIP":"whip","HR/9":"hr_per9","HR/FB":"hr_fb_pct",
    "GB%":"gb_pct","FB%":"fb_pct","Hard%":"hard_hit_pct","HardHit%":"hard_hit_pct","IP":"ip",
}

# Pitch usage columns in FanGraphs pitching data (Statcast codes)
PITCH_USAGE_COLS = {
    "FA% (sc)":"FF","FT% (sc)":"FT","FC% (sc)":"FC","FS% (sc)":"FS",
    "SI% (sc)":"SI","SL% (sc)":"SL","CU% (sc)":"CU","CH% (sc)":"CH",
    "KC% (sc)":"KC","KN% (sc)":"KN",
}

def load_savant_data():
    print("Loading Statcast data...")
    for season, kb, kp in [(SEASON_CURRENT,"batting_current","pitching_current"),(SEASON_PRIOR,"batting_prior","pitching_prior")]:
        try:
            bat = batting_stats(season, qual=10)
            rename = {k:v for k,v in BAT_COL_MAP.items() if k in bat.columns}
            bat = bat.rename(columns=rename)
            for col in ["barrel_pct","hard_hit_pct","hr_fb_pct","pull_pct","fb_pct"]:
                if col in bat.columns:
                    bat[col] = pd.to_numeric(bat[col], errors="coerce").fillna(0)
                    if bat[col].median() < 1.0: bat[col] = bat[col] * 100
            _cache[kb] = bat
            print(f"{season} batting: {len(bat)} players")
        except Exception as e:
            print(f"{season} batting error: {e}")
        try:
            pit = pitching_stats(season, qual=5)
            rename = {k:v for k,v in PIT_COL_MAP.items() if k in pit.columns}
            pit = pit.rename(columns=rename)
            for col in ["hr_fb_pct","gb_pct","fb_pct","hard_hit_pct"]:
                if col in pit.columns:
                    pit[col] = pd.to_numeric(pit[col], errors="coerce").fillna(0)
                    if pit[col].median() < 1.0: pit[col] = pit[col] * 100
            _cache[kp] = pit
            print(f"{season} pitching: {len(pit)} pitchers")
        except Exception as e:
            print(f"{season} pitching error: {e}")

    # Load batter arsenal stats (barrel rate vs pitch type) for 2026
    try:
        arsenal = statcast_batter_pitch_arsenal(SEASON_CURRENT, minPA=10)
        print(f"Arsenal stats columns: {list(arsenal.columns[:20])}")
        _cache["batter_arsenal"] = arsenal
        print(f"Batter arsenal loaded: {len(arsenal)} rows")
    except Exception as e:
        print(f"Arsenal stats error: {e}")
        # Try with different min PA
        try:
            arsenal = statcast_batter_pitch_arsenal(SEASON_CURRENT, minPA=1)
            _cache["batter_arsenal"] = arsenal
            print(f"Batter arsenal loaded (minPA=1): {len(arsenal)} rows")
        except Exception as e2:
            print(f"Arsenal stats fallback error: {e2}")

    _cache["ready"] = True
    print("All data ready.")

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=load_savant_data, daemon=True).start()

def fuzzy_match(name, df):
    if df is None or df.empty or "name" not in df.columns: return None
    name_lower = name.lower()
    for _, row in df.iterrows():
        rn = str(row.get("name","")).lower()
        if rn == name_lower or rn in name_lower or name_lower in rn: return row
    parts = name_lower.split()
    if len(parts) >= 2:
        last = parts[-1]
        for _, row in df.iterrows():
            if last in str(row.get("name","")).lower(): return row
    return None

def get_pitcher_top_pitches(pitcher_name):
    """Get top 2 pitch types and usage % from FanGraphs pitching data"""
    pit_df = _cache["pitching_current"]
    if pit_df.empty:
        pit_df = _cache["pitching_prior"]
    row = fuzzy_match(pitcher_name, pit_df)
    if row is None:
        # Try prior season
        row = fuzzy_match(pitcher_name, _cache["pitching_prior"])
    if row is None:
        return []

    pitches = []
    for col, code in PITCH_USAGE_COLS.items():
        if col in row.index:
            val = float(row.get(col, 0) or 0)
            if val > 5:  # Only pitches thrown more than 5% of the time
                name = PITCH_NAMES.get(code, code)
                pitches.append({"code": code, "name": name, "pct": round(val, 1)})

    pitches.sort(key=lambda x: x["pct"], reverse=True)
    return pitches[:2]

def get_batter_barrel_vs_pitches(player_id, pitch_codes):
    """Get batter's barrel rate vs specific pitch types from arsenal stats"""
    arsenal = _cache["batter_arsenal"]
    if arsenal.empty or not pitch_codes:
        return {}

    try:
        # Arsenal stats indexed by player_id (mlb_id) and pitch_type
        cols = list(arsenal.columns)
        print(f"Arsenal cols: {cols[:15]}")

        # Find player rows
        id_col = None
        for c in ["player_id","mlb_id","batter","IDfg"]:
            if c in cols:
                id_col = c
                break

        if not id_col:
            return {}

        player_rows = arsenal[arsenal[id_col] == player_id]
        if player_rows.empty:
            return {}

        result = {}
        pitch_col = None
        for c in ["pitch_type","pitch_hand","pitch"]:
            if c in cols:
                pitch_col = c
                break

        barrel_col = None
        for c in ["brl_percent","barrel_pct","brl%","Barrel%"]:
            if c in cols:
                barrel_col = c
                break

        if not pitch_col or not barrel_col:
            return {}

        for _, row in player_rows.iterrows():
            pt = str(row.get(pitch_col,"")).upper()
            barrel = float(row.get(barrel_col, 0) or 0)
            if pt in pitch_codes:
                result[pt] = round(barrel, 1)

        return result
    except Exception as e:
        print(f"Arsenal lookup error: {e}")
        return {}

def get_park_dimension_bonus(home_team, bat_hand, pull_pct=40):
    """
    Calculate bonus/penalty based on park dimensions vs league average
    RHB pull to LF → compare LF to league avg 331
    LHB pull to RF → compare RF to league avg 327
    High pull rate batters get more weight
    """
    dims = PARK_DIMS.get(home_team)
    if not dims: return 0, ""

    if dims.get("dome"): return 0, ""

    pull_weight = min(pull_pct / 40.0, 1.5)  # normalize pull rate

    if bat_hand == "R":
        # Righty pulls to left field
        lf = dims.get("lf", LG_AVG_LF)
        diff = LG_AVG_LF - lf  # positive = shorter than avg = good
        bonus = round((diff / 20.0) * 5 * pull_weight, 1)
        if diff > 10: note = f"Short LF ({lf}ft)"
        elif diff < -10: note = f"Deep LF ({lf}ft)"
        else: note = ""
    else:
        # Lefty pulls to right field
        rf = dims.get("rf", LG_AVG_RF)
        diff = LG_AVG_RF - rf  # positive = shorter = good
        bonus = round((diff / 20.0) * 5 * pull_weight, 1)
        if diff > 10: note = f"Short RF ({rf}ft)"
        elif diff < -10: note = f"Deep RF ({rf}ft)"
        else: note = ""

    return bonus, note

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

def get_park_factor(home_team, batter_hand):
    pf = PARK_FACTORS.get(home_team, {"L":1.0,"R":1.0})
    return pf.get(batter_hand if batter_hand in ("L","R") else "R", 1.0)

def angle_diff(a, b):
    diff = abs(a-b) % 360
    return diff if diff <= 180 else 360-diff

def calc_weather_effect(home_team, wind_speed, wind_direction, temperature, batter_hand="R"):
    stadium = STADIUMS.get(home_team)
    if not stadium: return 0, "Unknown", f"{temperature}°F · {wind_speed} mph"
    if stadium.get("dome"): return 0, "Dome", f"Indoor · {temperature}°F"
    hr_bearing = stadium.get("hr_bearing", 225)
    open_factor = stadium.get("open_factor", 0.5)
    diff = angle_diff(wind_direction, hr_bearing)
    alignment = math.cos(math.radians(diff))
    if wind_speed < 5: speed_factor, wind_label = 0, "Calm"
    elif wind_speed < 10: speed_factor, wind_label = 0.3, f"{wind_speed} mph"
    elif wind_speed < 16: speed_factor, wind_label = 0.7, f"{wind_speed} mph"
    else: speed_factor, wind_label = 1.0, f"{wind_speed} mph"
    wind_effect = alignment * speed_factor * 10 * open_factor
    temp_effect = 3 if temperature >= 80 else 1 if temperature >= 70 else -3 if temperature < 50 else -1 if temperature < 60 else 0
    total = round(wind_effect + temp_effect)
    if abs(alignment) <= 0.5 and wind_speed >= 10:
        cross_diff = (wind_direction - hr_bearing) % 360
        blowing_l_to_r = 45 < cross_diff < 225
        if blowing_l_to_r:
            direction_label = "Favors Lefties"
            total += 3 if batter_hand == "L" else -2
        else:
            direction_label = "Favors Righties"
            total += 3 if batter_hand == "R" else -2
    elif alignment > 0.5 and wind_speed >= 10: direction_label = "Blowing Out"
    elif alignment < -0.5 and wind_speed >= 10: direction_label = "Blowing In"
    elif wind_speed < 5: direction_label = "Calm"
    else: direction_label = "Crosswind"
    return total, direction_label, f"{temperature}°F · {wind_label}"

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
    except Exception as e:
        print(f"Weather error: {e}"); return 70, 0, 0

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
                        "ops":float(s.get("ops",0) or 0),"hr_rate":(hr/max(pa,1))*600}
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
                result[code] = {"hr9":round((hr/ip)*9,2) if ip>0 else 0,
                                "era":float(s.get("era",0) or 0),
                                "whip":float(s.get("whip",0) or 0),
                                "ip":ip,"pa":int(s.get("battersFaced",0))}
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
        return {"era":float(s.get("era",0) or 0),"whip":float(s.get("whip",0) or 0),
                "hr9":round((hr/ip)*9,2) if ip>0 else 0,"ip":ip}
    except: return {}

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
        return {"hr":int(s.get("homeRuns",0)),"pa":int(s.get("plateAppearances",0)),"slg":slg,"iso":slg-avg}
    except: return None

def score_batter_detailed(c, p, recent, park_factor, bat_hand, opp_hand,
                           weather_bonus, park_dim_bonus, bat_splits=None,
                           barrel_vs_pitches=None, pitcher_top_pitches=None):
    pa_cur = float(c.get("pa",0) or 0)
    wc, wp = get_batter_blend_weights(pa_cur)

    if bat_splits and bat_splits.get("pa",0) >= 30:
        iso = bat_splits.get("iso",0) or blend_stat(c.get("iso"), p.get("iso"), wc, wp)
        hr_rate_split = bat_splits.get("hr_rate",0)
        split_used = True
        split_label = f"vs {'LHP' if opp_hand=='L' else 'RHP'} split"
    else:
        iso = blend_stat(c.get("iso"), p.get("iso"), wc, wp)
        hr_rate_split = 0
        split_used = False
        split_label = f"2025/26 blend ({int(wc*100)}%/{int(wp*100)}%)"

    barrel = blend_stat(c.get("barrel_pct"), p.get("barrel_pct"), wc, wp)
    hr_fb  = blend_stat(c.get("hr_fb_pct"),  p.get("hr_fb_pct"), wc, wp)
    ev     = blend_stat(c.get("exit_velo"),  p.get("exit_velo"), wc, wp)
    pull   = blend_stat(c.get("pull_pct"),   p.get("pull_pct"),  wc, wp)

    breakdown = []
    ss = 0

    if barrel > 0:
        pts = round(min(barrel/15.0,1.0)*25, 1)
        ss += pts
        breakdown.append({"label":f"Barrel rate {barrel:.1f}%","pts":pts,"max":25,"note":split_label})
    else:
        breakdown.append({"label":"Barrel rate","pts":0,"max":25,"note":"No data"})

    if iso > 0:
        pts = round(min(iso/0.280,1.0)*20, 1)
        ss += pts
        breakdown.append({"label":f"ISO .{int(iso*1000):03d}","pts":pts,"max":20,"note":split_label})
    else:
        breakdown.append({"label":"ISO","pts":0,"max":20,"note":"No data"})

    if hr_fb > 0:
        pts = round(min(hr_fb/25.0,1.0)*15, 1)
        ss += pts
        breakdown.append({"label":f"HR/FB {hr_fb:.1f}%","pts":pts,"max":15,"note":split_label})
    else:
        breakdown.append({"label":"HR/FB","pts":0,"max":15,"note":"No data"})

    if ev > 0:
        pts = round(min((ev-85)/15.0,1.0)*15, 1)
        ss += pts
        breakdown.append({"label":f"Exit velo {ev:.1f} mph","pts":pts,"max":15,"note":split_label})
    else:
        breakdown.append({"label":"Exit velo","pts":0,"max":15,"note":"No data"})

    if pull > 0:
        pts = round(min(pull/50.0,1.0)*5, 1)
        ss += pts
        breakdown.append({"label":f"Pull rate {pull:.1f}%","pts":pts,"max":5,"note":split_label})

    # Pitch matchup bonus — barrel rate vs pitcher's top pitches
    pitch_bonus = 0
    pitch_breakdown = []
    if barrel_vs_pitches and pitcher_top_pitches and barrel > 0:
        for pitch in pitcher_top_pitches:
            code = pitch["code"]
            usage = pitch["pct"] / 100.0
            brl_vs = barrel_vs_pitches.get(code)
            if brl_vs is not None:
                # Difference from overall barrel rate
                diff = brl_vs - barrel
                # Weight by pitch usage
                pts = round(diff * usage * 2, 1)
                pitch_bonus += pts
                pitch_breakdown.append({
                    "label": f"Barrel vs {pitch['name']} ({pitch['pct']}% usage): {brl_vs:.1f}% (overall {barrel:.1f}%)",
                    "pts": pts,
                    "max": 10
                })
        pitch_bonus = round(max(min(pitch_bonus, 10), -10), 1)
        ss += pitch_bonus
        breakdown.extend(pitch_breakdown)

    # Split bonus
    if split_used and hr_rate_split > 0:
        pts = round(min(hr_rate_split/30.0,1.0)*15, 1)
        ss += pts
        breakdown.append({"label":f"HR rate vs {'LHP' if opp_hand=='L' else 'RHP'} ({hr_rate_split:.0f}/600)","pts":pts,"max":15,"note":"split data"})
    elif opp_hand and bat_hand:
        if bat_hand != opp_hand: ss += 4
        else: ss -= 3

    season_score = round(ss, 1)

    fs = 0
    form_breakdown = []
    if recent and recent.get("pa",0) >= 5:
        rf_pa, rf_hr = recent.get("pa",1), recent.get("hr",0)
        rf_slg, rf_iso = recent.get("slg",0), recent.get("iso",0)
        hr_rate = (rf_hr/rf_pa)*100 if rf_pa > 0 else 0
        p1 = round(min(hr_rate/8.0,1.0)*50, 1)
        p2 = round(min(rf_slg/0.600,1.0)*30, 1)
        p3 = round(min(rf_iso/0.300,1.0)*20, 1)
        fs = p1+p2+p3
        form_breakdown = [
            {"label":f"HR rate last 14d ({rf_hr} HR/{rf_pa} PA)","pts":p1,"max":50},
            {"label":f"SLG last 14d ({rf_slg:.3f})","pts":p2,"max":30},
            {"label":f"ISO last 14d ({rf_iso:.3f})","pts":p3,"max":20},
        ]
    else:
        fs = season_score * 0.5
        form_breakdown = [{"label":"No recent form — using 50% of season score","pts":round(fs,1),"max":100}]

    blended = round(season_score * 0.70 + fs * 0.30, 1)
    park_adj = round(blended * park_factor - blended, 1)
    after_park = round(blended * park_factor, 1)
    after_weather = round(after_park + weather_bonus + park_dim_bonus, 1)

    reasons = []
    if barrel > 8: reasons.append(f"Barrel {barrel:.1f}%")
    if iso > 0.160: reasons.append(f"ISO .{int(iso*1000):03d}")
    if hr_fb > 14: reasons.append(f"HR/FB {hr_fb:.1f}%")
    if ev > 91: reasons.append(f"EV {ev:.1f} mph")
    if split_used and hr_rate_split > 15: reasons.append(f"Strong vs {'LHP' if opp_hand=='L' else 'RHP'}")
    elif bat_hand != opp_hand and not split_used: reasons.append(f"Platoon edge ({bat_hand} vs {opp_hand}P)")

    score_breakdown = {
        "season_components": breakdown,
        "season_subtotal": season_score,
        "form_components": form_breakdown,
        "form_subtotal": round(fs,1),
        "blend_note": f"70% season ({round(season_score*0.70,1)}) + 30% form ({round(fs*0.30,1)}) = {blended}",
        "blended": blended,
        "park_factor": park_factor,
        "park_adj": park_adj,
        "after_park": after_park,
        "weather_bonus": weather_bonus,
        "park_dim_bonus": park_dim_bonus,
        "after_weather": after_weather,
    }

    return round(min(after_weather,99)), reasons, {
        "barrel":round(barrel,1),"ev":round(ev,1),"iso":round(iso,3),"hr_fb":round(hr_fb,1),
        "split_used":split_used,
    }, score_breakdown

def score_pitcher_detailed(c, p, pit_splits=None, vs_hand=None, ip_current=0):
    ip_cur = float(ip_current or c.get("ip",0) or 0)
    wc, wp = get_pitcher_blend_weights(ip_cur)

    if pit_splits and vs_hand and vs_hand in pit_splits:
        split = pit_splits[vs_hand]
        split_ip = split.get("ip",0)
        if split_ip >= 5:
            hr9 = split.get("hr9",0)
            split_note = f"split vs {'LHB' if vs_hand=='vsl' else 'RHB'} ({split_ip:.0f} IP)"
        else:
            hr9 = blend_stat(c.get("hr_per9"), p.get("hr_per9"), wc, wp)
            split_note = f"overall ({ip_cur:.0f} IP {int(wc*100)}% 2026)"
    else:
        hr9 = blend_stat(c.get("hr_per9"), p.get("hr_per9"), wc, wp)
        split_note = f"overall ({ip_cur:.0f} IP {int(wc*100)}% 2026)"

    hrfb = blend_stat(c.get("hr_fb_pct"), p.get("hr_fb_pct"), wc, wp)
    gb   = blend_stat(c.get("gb_pct"),    p.get("gb_pct"),    wc, wp)
    hard = blend_stat(c.get("hard_hit_pct"), p.get("hard_hit_pct"), wc, wp)

    v, reasons, pit_breakdown = 0, [], []
    if hr9 > 0.5:
        pts = round(min(hr9/2.5,1.0)*10, 1)
        v += pts
        pit_breakdown.append({"label":f"HR/9 {hr9:.2f} ({split_note})","pts":pts,"max":10})
        if hr9 > 1.3: reasons.append(f"SP {hr9:.1f} HR/9")
    if hrfb > 0:
        pts = round(min(hrfb/20.0,1.0)*5, 1)
        v += pts
        pit_breakdown.append({"label":f"HR/FB {hrfb:.1f}%","pts":pts,"max":5})
    if gb > 5 and gb < 36:
        v += 5; pit_breakdown.append({"label":f"Fly ball pitcher (GB% {gb:.1f}%)","pts":5,"max":5})
        reasons.append("Fly ball SP")
    if hard > 5 and hard > 40:
        v += 3; pit_breakdown.append({"label":f"Hard contact {hard:.0f}%","pts":3,"max":3})
        reasons.append(f"SP {hard:.0f}% hard contact")

    return round(v,1), reasons, pit_breakdown

@app.get("/")
def root():
    return {"status":"MLB HR Model running","data_ready":_cache["ready"],"season":SEASON_CURRENT}

@app.get("/status")
def status():
    arsenal_count = len(_cache["batter_arsenal"]) if not _cache["batter_arsenal"].empty else 0
    return {"ready":_cache["ready"],
            "batters_2026":len(_cache["batting_current"]),
            "batters_2025":len(_cache["batting_prior"]),
            "pitchers_2026":len(_cache["pitching_current"]),
            "pitchers_2025":len(_cache["pitching_prior"]),
            "arsenal_rows":arsenal_count}

@app.get("/debug-arsenal")
def debug_arsenal():
    arsenal = _cache["batter_arsenal"]
    if arsenal.empty:
        return {"status":"empty","cols":[]}
    return {"cols":list(arsenal.columns),"rows":len(arsenal),"sample":arsenal.head(3).to_dict(orient="records")}

@app.get("/games")
async def get_games(form_days: int = 14):
    if not _cache["ready"]:
        return {"games":[],"date":date.today().isoformat(),"loading":True,
                "message":"Statcast data loading — try again in 60 seconds."}
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

        away_p_id = away_p.get("id")
        home_p_id = home_p.get("id")
        away_p_hand = home_p_hand = "R"
        away_p_cur = home_p_cur = {}
        away_p_pri = home_p_pri = {}
        away_p_splits = home_p_splits = {}
        away_p_pitches = home_p_pitches = []

        if away_p_id:
            info = await fetch_player_hand(away_p_id)
            away_p_hand = info.get("pitch_hand","R")
            away_p_cur = await fetch_pitcher_season_stats(away_p_id, SEASON_CURRENT)
            away_p_pri = await fetch_pitcher_season_stats(away_p_id, SEASON_PRIOR)
            away_p_splits = await fetch_pitcher_splits(away_p_id, SEASON_CURRENT)
            if not away_p_splits:
                away_p_splits = await fetch_pitcher_splits(away_p_id, SEASON_PRIOR)
            away_p_pitches = get_pitcher_top_pitches(away_p.get("fullName",""))

        if home_p_id:
            info = await fetch_player_hand(home_p_id)
            home_p_hand = info.get("pitch_hand","R")
            home_p_cur = await fetch_pitcher_season_stats(home_p_id, SEASON_CURRENT)
            home_p_pri = await fetch_pitcher_season_stats(home_p_id, SEASON_PRIOR)
            home_p_splits = await fetch_pitcher_splits(home_p_id, SEASON_CURRENT)
            if not home_p_splits:
                home_p_splits = await fetch_pitcher_splits(home_p_id, SEASON_PRIOR)
            home_p_pitches = get_pitcher_top_pitches(home_p.get("fullName",""))

        stadium = STADIUMS.get(home_team,{})
        temp, wind_speed, wind_dir = 70, 0, 0
        if not stadium.get("dome") and stadium.get("lat"):
            temp, wind_speed, wind_dir = await fetch_weather(stadium["lat"], stadium["lon"], gtime)

        def get_pit_df(name):
            c = fuzzy_match(name, _cache["pitching_current"])
            p = fuzzy_match(name, _cache["pitching_prior"])
            return c.to_dict() if c is not None else {}, p.to_dict() if p is not None else {}

        hpc, hpp = get_pit_df(home_p.get("fullName",""))
        apc, app_ = get_pit_df(away_p.get("fullName",""))

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

        if not lineup_away:
            lineup_away, _ = await fetch_projected_lineup(away_team_id, away_team)
        if not lineup_home:
            lineup_home, _ = await fetch_projected_lineup(home_team_id, home_team)

        all_batters = []

        async def process(batter, team, opp_p_id, opp_p_hand, opp_p_splits,
                          opp_p_cur, opp_p_pri, opp_p_name, pit_c, pit_p,
                          opp_p_pitches, is_proj):
            if "person" in batter:
                name     = batter.get("person",{}).get("fullName","")
                pid      = batter.get("person",{}).get("id")
                bat_hand = batter.get("person",{}).get("batSide",{}).get("code","")
            else:
                name = batter.get("name",""); pid = batter.get("id"); bat_hand = ""

            if pid:
                info = await fetch_player_hand(pid)
                real_hand = info.get("bat_side","")
                if real_hand: bat_hand = real_hand
            if not bat_hand: bat_hand = "R"
            if bat_hand == "S":
                bat_hand = "L" if opp_p_hand == "R" else "R"

            bc = fuzzy_match(name, _cache["batting_current"])
            bp = fuzzy_match(name, _cache["batting_prior"])
            recent = await fetch_recent_form(pid, days=form_days) if pid else None

            bat_splits = None
            if pid:
                bat_splits = await fetch_batter_splits(pid, opp_p_hand, SEASON_CURRENT)
                if not bat_splits or bat_splits.get("pa",0) < 30:
                    prior_splits = await fetch_batter_splits(pid, opp_p_hand, SEASON_PRIOR)
                    if prior_splits and prior_splits.get("pa",0) >= 50:
                        bat_splits = prior_splits

            # Get barrel rate vs pitcher's top pitches
            barrel_vs_pitches = {}
            if pid and opp_p_pitches and not _cache["batter_arsenal"].empty:
                pitch_codes = [p["code"] for p in opp_p_pitches]
                barrel_vs_pitches = get_batter_barrel_vs_pitches(pid, pitch_codes)

            vs_hand_code = "vsl" if bat_hand == "L" else "vsr"
            ip_current = opp_p_cur.get("ip",0)
            pit_vuln, pit_reasons, pit_breakdown = score_pitcher_detailed(
                pit_c, pit_p, opp_p_splits, vs_hand_code, ip_current
            )

            pf = get_park_factor(home_team, bat_hand)
            weather_bonus, weather_label, _ = calc_weather_effect(home_team, wind_speed, wind_dir, temp, bat_hand)

            # Park dimensions bonus
            bc_dict = bc.to_dict() if bc is not None else {}
            pull_pct = float(bc_dict.get("pull_pct", 0) or 0)
            park_dim_bonus, park_dim_note = get_park_dimension_bonus(home_team, bat_hand, pull_pct)

            score, reasons, statline, score_breakdown = score_batter_detailed(
                bc_dict,
                bp.to_dict() if bp is not None else {},
                recent, pf, bat_hand, opp_p_hand, weather_bonus, park_dim_bonus,
                bat_splits, barrel_vs_pitches, opp_p_pitches
            )

            final_score = min(99, score + pit_vuln)
            score_breakdown["pitcher_vuln"] = pit_vuln
            score_breakdown["pitcher_breakdown"] = pit_breakdown
            score_breakdown["park_dim_bonus"] = park_dim_bonus
            score_breakdown["park_dim_note"] = park_dim_note
            score_breakdown["final"] = final_score

            reasons += pit_reasons
            if pf >= 1.10: reasons.append("HR-friendly park")
            elif pf <= 0.90: reasons.append("Pitcher-friendly park")
            if park_dim_note: reasons.append(park_dim_note)

            platoon_tag = None
            if opp_p_splits and vs_hand_code in opp_p_splits:
                split = opp_p_splits[vs_hand_code]
                if split.get("ip",0) >= 10 and split.get("hr9",0) > 1.3:
                    platoon_tag = f"SP weak vs {'LHB' if bat_hand=='L' else 'RHB'} ({split['hr9']:.1f} HR/9)"

            # Build pitch matchup display
            pitch_matchup = []
            for pitch in opp_p_pitches:
                code = pitch["code"]
                brl = barrel_vs_pitches.get(code)
                pitch_matchup.append({
                    "name": pitch["name"],
                    "pct": pitch["pct"],
                    "barrel_vs": brl,  # None if no data
                })

            split_display = None
            if bat_splits and bat_splits.get("pa",0) >= 30:
                split_display = {
                    "vs": f"vs {'LHP' if opp_p_hand=='L' else 'RHP'}",
                    "hr": bat_splits.get("hr",0),
                    "pa": bat_splits.get("pa",0),
                    "iso": round(bat_splits.get("iso",0),3),
                    "slg": round(bat_splits.get("slg",0),3),
                }

            all_batters.append({
                "name":name,"team":team,"score":final_score,"reasons":reasons,
                "opp_pitcher":opp_p_name,"bat_hand":bat_hand,"opp_p_hand":opp_p_hand,
                "park_factor":round(pf,2),"statline":statline,
                "recent_hr":recent.get("hr",0) if recent else 0,
                "recent_pa":recent.get("pa",0) if recent else 0,
                "recent_slg":round(recent.get("slg",0),3) if recent else 0,
                "dk_odds":fmt_odds(match_dk_odds(name, dk_props)),
                "projected":is_proj,
                "split_display":split_display,
                "platoon_tag":platoon_tag,
                "pitch_matchup":pitch_matchup,
                "score_breakdown":score_breakdown,
            })

        away_proj = lineup_away_status == "projected"
        home_proj = lineup_home_status == "projected"

        for b in lineup_away:
            await process(b, away_team, home_p_id, home_p_hand, home_p_splits,
                         home_p_cur, home_p_pri, home_p.get("fullName","TBD"),
                         hpc, hpp, home_p_pitches, away_proj)
        for b in lineup_home:
            await process(b, home_team, away_p_id, away_p_hand, away_p_splits,
                         away_p_cur, away_p_pri, away_p.get("fullName","TBD"),
                         apc, app_, away_p_pitches, home_proj)

        all_batters.sort(key=lambda x: x["score"], reverse=True)

        def fmt_pit_display(cur_stats, pri_stats, splits, name, hand, pitches):
            ip_cur = cur_stats.get("ip",0)
            wc, wp = get_pitcher_blend_weights(ip_cur)
            era = cur_stats.get("era",0) if ip_cur >= 3 else pri_stats.get("era",0)
            hr9 = cur_stats.get("hr9",0) if ip_cur >= 3 else pri_stats.get("hr9",0)
            vs_l = splits.get("vsl",{})
            vs_r = splits.get("vsr",{})
            return {
                "name":name,"hand":hand,
                "era":round(era,2) if era else None,
                "hr9":round(hr9,2) if hr9 else None,
                "ip_2026":round(ip_cur,1),
                "blend_note":f"{int(wc*100)}% 2026 / {int(wp*100)}% 2025",
                "vs_L_hr9":round(vs_l.get("hr9",0),2) if vs_l.get("ip",0)>=5 else None,
                "vs_R_hr9":round(vs_r.get("hr9",0),2) if vs_r.get("ip",0)>=5 else None,
                "top_pitches":pitches,
            }

        wx_bonus, wx_label, wx_desc = calc_weather_effect(home_team, wind_speed, wind_dir, temp)
        games_out.append({
            "game_id":gid,"away":away_team,"home":home_team,"time":gtime,
            "away_pitcher":fmt_pit_display(away_p_cur, away_p_pri, away_p_splits,
                                           away_p.get("fullName","TBD"), away_p_hand, away_p_pitches),
            "home_pitcher":fmt_pit_display(home_p_cur, home_p_pri, home_p_splits,
                                           home_p.get("fullName","TBD"), home_p_hand, home_p_pitches),
            "top_hr_candidates":all_batters[:3],
            "lineups_posted":lineup_away_status=="confirmed" or lineup_home_status=="confirmed",
            "lineup_away_status":lineup_away_status,
            "lineup_home_status":lineup_home_status,
            "weather":{"label":wx_label,"desc":wx_desc,"temp":temp,"wind_speed":wind_speed,"wind_dir":wind_dir}
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
