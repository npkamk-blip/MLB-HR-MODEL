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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SEASON_CURRENT = 2026
SEASON_PRIOR = 2025
MLB_API = "https://statsapi.mlb.com/api/v1"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

_cache = {
    "batting_current": pd.DataFrame(),
    "batting_prior": pd.DataFrame(),
    "pitching_current": pd.DataFrame(),
    "pitching_prior": pd.DataFrame(),
    "batter_arsenal": pd.DataFrame(),
    "player_hands": {},
    "ready": False
}

PITCH_NAMES = {
    "FF":"Fastball","FA":"Fastball","SI":"Sinker","SL":"Slider",
    "CH":"Changeup","CU":"Curve","FC":"Cutter","FS":"Splitter",
    "ST":"Sweeper","KC":"Knuckle-Curve","KN":"Knuckleball",
}

PITCH_USAGE_COLS = {
    "FA% (sc)":"FF","FT% (sc)":"FT","FC% (sc)":"FC","FS% (sc)":"FS",
    "SI% (sc)":"SI","SL% (sc)":"SL","CU% (sc)":"CU","CH% (sc)":"CH",
    "KC% (sc)":"KC","KN% (sc)":"KN",
}

# Park HR factors by batter hand — HR-specific, not general offense
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

PIT_COL_MAP = {
    "Name":"name","PA":"pa","TBF":"pa","HR":"hr","ERA":"era",
    "WHIP":"whip","HR/9":"hr_per9","HR/FB":"hr_fb_pct",
    "GB%":"gb_pct","FB%":"fb_pct","Hard%":"hard_hit_pct",
    "HardHit%":"hard_hit_pct","IP":"ip","K%":"k_pct","Barrel%":"barrel_pct_allowed",
}

def load_savant_data():
    print("Loading Statcast data...")
    for season, kb, kp in [(SEASON_CURRENT,"batting_current","pitching_current"),
                           (SEASON_PRIOR,"batting_prior","pitching_prior")]:
        try:
            bat = batting_stats(season, qual=10)
            rename = {k:v for k,v in BAT_COL_MAP.items() if k in bat.columns}
            bat = bat.rename(columns=rename)
            for col in ["barrel_pct","hard_hit_pct","hr_fb_pct","pull_pct","fb_pct","k_pct"]:
                if col in bat.columns:
                    bat[col] = pd.to_numeric(bat[col], errors="coerce").fillna(0)
                    if bat[col].median() < 1.0: bat[col] = bat[col] * 100
            _cache[kb] = bat
            print(f"{season} batting: {len(bat)} players")
        except Exception as e:
            print(f"{season} batting error: {e}")
        try:
            pit = pitching_stats(season, qual=1)
            rename = {k:v for k,v in PIT_COL_MAP.items() if k in pit.columns}
            pit = pit.rename(columns=rename)
            for col in ["hr_fb_pct","gb_pct","fb_pct","hard_hit_pct","barrel_pct_allowed"]:
                if col in pit.columns:
                    pit[col] = pd.to_numeric(pit[col], errors="coerce").fillna(0)
                    if pit[col].median() < 1.0: pit[col] = pit[col] * 100
            _cache[kp] = pit
            print(f"{season} pitching: {len(pit)} pitchers")
        except Exception as e:
            print(f"{season} pitching error: {e}")

    # Batter arsenal — barrel rate vs pitch type 2026
    try:
        arsenal = statcast_batter_pitch_arsenal(SEASON_CURRENT, minPA=10)
        _cache["batter_arsenal"] = arsenal
        print(f"Arsenal loaded: {len(arsenal)} rows, cols: {list(arsenal.columns[:10])}")
    except Exception as e:
        print(f"Arsenal error: {e}")
        try:
            arsenal = statcast_batter_pitch_arsenal(SEASON_CURRENT, minPA=1)
            _cache["batter_arsenal"] = arsenal
            print(f"Arsenal loaded (minPA=1): {len(arsenal)} rows")
        except Exception as e2:
            print(f"Arsenal fallback error: {e2}")

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
    wind_mult = 1.0 + (alignment * speed_factor * 0.08 * open_factor)
    temp_mult = 1.05 if temperature >= 80 else 1.02 if temperature >= 70 else 0.93 if temperature < 50 else 0.97 if temperature < 60 else 1.0
    # Crosswind handedness
    direction_label = "Calm"
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
    """
    Convert raw score (0-100) to HR probability (2-25%)
    Calibration anchors:
      20 → ~2%, 45 → ~5%, 60 → ~9%, 75 → ~14%, 85 → ~20%, 90+ → ~25%
    """
    # Center and scale so midpoint ~50 maps to ~5-6%
    centered = (raw_score - 50) / 18.0
    sigmoid = 1 / (1 + math.exp(-centered))
    # Scale to 2-25% range
    prob = 0.02 + sigmoid * 0.25
    return round(min(max(prob, 0.02), 0.25) * 100, 1)

def get_archetype(barrel_pct, k_pct, fb_pct, iso):
    """Classify batter archetype based on stats"""
    if barrel_pct >= 10 and k_pct >= 28:
        return "Boom/Bust"
    elif barrel_pct >= 10 and k_pct < 22:
        return "Pure Power"
    elif barrel_pct >= 7 and fb_pct >= 38:
        return "Power"
    elif iso >= 0.180 and k_pct < 20:
        return "Balanced"
    elif k_pct >= 28:
        return "High K"
    else:
        return "Contact"

def get_trend(barrel_14d, barrel_season, hard_14d, hard_season):
    """Determine if batter is heating up, cooling off, or steady"""
    if barrel_season == 0: return "Steady", None
    barrel_diff = barrel_14d - barrel_season if barrel_14d > 0 else 0
    hard_diff = hard_14d - hard_season if hard_14d > 0 else 0
    combined = (barrel_diff * 0.6) + (hard_diff * 0.4)
    if combined >= 3: return "Heating Up", round(combined, 1)
    elif combined <= -3: return "Cooling Off", round(combined, 1)
    else: return "Steady", round(combined, 1)

def get_confidence(pa_current, ip_pitcher, is_projected, bat_splits_pa):
    """Confidence in the probability estimate"""
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

def get_pitcher_top_pitches(pitcher_name):
    pit_df = _cache["pitching_current"]
    if pit_df.empty: pit_df = _cache["pitching_prior"]
    row = fuzzy_match(pitcher_name, pit_df)
    if row is None: row = fuzzy_match(pitcher_name, _cache["pitching_prior"])
    if row is None: return []
    pitches = []
    for col, code in PITCH_USAGE_COLS.items():
        if col in row.index:
            val = float(row.get(col, 0) or 0)
            if val > 5:
                pitches.append({"code":code,"name":PITCH_NAMES.get(code,code),"pct":round(val,1)})
    pitches.sort(key=lambda x: x["pct"], reverse=True)
    return pitches[:2]

def get_batter_barrel_vs_pitches(player_id, pitch_codes):
    arsenal = _cache["batter_arsenal"]
    if arsenal.empty or not pitch_codes: return {}
    try:
        cols = list(arsenal.columns)
        id_col = next((c for c in ["player_id","mlb_id","batter","IDfg"] if c in cols), None)
        pitch_col = next((c for c in ["pitch_type","pitch_hand","pitch"] if c in cols), None)
        barrel_col = next((c for c in ["brl_percent","barrel_pct","brl%","Barrel%"] if c in cols), None)
        if not id_col or not pitch_col or not barrel_col: return {}
        player_rows = arsenal[arsenal[id_col] == player_id]
        if player_rows.empty: return {}
        result = {}
        for _, row in player_rows.iterrows():
            pt = str(row.get(pitch_col,"")).upper()
            if pt in pitch_codes:
                result[pt] = round(float(row.get(barrel_col,0) or 0), 1)
        return result
    except Exception as e:
        print(f"Arsenal lookup error: {e}"); return {}

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
                    "ip":ip, "pa":int(s.get("battersFaced",0)),
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
        return {"era":float(s.get("era",0) or 0),"whip":float(s.get("whip",0) or 0),
                "hr9":round((hr/ip)*9,2) if ip>0 else 0,"ip":ip}
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
        # We need barrel and hard hit from statcast for recent form
        # Approximate from available data
        return {
            "hr":hr,"pa":pa,"slg":slg,"iso":slg-avg,
            "hr_rate":(hr/max(pa,1))*600,
            "ops":float(s.get("ops",0) or 0),
        }
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

def compute_hr_probability(
    bc, bp, recent, bat_hand, opp_hand,
    bat_splits, barrel_vs_pitches, pitcher_top_pitches,
    park_factor, weather_mult, pit_c, pit_p,
    pit_splits, ip_current, pa_current
):
    """
    Full 5-step HR probability model.
    Returns: (hr_prob_pct, breakdown_dict, archetype, trend_label, reasons)
    """
    wc, wp = get_batter_blend_weights(pa_current)
    pit_wc, pit_wp = get_pitcher_blend_weights(ip_current)

    # ── STEP 1: Batter HR Score (0-60) ──
    barrel_s = blend_stat(bc.get("barrel_pct"), bp.get("barrel_pct"), wc, wp)
    hard_s   = blend_stat(bc.get("hard_hit_pct"), bp.get("hard_hit_pct"), wc, wp)
    fb_s     = blend_stat(bc.get("fb_pct"), bp.get("fb_pct"), wc, wp)
    pull_s   = blend_stat(bc.get("pull_pct"), bp.get("pull_pct"), wc, wp)
    iso_s    = blend_stat(bc.get("iso"), bp.get("iso"), wc, wp)
    k_s      = blend_stat(bc.get("k_pct"), bp.get("k_pct"), wc, wp)
    la_s     = float(bc.get("launch_angle",0) or 0)

    # Use split ISO if available
    if bat_splits and bat_splits.get("pa",0) >= 30:
        iso_use = bat_splits.get("iso", iso_s)
    else:
        iso_use = iso_s

    s1_barrel = round(min(barrel_s/15.0,1.0)*20, 2)   # 20pts max
    s1_hard   = round(min(hard_s/45.0,1.0)*10, 2)     # 10pts max
    s1_iso    = round(min(iso_use/0.280,1.0)*10, 2)   # 10pts max
    s1_fb     = round(min(fb_s/45.0,1.0)*8, 2)        # 8pts max
    s1_pull   = round(min(pull_s/50.0,1.0)*7, 2)      # 7pts max
    s1_la     = round(min(max(la_s-10,0)/20.0,1.0)*5, 2) if la_s > 0 else 0  # 5pts max

    batter_score = s1_barrel + s1_hard + s1_iso + s1_fb + s1_pull + s1_la

    # Recent form barrel trend
    barrel_14d = 0.0
    hard_14d   = 0.0
    hr_rate_14d = 0.0
    if recent and recent.get("pa",0) >= 5:
        hr_rate_14d = recent.get("hr_rate", 0)
        # Approximate barrel trend from HR rate (no direct barrel in recent form API)
        # If hitting HRs at elevated rate, boost batter score slightly
        if hr_rate_14d > 30:   # >30 HR/600 PA rate = very hot
            batter_score = min(batter_score * 1.15, 60)
        elif hr_rate_14d > 15:
            batter_score = min(batter_score * 1.07, 60)
        elif hr_rate_14d < 3 and recent.get("pa",0) >= 15:
            batter_score = batter_score * 0.95

    # Archetype
    archetype = get_archetype(barrel_s, k_s, fb_s, iso_s)
    trend_label, trend_diff = get_trend(barrel_14d, barrel_s, hard_14d, hard_s)

    # ── STEP 2: Pitcher HR Modifier (multiplier) ──
    pit_hr9  = blend_stat(pit_c.get("hr_per9"), pit_p.get("hr_per9"), pit_wc, pit_wp)
    pit_hrfb = blend_stat(pit_c.get("hr_fb_pct"), pit_p.get("hr_fb_pct"), pit_wc, pit_wp)
    pit_hard = blend_stat(pit_c.get("hard_hit_pct"), pit_p.get("hard_hit_pct"), pit_wc, pit_wp)
    pit_fb   = blend_stat(pit_c.get("fb_pct"), pit_p.get("fb_pct"), pit_wc, pit_wp)
    pit_brl  = blend_stat(pit_c.get("barrel_pct_allowed"), pit_p.get("barrel_pct_allowed"), pit_wc, pit_wp)

    # Use splits if available
    vs_hand_code = "vsl" if bat_hand == "L" else "vsr"
    if pit_splits and vs_hand_code in pit_splits:
        split = pit_splits[vs_hand_code]
        if split.get("ip",0) >= 10:
            pit_hr9 = split.get("hr9", pit_hr9)

    # Convert to multiplier (1.0 = average, >1.0 = more vulnerable)
    # League avg: HR/9 ~1.15, HR/FB ~11%, HardHit ~32%, FB% ~36%, Barrel% ~8%
    m_hr9  = 1.0 + (pit_hr9 - 1.15) / 1.15 * 0.35  if pit_hr9 > 0 else 1.0
    m_hrfb = 1.0 + (pit_hrfb - 11.0) / 11.0 * 0.30 if pit_hrfb > 0 else 1.0
    m_hard = 1.0 + (pit_hard - 32.0) / 32.0 * 0.20 if pit_hard > 0 else 1.0
    m_fb   = 1.0 + (pit_fb - 36.0) / 36.0 * 0.15   if pit_fb > 0 else 1.0

    # Weighted average of multipliers
    if pit_hr9 > 0 or pit_hrfb > 0:
        pit_modifier = (m_hr9*0.35 + m_hrfb*0.30 + m_hard*0.20 + m_fb*0.15)
    else:
        pit_modifier = 1.0

    pit_modifier = round(max(min(pit_modifier, 1.6), 0.6), 3)

    # Pitch matchup bonus
    pitch_bonus = 0.0
    pitch_breakdown = []
    if barrel_vs_pitches and pitcher_top_pitches and barrel_s > 0:
        for pitch in pitcher_top_pitches:
            code = pitch["code"]
            usage = pitch["pct"] / 100.0
            brl_vs = barrel_vs_pitches.get(code)
            if brl_vs is not None:
                diff = brl_vs - barrel_s
                pts = diff * usage * 2
                pitch_bonus += pts
                pitch_breakdown.append({
                    "name": pitch["name"],
                    "pct": pitch["pct"],
                    "barrel_vs": brl_vs,
                    "diff": round(diff,1),
                    "bonus": round(pts,1)
                })
        pitch_bonus = max(min(pitch_bonus, 8), -8)

    after_pitch = min(batter_score + pitch_bonus, 60)

    # ── STEP 3: Contact Gate (K% ceiling) ──
    k_cap = 1.0
    if k_s >= 35: k_cap = 0.75
    elif k_s >= 30: k_cap = 0.88
    elif k_s >= 28: k_cap = 0.94

    after_k = after_pitch * k_cap

    # ── STEP 4: Context multipliers ──
    after_context = after_k * pit_modifier * park_factor * weather_mult

    # ── STEP 5: Sigmoid → HR% ──
    hr_prob = sigmoid_to_prob(after_context)

    reasons = []
    if barrel_s > 10: reasons.append(f"Barrel {barrel_s:.1f}%")
    if iso_use > 0.200: reasons.append(f"ISO .{int(iso_use*1000):03d}")
    if pit_hr9 > 1.3: reasons.append(f"SP {pit_hr9:.1f} HR/9")
    if pit_hrfb > 15: reasons.append(f"SP {pit_hrfb:.1f}% HR/FB")
    if pit_fb < 36 and pit_fb > 5: reasons.append("Fly ball SP")
    if park_factor >= 1.15: reasons.append("HR-friendly park")
    elif park_factor <= 0.90: reasons.append("Pitcher-friendly park")

    breakdown = {
        # Step 1
        "barrel_s": round(barrel_s,1), "s1_barrel": s1_barrel,
        "hard_s": round(hard_s,1), "s1_hard": s1_hard,
        "iso_use": round(iso_use,3), "s1_iso": s1_iso,
        "fb_s": round(fb_s,1), "s1_fb": s1_fb,
        "pull_s": round(pull_s,1), "s1_pull": s1_pull,
        "la_s": round(la_s,1), "s1_la": s1_la,
        "batter_score": round(batter_score,1),
        "k_s": round(k_s,1), "k_cap": k_cap,
        # Step 2
        "pit_hr9": round(pit_hr9,2), "pit_hrfb": round(pit_hrfb,1),
        "pit_hard": round(pit_hard,1), "pit_fb": round(pit_fb,1),
        "pit_modifier": pit_modifier,
        "pitch_bonus": round(pitch_bonus,1),
        "pitch_breakdown": pitch_breakdown,
        # Step 3-4
        "after_pitch": round(after_pitch,1),
        "after_k": round(after_k,1),
        "park_factor": park_factor,
        "weather_mult": weather_mult,
        "after_context": round(after_context,1),
        # Step 5
        "hr_prob": hr_prob,
        # Blend info
        "blend_note": f"{int(wc*100)}% 2026 / {int(wp*100)}% 2025 ({int(pa_current)} PA)",
        "pit_blend_note": f"{int(pit_wc*100)}% 2026 / {int(pit_wp*100)}% 2025 ({ip_current:.0f} IP)",
    }

    return hr_prob, breakdown, archetype, trend_label, reasons

@app.get("/")
def root():
    return {"status":"Sharp MLB HR Model","data_ready":_cache["ready"],"season":SEASON_CURRENT}

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
    if arsenal.empty: return {"status":"empty","cols":[]}
    return {"cols":list(arsenal.columns),"rows":len(arsenal),
            "sample":arsenal.head(2).to_dict(orient="records")}

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
            if bat_hand == "S": bat_hand = "L" if opp_p_hand == "R" else "R"

            bc = fuzzy_match(name, _cache["batting_current"])
            bp = fuzzy_match(name, _cache["batting_prior"])
            bc_dict = bc.to_dict() if bc is not None else {}
            bp_dict = bp.to_dict() if bp is not None else {}
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
            if pid and opp_p_pitches and not _cache["batter_arsenal"].empty:
                pitch_codes = [p["code"] for p in opp_p_pitches]
                barrel_vs_pitches = get_batter_barrel_vs_pitches(pid, pitch_codes)

            park_factor = get_park_hr_factor(home_team, bat_hand)
            weather_mult, weather_label = calc_weather_multiplier(
                home_team, wind_speed, wind_dir, temp, bat_hand
            )
            ip_current = float(opp_p_cur.get("ip",0) or 0)

            # If FanGraphs pitching data missing, build fallback from MLB API stats
            pit_c_use = dict(pit_c) if pit_c else {}
            pit_p_use = dict(pit_p) if pit_p else {}

            # Fallback: inject MLB API stats when FanGraphs returns zeros
            if ip_current > 0 and pit_c_use.get("hr_per9",0) == 0:
                pit_c_use["hr_per9"] = opp_p_cur.get("hr9", 0)
                pit_c_use["era"] = opp_p_cur.get("era", 0)
                pit_c_use["ip"] = ip_current
            if pit_p_use.get("hr_per9",0) == 0:
                pit_p_use["hr_per9"] = opp_p_pri.get("hr9", 0)
                pit_p_use["era"] = opp_p_pri.get("era", 0)

            hr_prob, breakdown, archetype, trend_label, reasons = compute_hr_probability(
                bc_dict, bp_dict, recent, bat_hand, opp_p_hand,
                bat_splits, barrel_vs_pitches, opp_p_pitches,
                park_factor, weather_mult, pit_c_use, pit_p_use,
                opp_p_splits, ip_current, pa_current
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
                brl = barrel_vs_pitches.get(code)
                pitch_matchup.append({"name":pitch["name"],"pct":pitch["pct"],"barrel_vs":brl})

            all_batters.append({
                "name":name,"team":team,
                "hr_prob":hr_prob,
                "archetype":archetype,
                "trend":trend_label,
                "confidence":confidence,
                "reasons":reasons,
                "opp_pitcher":opp_p_name,
                "bat_hand":bat_hand,"opp_p_hand":opp_p_hand,
                "park_factor":round(park_factor,2),
                "statline":{
                    "barrel":round(breakdown["barrel_s"],1),
                    "ev":round(float(bc_dict.get("exit_velo",0) or 0),1),
                    "iso":round(breakdown["iso_use"],3),
                    "hr_fb":round(float(bc_dict.get("hr_fb_pct",0) or 0),1),
                },
                "dk_odds":fmt_odds(match_dk_odds(name, dk_props)),
                "projected":is_proj,
                "platoon_tag":platoon_tag,
                "pitch_matchup":pitch_matchup,
                "breakdown":breakdown,
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

        all_batters.sort(key=lambda x: x["hr_prob"], reverse=True)

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

        wx_mult, wx_label = calc_weather_multiplier(home_team, wind_speed, wind_dir, temp)
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
