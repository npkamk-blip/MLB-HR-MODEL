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

# ── Baseball Savant URLs ──
SAVANT_BASE = "https://baseballsavant.mlb.com"

def savant_batter_url(year=None, min_pa=10, extra=""):
    yr = year or current_season()
    return (f"{SAVANT_BASE}/leaderboard/custom?year={yr}&type=batter&filter=&sort=4"
            f"&sortDir=desc&min={min_pa}&selections=pa,ab,hit,home_run,strikeout,"
            f"k_percent,slg_percent,batting_avg,barrel_batted_rate,exit_velocity_avg,"
            f"launch_angle_avg,hard_hit_percent,pull_percent{extra}&csv=true")

def savant_pitcher_url(year=None, min_pa=5, extra=""):
    yr = year or current_season()
    return (f"{SAVANT_BASE}/leaderboard/custom?year={yr}&type=pitcher&filter=&sort=4"
            f"&sortDir=desc&min={min_pa}&selections=pa,home_run,barrel_batted_rate,"
            f"exit_velocity_avg,hard_hit_percent,k_percent,p_era{extra}&csv=true")

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
    "bat_vs_lhp":   pd.DataFrame(),
    "bat_vs_rhp":   pd.DataFrame(),
    "pit_2026":     pd.DataFrame(),
    "pit_2025":     pd.DataFrame(),
    "pit_vs_lhh":   pd.DataFrame(),
    "pit_vs_rhh":   pd.DataFrame(),
    "pit_arsenal":  pd.DataFrame(),
    "bat_arsenal":  pd.DataFrame(),
    "player_hands": {},
    "player_ip":    {},
    "ready":        False,
    "last_updated": None,
    "last_8d_update": None,
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
        r = await session.get(url, headers=headers, timeout=30, follow_redirects=True)
        if not r.is_success:
            print(f"Savant fetch failed {r.status_code}: {url[:80]}")
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
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
                hr9_str = stat.get("homeRunsPer9", "0") or "0"
                try: hr9 = float(hr9_str)
                except: hr9 = 0
                era_str = stat.get("era", "0") or "0"
                try: era = float(era_str)
                except: era = 0
                if name:
                    ip_map[name.lower()] = {"ip": ip, "hr9": hr9, "era": era, "name": name}
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

async def load_all_savant_data():
    """Fetch all data from Baseball Savant + FanGraphs via pybaseball"""
    print("Loading data from Baseball Savant...")

    # Start pybaseball load in background thread (non-blocking)
    async with httpx.AsyncClient(timeout=30) as client:
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

        # Batter vs LHP/RHP and Pitcher vs LHH/RHH — MLB Stats API statSplits
        # (Savant leaderboard ignores pitchHand/batSide filters — MLB API works correctly)

    # Fetch all handedness splits via MLB Stats API
    splits = await fetch_splits_mlb(current_season())
    for key, rows in splits.items():
        if rows:
            df_split = pd.DataFrame(rows)
            _cache[key] = df_split
            print(f"{key}: {len(df_split)} rows")

        # Pitch arsenal - pitcher (try current year, fall back to 2025 early season)
        for yr_try in [current_season(), current_season()-1]:
            df = await fetch_savant_csv(savant_pitch_arsenal_url("pitcher", year=yr_try, min_pa=1), client)
            if not df.empty:
                _cache["pit_arsenal"] = parse_player_name(df)
                print(f"pit_arsenal ({yr_try}): {len(_cache['pit_arsenal'])} rows")
                break
        else:
            print("WARNING: pit_arsenal empty — pitch pills will not show")

        # Pitch arsenal - batter (try current year, fall back to 2025 early season)
        for yr_try in [current_season(), current_season()-1]:
            df = await fetch_savant_csv(savant_pitch_arsenal_url("batter", year=yr_try, min_pa=1), client)
            if not df.empty:
                _cache["bat_arsenal"] = parse_player_name(df)
                print(f"bat_arsenal ({yr_try}): {len(_cache['bat_arsenal'])} rows")
                break
        else:
            print("WARNING: bat_arsenal empty — pitch pills will not show")

    # Pitcher IP from MLB Stats API
    ip_data = await fetch_pitcher_ip(current_season())
    _cache["player_ip"] = ip_data

    # Last 5 games batting from MLB Stats API
    l5g_data = await fetch_last5_games_batting()
    _cache["bat_l5g"] = l5g_data

    _cache["last_updated"] = datetime.now().isoformat()
    _cache["ready"] = True
    print("All data loaded successfully!")

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
        if now.hour == 9:
            try:
                await load_all_savant_data()
            except Exception as e:
                print(f"Daily refresh error: {e}")

def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)
    loop.close()

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=run_async, args=(load_all_savant_data(),), daemon=True).start()
    asyncio.create_task(daily_refresh_loop())

# ── Player matching ──
def fuzzy_match(name: str, df: pd.DataFrame, col="name"):
    if df is None or df.empty or col not in df.columns:
        return None
    nl = name.lower().strip()
    exact = df[df[col].str.lower().str.strip() == nl]
    if not exact.empty:
        return exact.iloc[0]
    last = nl.split()[-1]
    matches = df[df[col].str.lower().str.contains(last, na=False)]
    if len(matches) == 1:
        return matches.iloc[0]
    if len(matches) > 1:
        # Try first name too
        first = nl.split()[0]
        refined = matches[matches[col].str.lower().str.contains(first, na=False)]
        if not refined.empty:
            return refined.iloc[0]
        return matches.iloc[0]
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
        "slg": slg,
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
    if row is None:
        return {"era": era, "ip": ip, "hr9": hr9, "hard_hit_pct": 0,
                "barrel_pct_allowed": 0, "fb_pct": 0, "k_pct": 0, "hr_fb_pct": 0}
    return {
        "era": era or gs(row, "era"),
        "ip": ip,
        "hr9": hr9,
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
    matches = df[df["name"].str.lower().str.contains(pitcher_name.split()[-1].lower(), na=False)]
    if matches.empty:
        return []
    pitches = []
    for _, row in matches.iterrows():
        pt = str(row.get("pitch_type", "")).upper()
        code = PITCH_TYPE_MAP.get(pt)
        if not code:
            continue
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
    pitches.sort(key=lambda x: x["usage"], reverse=True)
    # Deduplicate by code
    seen = set()
    unique = []
    for p in pitches:
        if p["code"] not in seen:
            seen.add(p["code"])
            unique.append(p)
    return unique[:3]

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
    if pa >= 150: return 1.0, 0.0
    elif pa >= 50: w = (pa - 50) / 100.0; return w, 1.0 - w
    else: w = 0.20 + (pa / 50.0) * 0.30; return w, 1.0 - w

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

def compute_hr_probability(name, bat_hand, opp_p_name, opp_p_hand, park_factor, weather_mult):
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
                            name = outcome.get("description") or outcome.get("name", "")
                            price = outcome.get("price", 0)
                            if name and price: props[name.lower()] = price
            except: continue
        return props
    except: return {}

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
    return {
        "name": p_name, "hand": p_hand,
        "era": round(blend(pc.get("era", 0), pp.get("era", 0), pwc, pwp), 2) or None,
        "hr9": round(blend(pc.get("hr9", 0), pp.get("hr9", 0), pwc, pwp), 2) or None,
        "hard_hit_pct": round(blend(pc.get("hard_hit_pct", 0), pp.get("hard_hit_pct", 0), pwc, pwp), 1) or None,
        "barrel_pct": round(blend(pc.get("barrel_pct_allowed", 0), pp.get("barrel_pct_allowed", 0), pwc, pwp), 1) or None,
        "ip_2026": round(ip_26, 1),
        "blend_note": f"{int(pwc*100)}% 2026 / {int(pwp*100)}% 2025",
        # Split fields — from MLB Stats API statSplits (actually handedness-specific)
        "vs_L_hr9":  round(vs_L.get("hr9", 0), 2) if vs_L.get("pa", 0) >= 1 else None,
        "vs_R_hr9":  round(vs_R.get("hr9", 0), 2) if vs_R.get("pa", 0) >= 1 else None,
        "vs_L_k":    round(vs_L.get("k_pct", 0), 1) if vs_L.get("pa", 0) >= 1 else None,
        "vs_R_k":    round(vs_R.get("k_pct", 0), 1) if vs_R.get("pa", 0) >= 1 else None,
        "vs_L_slg":  round(vs_L.get("slg", 0), 3) if vs_L.get("pa", 0) >= 1 else None,
        "vs_R_slg":  round(vs_R.get("slg", 0), 3) if vs_R.get("pa", 0) >= 1 else None,
        "vs_L_woba": round(vs_L.get("woba", 0), 3) if vs_L.get("pa", 0) >= 1 else None,
        "vs_R_woba": round(vs_R.get("woba", 0), 3) if vs_R.get("pa", 0) >= 1 else None,
        "top_pitches": [{"name": p["name"], "usage": p["usage"]} for p in top_pitches],
    }

# ── API Endpoints ──
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
        "bat_2026": len(_cache["bat_2026"]),
        "bat_2025": len(_cache["bat_2025"]),
        "bat_8d": len(_cache["bat_8d"]),
        "bat_l5g": len(_cache["bat_l5g"]),
        "bat_vs_lhp": len(_cache["bat_vs_lhp"]),
        "bat_vs_rhp": len(_cache["bat_vs_rhp"]),
        "pit_2026": len(_cache["pit_2026"]),
        "pit_2025": len(_cache["pit_2025"]),
        "pit_vs_lhh": len(_cache["pit_vs_lhh"]),
        "pit_vs_rhh": len(_cache["pit_vs_rhh"]),
        "pit_arsenal": len(_cache["pit_arsenal"]),
        "bat_arsenal": len(_cache["bat_arsenal"]),
        "player_ip": len(_cache["player_ip"]),
    }

@app.post("/reload")
async def reload_data():
    threading.Thread(target=run_async, args=(load_all_savant_data(),), daemon=True).start()
    return {"status": "Reloading data from Baseball Savant"}

@app.get("/games")
async def get_games(date: str = None):
    if not _cache["ready"]:
        return {"games": [], "loading": True, "message": "Data loading — try again in 30 seconds."}

    from datetime import date as date_cls
    today = date if date else date_cls.today().isoformat()
    date = None  # clear to avoid shadowing

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{MLB_API}/schedule?sportId=1&date={today}&hydrate=team,probablePitcher")
        data = r.json()

    dk_props = await fetch_dk_hr_props()
    dates = data.get("dates", [])
    if not dates: return {"games": [], "date": today, "loading": False}

    games_out = []
    for game in dates[0].get("games", []):
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
                name, bat_hand, opp_p_name, opp_p_hand, park_factor, batter_wx_mult)

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
                "season": {
                    "barrel": round(blend(bc.get("barrel_pct", 0), bp.get("barrel_pct", 0), bwc, bwp), 1),
                    "ev":     round(blend(bc.get("exit_velo", 0), bp.get("exit_velo", 0), bwc, bwp), 1),
                    "la":     round(blend(bc.get("launch_angle", 0), bp.get("launch_angle", 0), bwc, bwp), 1),
                    "hh":     round(blend(bc.get("hard_hit_pct", 0), bp.get("hard_hit_pct", 0), bwc, bwp), 1),
                    "iso":    round(blend(bc.get("iso", 0), bp.get("iso", 0), bwc, bwp), 3),
                    "slg":    round(blend(bc.get("slg_percent", 0), bp.get("slg_percent", 0), bwc, bwp), 3),
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
                    "pull":   round(b8d.get("pull_pct", 0), 1),
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

        for b in lineup_away:
            await process(b, away_team, home_p.get("fullName", "TBD"), home_p_hand, lineup_away_status == "projected")
        for b in lineup_home:
            await process(b, home_team, away_p.get("fullName", "TBD"), away_p_hand, lineup_home_status == "projected")

        away_lineup_ordered = [b for b in all_batters if b["team"] == away_team]
        home_lineup_ordered = [b for b in all_batters if b["team"] == home_team]
        all_batters.sort(key=lambda x: x["hr_prob"], reverse=True)

        games_out.append({
            "game_id": gid, "away": away_team, "home": home_team, "time": gtime,
            "away_pitcher": pit_display(away_p.get("fullName", "TBD"), away_p_hand),
            "home_pitcher": pit_display(home_p.get("fullName", "TBD"), home_p_hand),
            "top_hr_candidates": all_batters,
            "away_lineup": away_lineup_ordered,
            "home_lineup": home_lineup_ordered,
            "lineup_away_status": lineup_away_status,
            "lineup_home_status": lineup_home_status,
            "weather": {"label": wx_label, "temp": temp, "wind_speed": wind_speed, "wind_dir": wind_dir}
        })

    return {"games": games_out, "date": today, "loading": False}

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
