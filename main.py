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
    "bat_l8d_hr":   {},
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
                    "k_per_9":     float(s.get("strikeoutsPer9Inn", "8.0").replace("-.--", "8.0") or 8.0),
                    "runs_per_g":  round(float(s.get("runs", 0) or 0) / max(g, 1), 2),
                    "games":       g,
                }
        _cache["team_hitting"]  = team_hitting
        _cache["team_pitching"] = team_pitching
        print(f"team_hitting: {len(team_hitting)} teams, team_pitching: {len(team_pitching)} teams")
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
        if now.hour == 9:
            try:
                await load_all_savant_data()
            except Exception as e:
                print(f"Daily refresh error: {e}")
        # Save predictions at 1pm ET (18:00 UTC) — lineups mostly confirmed
        if now.hour == 13:
            try:
                await save_daily_predictions()
            except Exception as e:
                print(f"Prediction save error: {e}")
        # Record results at 2am ET (07:00 UTC) — all games finished
        if now.hour == 2:
            try:
                yesterday = (date.today() - timedelta(days=1)).isoformat()
                await record_results(yesterday)
            except Exception as e:
                print(f"Result recording error: {e}")

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
    """Save today's predictions to GitHub as data/predictions/YYYY-MM-DD.json"""
    if not _cache["ready"]: return
    today = date.today().isoformat()
    path = f"data/predictions/{today}.json"
    # Check if already saved today
    existing, _ = await github_get_file(path)
    if existing:
        print(f"Predictions already saved for {today}")
        return
    try:
        # Fetch today's games and collect all batter predictions above 5%
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{MLB_API}/schedule?sportId=1&date={today}&hydrate=team,probablePitcher")
            data = r.json()
        records = []
        for game_date in data.get("dates", []):
            for game in game_date.get("games", []):
                if game.get("status", {}).get("abstractGameState") == "Final": continue
                home_team = game["teams"]["home"]["team"]["name"]
                away_team = game["teams"]["away"]["team"]["name"]
                away_p = game["teams"]["away"].get("probablePitcher", {})
                home_p = game["teams"]["home"].get("probablePitcher", {})
                # Get lineups from cache — use top_hr_candidates from all_batters if available
                # We'll store predictions for any batter we've computed
                for side, opp_p, team in [("away", home_p, away_team), ("home", away_p, home_team)]:
                    opp_p_name = opp_p.get("fullName", "TBD")
                    opp_p_id = opp_p.get("id")
                    opp_p_hand = "R"
                    if opp_p_id:
                        info = await fetch_player_hand(opp_p_id)
                        opp_p_hand = info.get("pitch_hand", "R")
                    # Get projected lineup
                    team_id = game["teams"][side]["team"]["id"]
                    lineup, _ = await fetch_projected_lineup(team_id, team)
                    for batter in lineup[:9]:
                        name = batter.get("name", "")
                        pid = batter.get("id")
                        if not name: continue
                        bat_hand = "R"
                        if pid:
                            info = await fetch_player_hand(pid)
                            bat_hand = info.get("bat_side", "R")
                        if bat_hand == "S": bat_hand = "L" if opp_p_hand == "R" else "R"
                        park_factor = get_park_hr_factor(home_team, bat_hand)
                        stadium = STADIUMS.get(home_team, {})
                        temp, wind_speed, wind_dir = 70, 0, 0
                        if not stadium.get("dome") and stadium.get("lat"):
                            temp, wind_speed, wind_dir = await fetch_weather(stadium["lat"], stadium["lon"], game.get("gameDate",""))
                        wx_mult, _ = calc_weather_multiplier(home_team, wind_speed, wind_dir, temp, bat_hand)
                        hr_prob, breakdown, _, _, _, _, _ = compute_hr_probability(name, bat_hand, opp_p_name, opp_p_hand, park_factor, wx_mult)
                        if hr_prob < 5: continue
                        records.append({
                            "date": today,
                            "name": name,
                            "team": team,
                            "opp_pitcher": opp_p_name,
                            "opp_pitcher_hand": opp_p_hand,
                            "bat_hand": bat_hand,
                            "home_team": home_team,
                            "model_hr_pct": hr_prob,
                            "hit_hr": None,  # filled in later by record_results
                            "barrel_pct": breakdown.get("barrel_use", 0),
                            "launch_angle": breakdown.get("la_use", 0),
                            "iso_vs_hand": breakdown.get("iso_vs_hand", 0),
                            "l8d_hr": get_l8d_hr(name),
                            "park_factor": breakdown.get("park_factor", 1.0),
                            "weather_mult": breakdown.get("weather_mult", 1.0),
                            "barrel_mult": breakdown.get("barrel_mult", 1.0),
                            "la_mult": breakdown.get("la_mult", 1.0),
                            "pit_vuln_mult": breakdown.get("pit_vuln_mult", 1.0),
                            "bat_platoon_mult": breakdown.get("bat_platoon_mult", 1.0),
                            "pit_platoon_mult": breakdown.get("pit_platoon_mult", 1.0),
                            "hot_cold_mult": breakdown.get("hot_cold_mult", 1.0),
                            "k_mult": breakdown.get("k_mult", 1.0),
                        })
        if not records:
            print(f"No predictions to save for {today}")
            return
        import json
        content = json.dumps(records, indent=2)
        await github_put_file(path, content, f"predictions: {today} ({len(records)} batters)")
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
                            if int(stats.get("homeRuns", 0) or 0) > 0:
                                name = p.get("person", {}).get("fullName", "")
                                if name: hr_hitters.add(name.lower())
                except Exception: continue
        # Update records with actual results
        updated = 0
        for rec in records:
            if rec.get("hit_hr") is None:
                rec["hit_hr"] = 1 if rec["name"].lower() in hr_hitters else 0
                updated += 1
        content_updated = json.dumps(records, indent=2)
        await github_put_file(path, content_updated, f"results: {target_date} ({len(hr_hitters)} HRs recorded)", sha)
        print(f"Recorded results for {target_date}: {len(hr_hitters)} HR hitters, {updated} records updated")
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

def compute_hr_prob_multiplicative(name, bat_hand, opp_p_name, opp_p_hand, park_factor, weather_mult):
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

    # PA-weighted blend
    if pa_26 >= 300:
        base_rate = hr_per_pa_season * 0.85 + hr_per_pa_career * 0.15
    elif pa_26 >= 150:
        base_rate = hr_per_pa_season * 0.60 + hr_per_pa_career * 0.40
    else:
        base_rate = hr_per_pa_season * 0.30 + hr_per_pa_career * 0.70

    # Floor: use league avg ~2.8% if no data
    if base_rate <= 0:
        base_rate = 0.028
    base_rate = min(base_rate, 0.12)  # no single batter truly > 12% per PA

    # Small sample confidence gate (not a hard cut — just dampens)
    if total_pa < 30:   base_rate = base_rate * 0.55 + 0.028 * 0.45
    elif total_pa < 60: base_rate = base_rate * 0.75 + 0.028 * 0.25

    running = base_rate

    # ── Step 2: Barrel% multiplier ──
    LG_BARREL = 8.0
    barrel_season = blend(bc.get("barrel_pct", 0), bp.get("barrel_pct", 0), bwc, bwp)
    barrel_8d     = b8d.get("barrel_pct", 0) if has_8d else 0
    # 50% relative divergence rule
    if has_8d and barrel_8d > 0 and barrel_season > 0:
        rel_div = abs(barrel_8d - barrel_season) / barrel_season
        barrel_use = barrel_8d if rel_div >= 0.50 else barrel_season
    else:
        barrel_use = barrel_season if barrel_season > 0 else LG_BARREL
    if barrel_use < 0.1: barrel_use = LG_BARREL  # no data → neutral
    barrel_mult = barrel_use / LG_BARREL
    # Min 50 BBE threshold (proxy: pa_26 >= 50 season, else fallback to 1.0)
    if pa_26 < 50 and barrel_8d == 0:
        barrel_mult = 1.0
    barrel_mult = max(min(barrel_mult, 3.5), 0.3)
    running *= barrel_mult

    # ── Step 3: Launch angle multiplier ──
    la_season = blend(bc.get("launch_angle", 0), bp.get("launch_angle", 0), bwc, bwp)
    la_8d     = b8d.get("launch_angle", 0) if has_8d else 0
    # Direction toward 25-35° sweet zone matters — higher is NOT always better
    if has_8d and la_8d > 0 and la_season > 0:
        rel_div_la = abs(la_8d - la_season) / max(abs(la_season), 1)
        if rel_div_la >= 0.50:
            # Check if moving toward or away from sweet zone
            sweet_mid = 30.0
            dist_season = abs(la_season - sweet_mid)
            dist_8d     = abs(la_8d - sweet_mid)
            la_use = la_8d if dist_8d < dist_season else la_season
        else:
            la_use = la_season
    else:
        la_use = la_season if la_season > 0 else 20.0
    if la_use <= 0: la_use = 20.0
    if 25 <= la_use <= 35:   la_mult = 1.00
    elif 20 <= la_use < 25:  la_mult = 0.90
    elif 35 < la_use <= 40:  la_mult = 0.90
    elif 18 <= la_use < 20:  la_mult = 0.80
    elif 40 < la_use <= 45:  la_mult = 0.80
    else:                     la_mult = 0.75
    running *= la_mult

    # ── Step 4: Pitcher vulnerability multiplier ──
    pc = get_pitcher_stats(opp_p_name, 2026)
    pp2 = get_pitcher_stats(opp_p_name, 2025)
    ip_26 = pc.get("ip", 0)
    pwc, pwp = get_pitcher_blend_weights(ip_26, pp2.get("ip", 0))

    LG_HR9  = 1.10
    LG_HH   = 38.0
    pit_hr9_season = blend(pc.get("hr9", 0), pp2.get("hr9", 0), pwc, pwp)
    pit_hard = blend(pc.get("hard_hit_pct", 0), pp2.get("hard_hit_pct", 0), pwc, pwp)

    m_hr9  = (pit_hr9_season / LG_HR9) if pit_hr9_season > 0 else 1.0
    m_hard = (pit_hard / LG_HH) if pit_hard > 0 else 1.0
    # Average — not multiply — to avoid double counting correlated stats
    active_pit = [x for x in [m_hr9 if pit_hr9_season > 0 else None,
                               m_hard if pit_hard > 0 else None] if x is not None]
    pit_vuln_mult = sum(active_pit) / len(active_pit) if active_pit else 1.0
    # Min 40 IP threshold
    if ip_26 < 40 and pp2.get("ip", 0) < 40:
        pit_vuln_mult = 0.5 + pit_vuln_mult * 0.5  # dampen to neutral
    pit_vuln_mult = max(min(pit_vuln_mult, 1.80), 0.50)
    running *= pit_vuln_mult

    # ── Step 5: Batter platoon multiplier (ISO vs pitcher hand / overall ISO) ──
    iso_vs_hand = b_split_vs_hand.get("iso", 0)
    iso_overall = blend(bc.get("iso", 0), bp.get("iso", 0), bwc, bwp)
    split_pa_vs_hand = b_split_vs_hand.get("pa", 0)
    if split_pa_vs_hand >= 80 and iso_vs_hand > 0 and iso_overall > 0:
        bat_platoon_mult = iso_vs_hand / iso_overall
        bat_platoon_mult = max(min(bat_platoon_mult, 1.60), 0.60)
    else:
        bat_platoon_mult = 1.0
    running *= bat_platoon_mult

    # ── Step 6: Pitcher platoon multiplier (SLG allowed vs batter hand / overall SLG) ──
    slg_vs_bat   = p_split_vs_bat.get("slg", 0)
    # Pitcher overall SLG allowed — average of vs L and vs R as proxy
    p_split_opp_hand = get_pitcher_split(opp_p_name, "L" if bat_hand == "R" else "R")
    slg_overall_pit = 0
    slg_sources = [x for x in [slg_vs_bat, p_split_opp_hand.get("slg", 0)] if x > 0]
    slg_overall_pit = sum(slg_sources) / len(slg_sources) if slg_sources else 0
    split_ip_vs_bat = p_split_vs_bat.get("ip", 0)
    if split_ip_vs_bat >= 20 and slg_vs_bat > 0 and slg_overall_pit > 0:
        pit_platoon_mult = slg_vs_bat / slg_overall_pit
        pit_platoon_mult = max(min(pit_platoon_mult, 1.60), 0.60)
    else:
        pit_platoon_mult = 1.0
    running *= pit_platoon_mult

    # ── Step 7: Park multiplier ──
    running *= park_factor

    # ── Step 8: Weather multiplier ──
    running *= weather_mult

    # ── Step 9: Hot/cold multiplier (L8D rate vs expected rate) ──
    hot_cold_mult = 1.0
    if has_8d and b8d.get("pa", 0) >= 8:
        pa_8d = b8d.get("pa", 0)
        # Use MLB Stats API lastXGames=8 for reliable HR count (Savant 8d returns season totals for some players)
        hr_8d_count = get_l8d_hr(name)
        hr_8d_rate  = hr_8d_count / pa_8d
        expected_8d_rate = base_rate
        if expected_8d_rate > 0:
            ratio = hr_8d_rate / expected_8d_rate
            hot_cold_mult = max(min(ratio, 1.20), 0.85)
    running *= hot_cold_mult

    # ── Step 10: K% penalty (smooth, applied last, only trims) ──
    k_season = blend(bc.get("k_pct", 0), bp.get("k_pct", 0), bwc, bwp)
    if k_season >= 35:   k_mult = 0.88
    elif k_season >= 30: k_mult = 0.94
    elif k_season >= 25: k_mult = 0.97
    else:                k_mult = 1.0
    running *= k_mult

    # ── Hard cap at 28% ──
    hr_prob = round(min(running * 100, 28.0), 1)

    # ── Build breakdown for frontend ──
    pitch_bonus, pitch_details = compute_pitch_matchup(opp_p_name, name)
    archetype = get_archetype(barrel_season, k_season,
                              blend(bc.get("fb_pct", 0), bp.get("fb_pct", 0), bwc, bwp),
                              iso_overall)
    trend = get_trend(b8d, bc)

    reasons = []
    if barrel_use >= 12: reasons.append(f"Barrel {barrel_use:.1f}%")
    if iso_vs_hand > 0.220: reasons.append(f"ISO vs hand .{int(iso_vs_hand*1000):03d}")
    if pit_hr9_season > 1.3: reasons.append(f"SP {pit_hr9_season:.1f} HR/9")
    if pit_hard > 40: reasons.append(f"SP {pit_hard:.1f}% HH")
    if park_factor >= 1.15: reasons.append("HR-friendly park")
    elif park_factor <= 0.90: reasons.append("Pitcher-friendly park")
    if hot_cold_mult >= 1.10: reasons.append("Hot streak")
    elif hot_cold_mult <= 0.90: reasons.append("Cold streak")

    # Platoon tag for display
    platoon_tag = None
    if bat_platoon_mult >= 1.20:
        hand_label = "LHB" if bat_hand == "L" else "RHB"
        platoon_tag = f"Batter strong vs {opp_p_hand}HP"
    if pit_platoon_mult >= 1.20:
        platoon_tag = (platoon_tag + " + " if platoon_tag else "") + f"SP weak vs {bat_hand}HB"

    n_components = len(active_pit)
    conf = "High" if n_components >= 2 and pa_26 >= 50 else "Medium" if n_components >= 1 else "Low"
    blend_note = f"{int(bwc*100)}% 2026/{int(bwp*100)}% 2025" + (" + 8d" if has_8d else "")

    breakdown = {
        # Base
        "base_rate": round(base_rate * 100, 2),
        # Barrel
        "barrel_use": round(barrel_use, 1), "barrel_season": round(barrel_season, 1),
        "barrel_8d": round(b8d.get("barrel_pct", 0), 1), "barrel_mult": round(barrel_mult, 3),
        # LA
        "la_use": round(la_use, 1), "la_season": round(la_season, 1),
        "la_8d": round(la_8d, 1), "la_mult": round(la_mult, 3),
        # Pitcher
        "pit_hr9": round(pit_hr9_season, 2), "pit_hard": round(pit_hard, 1),
        "pit_vuln_mult": round(pit_vuln_mult, 3),
        # Platoon
        "bat_platoon_mult": round(bat_platoon_mult, 3),
        "pit_platoon_mult": round(pit_platoon_mult, 3),
        "iso_vs_hand": round(iso_vs_hand, 3), "iso_overall": round(iso_overall, 3),
        "slg_vs_bat": round(slg_vs_bat, 3) if split_ip_vs_bat >= 5 else 0,
        "split_pa_vs_hand": split_pa_vs_hand,
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
        "split_pa": split_pa_vs_hand,
        "hr_season": int(bc.get("hr", 0)),
        "pa_8d": int(b8d.get("pa", 0)),
        "barrel_8d_raw": round(b8d.get("barrel_pct", 0), 1),
        "iso_8d": round(b8d.get("iso", 0), 3),
        "pull_8d": round(b8d.get("pull_pct", 0), 1),
        "la_8d_raw": round(la_8d, 1),
        "pitch_bonus": pitch_bonus, "pitch_breakdown": pitch_details,
        "after_k": round(running * 100, 1), "after_context": round(running * 100, 1),
        "n_pit_components": n_components,
        # Pitcher SLG vs batter hand for table display
        "pit_slg_vs_bat": round(slg_vs_bat, 3) if split_ip_vs_bat >= 5 else 0,
        "pit_slg_overall": round(slg_overall_pit, 3),
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

def compute_hr_probability(name, bat_hand, opp_p_name, opp_p_hand, park_factor, weather_mult):
    return compute_hr_prob_multiplicative(name, bat_hand, opp_p_name, opp_p_hand, park_factor, weather_mult)

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
    k_props  = await fetch_pitcher_k_props()
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

        for b in lineup_away:
            await process(b, away_team, home_p.get("fullName", "TBD"), home_p_hand, lineup_away_status == "projected")
        for b in lineup_home:
            await process(b, home_team, away_p.get("fullName", "TBD"), away_p_hand, lineup_home_status == "projected")

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
