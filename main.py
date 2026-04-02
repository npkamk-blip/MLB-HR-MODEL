from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import pandas as pd
from pybaseball import batting_stats, pitching_stats
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
    "ready": False
}

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
    "Kansas City Royals":     {"L": 0.97, "R": 0.98},
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

def load_savant_data():
    print("Loading Statcast data...")
    for season, kb, kp in [(SEASON_CURRENT,"batting_current","pitching_current"),(SEASON_PRIOR,"batting_prior","pitching_prior")]:
        try:
            bat = batting_stats(season, qual=10)
            print(f"{season} batting columns: {list(bat.columns[:30])}")
            # Flexible column mapping — find the right columns
            col_map = {}
            for col in bat.columns:
                cl = col.lower().strip()
                if cl == 'name': col_map[col] = 'name'
                elif 'barrel' in cl and '%' in cl: col_map[col] = 'barrel_pct'
                elif cl in ['ev','exit velocity','avg exit velo']: col_map[col] = 'exit_velo'
                elif cl == 'la' or cl == 'launch angle': col_map[col] = 'launch_angle'
                elif 'hard' in cl and '%' in cl: col_map[col] = 'hard_hit_pct'
                elif cl == 'iso': col_map[col] = 'iso'
                elif cl == 'hr' or cl == 'home runs': col_map[col] = 'hr'
                elif cl == 'fb%' or cl == 'fb %': col_map[col] = 'fb_pct'
                elif 'hr/fb' in cl or 'hr fb' in cl: col_map[col] = 'hr_fb_pct'
                elif 'pull' in cl and '%' in cl: col_map[col] = 'pull_pct'
                elif cl == 'pa' or cl == 'plate appearances': col_map[col] = 'pa'
            bat = bat.rename(columns=col_map)
            # Convert pct columns from 0-100 scale if needed
            for pct_col in ['barrel_pct','hard_hit_pct','hr_fb_pct','pull_pct','fb_pct']:
                if pct_col in bat.columns:
                    vals = pd.to_numeric(bat[pct_col], errors='coerce').fillna(0)
                    # If values are 0-1 range, multiply by 100
                    if vals.median() < 1.0:
                        bat[pct_col] = vals * 100
                    else:
                        bat[pct_col] = vals
            _cache[kb] = bat
            print(f"{season} batting loaded: {len(bat)} players")
            if 'barrel_pct' in bat.columns:
                print(f"  Sample barrel_pct: {bat['barrel_pct'].dropna().head(5).tolist()}")
        except Exception as e:
            print(f"{season} batting error: {e}")

        try:
            pit = pitching_stats(season, qual=10)
            col_map_p = {}
            for col in pit.columns:
                cl = col.lower().strip()
                if cl == 'name': col_map_p[col] = 'name'
                elif 'hr/9' in cl or 'hr9' in cl: col_map_p[col] = 'hr_per9'
                elif 'hr/fb' in cl or 'hr fb' in cl: col_map_p[col] = 'hr_fb_pct'
                elif cl == 'gb%' or cl == 'gb %' or 'ground' in cl and '%' in cl: col_map_p[col] = 'gb_pct'
                elif cl == 'fb%' or cl == 'fb %': col_map_p[col] = 'fb_pct'
                elif 'hard' in cl and '%' in cl: col_map_p[col] = 'hard_hit_pct'
                elif cl == 'era': col_map_p[col] = 'era'
                elif cl == 'whip': col_map_p[col] = 'whip'
                elif cl == 'pa' or cl == 'plate appearances': col_map_p[col] = 'pa'
            pit = pit.rename(columns=col_map_p)
            for pct_col in ['hr_fb_pct','gb_pct','fb_pct','hard_hit_pct']:
                if pct_col in pit.columns:
                    vals = pd.to_numeric(pit[pct_col], errors='coerce').fillna(0)
                    if vals.median() < 1.0:
                        pit[pct_col] = vals * 100
                    else:
                        pit[pct_col] = vals
            _cache[kp] = pit
            print(f"{season} pitching loaded: {len(pit)} pitchers")
        except Exception as e:
            print(f"{season} pitching error: {e}")

    _cache["ready"] = True
    print("All data ready.")

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=load_savant_data, daemon=True).start()

def fuzzy_match(name, df):
    if df is None or df.empty or "name" not in df.columns:
        return None
    name_lower = name.lower()
    for _, row in df.iterrows():
        rn = str(row.get("name","")).lower()
        if rn == name_lower or rn in name_lower or name_lower in rn:
            return row
    parts = name_lower.split()
    if len(parts) >= 2:
        last = parts[-1]
        for _, row in df.iterrows():
            if last in str(row.get("name","")).lower():
                return row
    return None

def get_blend_weights(pa_current):
    pa = float(pa_current or 0)
    if pa >= 150: return 1.0, 0.0
    elif pa >= 50:
        w = (pa - 50) / 100.0
        return w, 1.0 - w
    else:
        w = 0.20 + (pa / 50.0) * 0.30
        return w, 1.0 - w

def blend_stat(cv, pv, wc, wp):
    c, p = float(cv or 0), float(pv or 0)
    if c == 0 and p == 0: return 0
    if c == 0: return p
    if p == 0: return c
    return c * wc + p * wp

def get_park_factor(home_team, batter_hand):
    pf = PARK_FACTORS.get(home_team, {"L": 1.0, "R": 1.0})
    return pf.get(batter_hand if batter_hand in ("L","R") else "R", 1.0)

def angle_diff(a, b):
    diff = abs(a - b) % 360
    return diff if diff <= 180 else 360 - diff

def calc_weather_effect(home_team, wind_speed, wind_direction, temperature, batter_hand="R"):
    stadium = STADIUMS.get(home_team)
    if not stadium:
        return 0, "Unknown", f"{temperature}°F · {wind_speed} mph"
    if stadium.get("dome"):
        return 0, "Dome", f"Indoor · {temperature}°F"

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

    # Crosswind logic — account for batter handedness
    if abs(alignment) <= 0.5 and wind_speed >= 10:
        # True crosswind — determine direction
        # If wind_direction is roughly 90 degrees clockwise from hr_bearing = blowing L to R
        cross_diff = (wind_direction - hr_bearing) % 360
        blowing_left_to_right = 45 < cross_diff < 225

        if blowing_left_to_right:
            direction_label = "Favors Lefties"
            if batter_hand == "L":
                total += 3
            else:
                total -= 2
        else:
            direction_label = "Favors Righties"
            if batter_hand == "R":
                total += 3
            else:
                total -= 2
    elif alignment > 0.5 and wind_speed >= 10:
        direction_label = "Blowing Out"
    elif alignment < -0.5 and wind_speed >= 10:
        direction_label = "Blowing In"
    elif wind_speed < 5:
        direction_label = "Calm"
    else:
        direction_label = "Crosswind"

    return total, direction_label, f"{temperature}°F · {wind_label}"

async def fetch_weather(lat, lon, game_time_utc):
    try:
        hour = 18
        if game_time_utc:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
                hour = dt.hour
            except: pass
        url = (f"https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}"
               f"&hourly=temperature_2m,windspeed_10m,winddirection_10m"
               f"&temperature_unit=fahrenheit&windspeed_unit=mph"
               f"&forecast_days=1&timezone=auto")
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            d = r.json()
        hourly = d.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        speeds = hourly.get("windspeed_10m", [])
        directions = hourly.get("winddirection_10m", [])
        idx = min(hour, len(temps) - 1)
        for i, t in enumerate(times):
            if f"T{hour:02d}:" in t:
                idx = i
                break
        return (round(temps[idx]) if idx < len(temps) else 70,
                round(speeds[idx]) if idx < len(speeds) else 0,
                round(directions[idx]) if idx < len(directions) else 0)
    except Exception as e:
        print(f"Weather error: {e}")
        return 70, 0, 0

async def fetch_projected_lineup(team_id, team_name):
    try:
        end = date.today()
        start = end - timedelta(days=10)
        url = f"{MLB_API}/schedule?sportId=1&teamId={team_id}&startDate={start}&endDate={end}&hydrate=boxscore"
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url)
            d = r.json()
        dates = d.get("dates", [])
        recent_games = []
        for date_entry in reversed(dates):
            for game in date_entry.get("games", []):
                if game.get("status", {}).get("abstractGameState") == "Final":
                    recent_games.append(game["gamePk"])
            if len(recent_games) >= 5:
                break
        player_data = defaultdict(lambda: {"name": "", "appearances": 0, "orders": [], "hand": "R", "id": 0})
        for gid in recent_games[:5]:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(f"{MLB_API}/game/{gid}/boxscore")
                    box = r.json()
                teams = box.get("teams", {})
                for side in ["away", "home"]:
                    team_data = teams.get(side, {})
                    t_name = team_data.get("team", {}).get("name", "")
                    if team_name.lower() not in t_name.lower() and t_name.lower() not in team_name.lower():
                        continue
                    players = team_data.get("players", {})
                    for pid, p in players.items():
                        order = p.get("battingOrder")
                        if order and int(order) <= 900:
                            person = p.get("person", {})
                            player_id = person.get("id", 0)
                            name = person.get("fullName", "")
                            hand = person.get("batSide", {}).get("code", "R")
                            order_pos = int(order) // 100
                            player_data[player_id]["name"] = name
                            player_data[player_id]["id"] = player_id
                            player_data[player_id]["appearances"] += 1
                            player_data[player_id]["orders"].append(order_pos)
                            player_data[player_id]["hand"] = hand
            except:
                continue
        projected = []
        for pid, data in player_data.items():
            if data["appearances"] >= 2 and data["name"]:
                avg_order = sum(data["orders"]) / len(data["orders"])
                projected.append({"id": data["id"], "name": data["name"], "hand": data["hand"], "appearances": data["appearances"], "avg_order": avg_order})
        projected.sort(key=lambda x: x["avg_order"])
        return projected[:9], "projected"
    except Exception as e:
        print(f"Projected lineup error for {team_name}: {e}")
        return [], "projected"

async def fetch_dk_hr_props():
    if not ODDS_API_KEY:
        return {}
    try:
        url = (f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
               f"?apiKey={ODDS_API_KEY}&regions=us&markets=batter_home_runs"
               f"&oddsFormat=american&bookmakers=draftkings")
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url)
            if not r.is_success:
                print(f"Odds API error: {r.status_code}")
                return {}
            games = r.json()
        props = {}
        for game in games:
            for bk in game.get("bookmakers", []):
                if bk.get("key") != "draftkings":
                    continue
                for market in bk.get("markets", []):
                    if market.get("key") != "batter_home_runs":
                        continue
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("description") or outcome.get("name", "")
                        price = outcome.get("price", 0)
                        if name and price:
                            props[name.lower()] = price
        print(f"DK HR props: {len(props)} players")
        return props
    except Exception as e:
        print(f"DK props error: {e}")
        return {}

def match_dk_odds(player_name, props):
    if not props: return None
    name_lower = player_name.lower()
    if name_lower in props: return props[name_lower]
    parts = name_lower.split()
    if len(parts) >= 2:
        last = parts[-1]
        for k, v in props.items():
            if last in k.lower(): return v
    return None

def fmt_odds(o):
    if o is None: return None
    return f"+{o}" if o > 0 else str(o)

async def fetch_recent_form(player_id, days=14):
    try:
        end = date.today()
        start = end - timedelta(days=days)
        url = f"{MLB_API}/people/{player_id}/stats?stats=byDateRange&group=hitting&season=2026&startDate={start}&endDate={end}"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            d = r.json()
        splits = d.get("stats",[{}])[0].get("splits",[])
        if not splits: return None
        s = splits[0].get("stat",{})
        slg = float(s.get("sluggingPercentage",0) or 0)
        avg = float(s.get("avg",0) or 0)
        return {"hr":int(s.get("homeRuns",0)),"pa":int(s.get("plateAppearances",0)),"slg":slg,"ops":float(s.get("ops",0) or 0),"iso":slg-avg}
    except:
        return None

def score_batter(c, p, recent, park_factor, bat_hand, opp_hand, weather_bonus):
    pa_cur = float(c.get("pa",0) or 0)
    wc, wp = get_blend_weights(pa_cur)
    barrel = blend_stat(c.get("barrel_pct"), p.get("barrel_pct"), wc, wp)
    iso    = blend_stat(c.get("iso"),        p.get("iso"),        wc, wp)
    hr_fb  = blend_stat(c.get("hr_fb_pct"),  p.get("hr_fb_pct"), wc, wp)
    ev     = blend_stat(c.get("exit_velo"),  p.get("exit_velo"), wc, wp)
    pull   = blend_stat(c.get("pull_pct"),   p.get("pull_pct"),  wc, wp)
    ss, reasons = 0, []
    if barrel > 0:
        ss += min(barrel/15.0,1.0)*30
        if barrel > 8: reasons.append(f"Barrel {barrel:.1f}%")
    if iso > 0:
        ss += min(iso/0.280,1.0)*20
        if iso > 0.160: reasons.append(f"ISO .{int(iso*1000)}")
    if hr_fb > 0:
        ss += min(hr_fb/25.0,1.0)*15
        if hr_fb > 14: reasons.append(f"HR/FB {hr_fb:.1f}%")
    if ev > 0:
        ss += min((ev-85)/15.0,1.0)*15
        if ev > 91: reasons.append(f"EV {ev:.1f} mph")
    if pull > 0:
        ss += min(pull/50.0,1.0)*5
    if opp_hand and bat_hand:
        if bat_hand != opp_hand:
            ss += 5
            reasons.append(f"Platoon edge ({bat_hand} vs {opp_hand}P)")
        else:
            ss -= 3
    fs = 0
    if recent and recent.get("pa",0) >= 5:
        rf_pa, rf_hr = recent.get("pa",1), recent.get("hr",0)
        rf_slg, rf_iso = recent.get("slg",0), recent.get("iso",0)
        hr_rate = (rf_hr/rf_pa)*100 if rf_pa > 0 else 0
        fs += min(hr_rate/8.0,1.0)*50
        fs += min(rf_slg/0.600,1.0)*30
        fs += min(rf_iso/0.300,1.0)*20
    else:
        fs = ss * 0.5
    final = (ss*0.70 + fs*0.30) * park_factor + weather_bonus
    return round(min(final,99)), reasons, {
        "barrel": round(barrel,1),
        "ev": round(ev,1),
        "iso": round(iso,3),
        "hr_fb": round(hr_fb,1),
    }

def score_pitcher(c, p):
    pa = float(c.get("pa",0) or 0)
    wc, wp = get_blend_weights(pa)
    hr9  = blend_stat(c.get("hr_per9"),   p.get("hr_per9"),   wc, wp)
    hrfb = blend_stat(c.get("hr_fb_pct"), p.get("hr_fb_pct"), wc, wp)
    gb   = blend_stat(c.get("gb_pct"),    p.get("gb_pct"),    wc, wp)
    hard = blend_stat(c.get("hard_hit_pct"), p.get("hard_hit_pct"), wc, wp)
    v, reasons = 0, []
    if hr9 > 0:
        v += min(hr9/2.5,1.0)*10
        if hr9 > 1.3: reasons.append(f"SP {hr9:.1f} HR/9")
    if hrfb > 0: v += min(hrfb/20.0,1.0)*5
    if gb < 40:
        v += 5
        reasons.append("Fly ball SP")
    if hard > 38:
        v += 3
        reasons.append(f"SP {hard:.0f}% hard contact")
    return v, reasons

@app.get("/")
def root():
    return {"status":"MLB HR Model running","data_ready":_cache["ready"],"season":SEASON_CURRENT}

@app.get("/status")
def status():
    return {"ready":_cache["ready"],"batters_2026":len(_cache["batting_current"]),"batters_2025":len(_cache["batting_prior"]),"pitchers_2026":len(_cache["pitching_current"]),"pitchers_2025":len(_cache["pitching_prior"])}

@app.get("/debug-columns")
def debug_columns():
    bat_cur = _cache["batting_current"]
    bat_pri = _cache["batting_prior"]
    sample = {}
    for col in ["barrel_pct","exit_velo","iso","hr_fb_pct","pull_pct","pa"]:
        if col in bat_pri.columns:
            sample[f"2025_{col}"] = round(float(bat_pri[col].dropna().median()),2)
        if col in bat_cur.columns:
            sample[f"2026_{col}"] = round(float(bat_cur[col].dropna().median()),2)
    return {"batting_2025_cols": list(bat_pri.columns[:40]) if not bat_pri.empty else [], "batting_2026_cols": list(bat_cur.columns[:40]) if not bat_cur.empty else [], "sample_medians": sample}

@app.get("/games")
async def get_games(form_days: int = 14):
    if not _cache["ready"]:
        return {"games":[],"date":date.today().isoformat(),"loading":True,"message":"Statcast data loading — try again in 60 seconds."}
    today = date.today().isoformat()
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{MLB_API}/schedule?sportId=1&date={today}&hydrate=team,probablePitcher")
        data = r.json()
    dk_props = await fetch_dk_hr_props()
    dates = data.get("dates",[])
    if not dates:
        return {"games":[],"date":today,"loading":False}
    games_out = []
    for game in dates[0].get("games",[]):
        if game.get("status",{}).get("abstractGameState") == "Final":
            continue
        gid       = game["gamePk"]
        away_team = game["teams"]["away"]["team"]["name"]
        home_team = game["teams"]["home"]["team"]["name"]
        away_team_id = game["teams"]["away"]["team"]["id"]
        home_team_id = game["teams"]["home"]["team"]["id"]
        away_p    = game["teams"]["away"].get("probablePitcher",{})
        home_p    = game["teams"]["home"].get("probablePitcher",{})
        gtime     = game.get("gameDate","")
        stadium   = STADIUMS.get(home_team, {})
        temp, wind_speed, wind_dir = 70, 0, 0
        if not stadium.get("dome") and stadium.get("lat"):
            temp, wind_speed, wind_dir = await fetch_weather(stadium["lat"], stadium["lon"], gtime)

        def get_pit(name):
            c = fuzzy_match(name, _cache["pitching_current"])
            p = fuzzy_match(name, _cache["pitching_prior"])
            return c.to_dict() if c is not None else {}, p.to_dict() if p is not None else {}

        hpc, hpp = get_pit(home_p.get("fullName",""))
        apc, app_ = get_pit(away_p.get("fullName",""))
        hp_vuln, hp_reasons = score_pitcher(hpc, hpp)
        ap_vuln, ap_reasons = score_pitcher(apc, app_)

        lineup_away, lineup_home = [], []
        lineup_away_status = "projected"
        lineup_home_status = "projected"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(f"{MLB_API}/game/{gid}/boxscore")
                box = r.json()
            teams = box.get("teams",{})
            def extract(side):
                players = teams.get(side,{}).get("players",{})
                return sorted([p for p in players.values() if p.get("battingOrder") and int(p["battingOrder"])<=900],key=lambda x:int(x["battingOrder"]))[:9]
            ca = extract("away")
            ch = extract("home")
            if ca: lineup_away = ca; lineup_away_status = "confirmed"
            if ch: lineup_home = ch; lineup_home_status = "confirmed"
        except:
            pass

        if not lineup_away:
            lineup_away, _ = await fetch_projected_lineup(away_team_id, away_team)
        if not lineup_home:
            lineup_home, _ = await fetch_projected_lineup(home_team_id, home_team)

        all_batters = []

        async def process(batter, team, pit_vuln, pit_reasons, pit_name, pit_hand, is_projected=False):
            if "person" in batter:
                name     = batter.get("person",{}).get("fullName","")
                pid      = batter.get("person",{}).get("id")
                bat_hand = batter.get("person",{}).get("batSide",{}).get("code","R")
            else:
                name     = batter.get("name","")
                pid      = batter.get("id")
                bat_hand = batter.get("hand","R")

            bc = fuzzy_match(name, _cache["batting_current"])
            bp = fuzzy_match(name, _cache["batting_prior"])
            recent = await fetch_recent_form(pid, days=form_days) if pid else None
            pf = get_park_factor(home_team, bat_hand)
            weather_bonus, weather_label, weather_desc = calc_weather_effect(home_team, wind_speed, wind_dir, temp, bat_hand)
            score, reasons, statline = score_batter(
                bc.to_dict() if bc is not None else {},
                bp.to_dict() if bp is not None else {},
                recent, pf, bat_hand, pit_hand, weather_bonus
            )
            score = min(99, score + pit_vuln)
            reasons += pit_reasons
            if pf >= 1.10: reasons.append("HR-friendly park")
            elif pf <= 0.90: reasons.append("Pitcher-friendly park")
            dk_odds = match_dk_odds(name, dk_props)
            all_batters.append({
                "name": name, "team": team, "score": score, "reasons": reasons,
                "opp_pitcher": pit_name, "bat_hand": bat_hand, "park_factor": round(pf,2),
                "statline": statline,
                "recent_hr": recent.get("hr",0) if recent else 0,
                "recent_pa": recent.get("pa",0) if recent else 0,
                "recent_slg": round(recent.get("slg",0),3) if recent else 0,
                "dk_odds": fmt_odds(dk_odds),
                "projected": is_projected,
                "weather_label": weather_label,
            })

        away_proj = lineup_away_status == "projected"
        home_proj = lineup_home_status == "projected"
        for b in lineup_away:
            await process(b, away_team, hp_vuln, hp_reasons, home_p.get("fullName","TBD"), "R", away_proj)
        for b in lineup_home:
            await process(b, home_team, ap_vuln, ap_reasons, away_p.get("fullName","TBD"), "R", home_proj)

        all_batters.sort(key=lambda x: x["score"], reverse=True)
        weather_bonus_game, weather_label_game, weather_desc_game = calc_weather_effect(home_team, wind_speed, wind_dir, temp)
        games_out.append({
            "game_id": gid, "away": away_team, "home": home_team, "time": gtime,
            "away_pitcher": away_p.get("fullName","TBD"),
            "home_pitcher": home_p.get("fullName","TBD"),
            "top_hr_candidates": all_batters[:3],
            "lineups_posted": lineup_away_status=="confirmed" or lineup_home_status=="confirmed",
            "lineup_away_status": lineup_away_status,
            "lineup_home_status": lineup_home_status,
            "weather": {"label": weather_label_game, "desc": weather_desc_game, "temp": temp, "wind_speed": wind_speed, "wind_dir": wind_dir}
        })
    return {"games":games_out,"date":today,"loading":False}

@app.post("/refresh-cache")
def refresh_cache():
    _cache["ready"] = False
    threading.Thread(target=load_savant_data, daemon=True).start()
    return {"status":"Cache refresh started"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
