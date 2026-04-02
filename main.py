from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import pandas as pd
from pybaseball import batting_stats, pitching_stats
from datetime import date, timedelta
import threading
import uvicorn
import math

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

_cache = {
    "batting_current": pd.DataFrame(),
    "batting_prior": pd.DataFrame(),
    "pitching_current": pd.DataFrame(),
    "pitching_prior": pd.DataFrame(),
    "ready": False
}

# Park factors by handedness
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

# Stadium data: lat, lon, dome, hr_wind_bearing (compass degrees wind FROM that blows OUT), open_factor (1.0=very open like Wrigley, 0.5=typical)
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
    cols_bat = {"Name":"name","Barrel%":"barrel_pct","EV":"exit_velo","LA":"launch_angle","Hard%":"hard_hit_pct","ISO":"iso","HR":"hr","FB%":"fb_pct","HR/FB":"hr_fb_pct","Pull%":"pull_pct","PA":"pa"}
    cols_pit = {"Name":"name","HR/9":"hr_per9","HR/FB":"hr_fb_pct","GB%":"gb_pct","FB%":"fb_pct","Hard%":"hard_hit_pct","ERA":"era","WHIP":"whip"}
    for season, kb, kp in [(SEASON_CURRENT,"batting_current","pitching_current"),(SEASON_PRIOR,"batting_prior","pitching_prior")]:
        try:
            bat = batting_stats(season, qual=10)
            bat = bat.rename(columns=cols_bat)
            _cache[kb] = bat
            print(f"{season} batting: {len(bat)} players")
        except Exception as e:
            print(f"{season} batting error: {e}")
        try:
            pit = pitching_stats(season, qual=10)
            pit = pit.rename(columns=cols_pit)
            _cache[kp] = pit
            print(f"{season} pitching: {len(pit)} pitchers")
        except Exception as e:
            print(f"{season} pitching error: {e}")
    _cache["ready"] = True
    print("Data ready.")

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=load_savant_data, daemon=True).start()

def fuzzy_match(name, df):
    if df is None or df.empty or "name" not in df.columns:
        return None
    name_lower = name.lower()
    for _, row in df.iterrows():
        rn = str(row["name"]).lower()
        if rn == name_lower or rn in name_lower or name_lower in rn:
            return row
    parts = name_lower.split()
    if len(parts) >= 2:
        last = parts[-1]
        for _, row in df.iterrows():
            if last in str(row["name"]).lower():
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

def calc_weather_effect(home_team, wind_speed, wind_direction, temperature):
    """
    Returns (score_bonus, label, description)
    wind_direction = degrees the wind is coming FROM
    hr_bearing = degrees wind FROM that blows OUT to outfield
    """
    stadium = STADIUMS.get(home_team)
    if not stadium:
        return 0, "Unknown", f"{temperature}°F · {wind_speed} mph"

    if stadium.get("dome"):
        return 0, "Dome", f"Indoor · {temperature}°F"

    hr_bearing = stadium.get("hr_bearing", 225)
    open_factor = stadium.get("open_factor", 0.5)

    # How aligned is the wind with the HR direction
    diff = angle_diff(wind_direction, hr_bearing)

    # diff=0 means wind blowing directly out, diff=180 means directly in
    # Map to -1 (blowing in) to +1 (blowing out)
    alignment = math.cos(math.radians(diff))  # 1=out, -1=in, 0=crosswind

    # Wind speed effect (3-4 feet per mph on long fly balls)
    if wind_speed < 5:
        speed_factor = 0
        wind_label = "Calm"
    elif wind_speed < 10:
        speed_factor = 0.3
        wind_label = f"{wind_speed} mph"
    elif wind_speed < 16:
        speed_factor = 0.7
        wind_label = f"{wind_speed} mph"
    else:
        speed_factor = 1.0
        wind_label = f"{wind_speed} mph"

    # Raw wind bonus/penalty (-10 to +10 before open_factor)
    wind_effect = alignment * speed_factor * 10 * open_factor

    # Temperature effect
    temp_effect = 0
    if temperature >= 80:
        temp_effect = 3
    elif temperature >= 70:
        temp_effect = 1
    elif temperature < 50:
        temp_effect = -3
    elif temperature < 60:
        temp_effect = -1

    total = round(wind_effect + temp_effect)

    # Label
    if alignment > 0.5 and wind_speed >= 10:
        direction_label = "Blowing Out"
    elif alignment < -0.5 and wind_speed >= 10:
        direction_label = "Blowing In"
    elif wind_speed < 5:
        direction_label = "Calm"
    else:
        direction_label = "Crosswind"

    desc = f"{temperature}°F · {wind_label}"
    return total, direction_label, desc

async def fetch_weather(lat, lon, game_time_utc):
    try:
        hour = 18  # default 6pm local if we can't parse
        if game_time_utc:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
                hour = dt.hour
            except:
                pass

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

        # Find the closest hour
        idx = min(hour, len(temps) - 1)
        for i, t in enumerate(times):
            if f"T{hour:02d}:" in t:
                idx = i
                break

        temp = round(temps[idx]) if idx < len(temps) else 70
        speed = round(speeds[idx]) if idx < len(speeds) else 0
        direction = round(directions[idx]) if idx < len(directions) else 0

        return temp, speed, direction
    except Exception as e:
        print(f"Weather fetch error: {e}")
        return 70, 0, 0

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
        return {
            "hr": int(s.get("homeRuns",0)),
            "pa": int(s.get("plateAppearances",0)),
            "slg": slg,
            "ops": float(s.get("ops",0) or 0),
            "iso": slg - avg
        }
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

    # Recent form — 30% weight
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

    final = (ss*0.70 + fs*0.30) * park_factor
    final += weather_bonus
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
    return {
        "ready":_cache["ready"],
        "batters_2026":len(_cache["batting_current"]),
        "batters_2025":len(_cache["batting_prior"]),
        "pitchers_2026":len(_cache["pitching_current"]),
        "pitchers_2025":len(_cache["pitching_prior"])
    }

@app.get("/games")
async def get_games(form_days: int = 14):
    if not _cache["ready"]:
        return {"games":[],"date":date.today().isoformat(),"loading":True,"message":"Statcast data loading — try again in 60 seconds."}

    today = date.today().isoformat()
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{MLB_API}/schedule?sportId=1&date={today}&hydrate=team,probablePitcher")
        data = r.json()

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
        away_p    = game["teams"]["away"].get("probablePitcher",{})
        home_p    = game["teams"]["home"].get("probablePitcher",{})
        gtime     = game.get("gameDate","")

        # Weather
        stadium = STADIUMS.get(home_team, {})
        temp, wind_speed, wind_dir = 70, 0, 0
        if not stadium.get("dome") and stadium.get("lat"):
            temp, wind_speed, wind_dir = await fetch_weather(stadium["lat"], stadium["lon"], gtime)
        weather_bonus, weather_label, weather_desc = calc_weather_effect(home_team, wind_speed, wind_dir, temp)

        # Pitcher data
        def get_pit(name):
            c = fuzzy_match(name, _cache["pitching_current"])
            p = fuzzy_match(name, _cache["pitching_prior"])
            return c.to_dict() if c is not None else {}, p.to_dict() if p is not None else {}

        hpc, hpp = get_pit(home_p.get("fullName",""))
        apc, app_ = get_pit(away_p.get("fullName",""))
        hp_vuln, hp_reasons = score_pitcher(hpc, hpp)
        ap_vuln, ap_reasons = score_pitcher(apc, app_)

        # Lineups
        lineup_away, lineup_home = [], []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(f"{MLB_API}/game/{gid}/boxscore")
                box = r.json()
            teams = box.get("teams",{})
            def extract(side):
                players = teams.get(side,{}).get("players",{})
                return sorted([p for p in players.values() if p.get("battingOrder") and int(p["battingOrder"])<=900],key=lambda x:int(x["battingOrder"]))[:9]
            lineup_away = extract("away")
            lineup_home = extract("home")
        except:
            pass

        all_batters = []

        async def process(batter, team, pit_vuln, pit_reasons, pit_name, pit_hand):
            name     = batter.get("person",{}).get("fullName","")
            pid      = batter.get("person",{}).get("id")
            bat_hand = batter.get("person",{}).get("batSide",{}).get("code","R")
            bc = fuzzy_match(name, _cache["batting_current"])
            bp = fuzzy_match(name, _cache["batting_prior"])
            recent = await fetch_recent_form(pid, days=form_days) if pid else None
            pf = get_park_factor(home_team, bat_hand)
            score, reasons, statline = score_batter(
                bc.to_dict() if bc is not None else {},
                bp.to_dict() if bp is not None else {},
                recent, pf, bat_hand, pit_hand, weather_bonus
            )
            score = min(99, score + pit_vuln)
            reasons += pit_reasons
            if pf >= 1.10: reasons.append(f"HR-friendly park")
            elif pf <= 0.90: reasons.append(f"Pitcher-friendly park")

            all_batters.append({
                "name": name,
                "team": team,
                "score": score,
                "reasons": reasons,
                "opp_pitcher": pit_name,
                "bat_hand": bat_hand,
                "park_factor": round(pf,2),
                "statline": statline,
                "recent_hr": recent.get("hr",0) if recent else 0,
                "recent_pa": recent.get("pa",0) if recent else 0,
                "recent_slg": round(recent.get("slg",0),3) if recent else 0,
            })

        for b in lineup_away:
            await process(b, away_team, hp_vuln, hp_reasons, home_p.get("fullName","TBD"), "R")
        for b in lineup_home:
            await process(b, home_team, ap_vuln, ap_reasons, away_p.get("fullName","TBD"), "R")

        all_batters.sort(key=lambda x: x["score"], reverse=True)

        games_out.append({
            "game_id": gid,
            "away": away_team,
            "home": home_team,
            "time": gtime,
            "away_pitcher": away_p.get("fullName","TBD"),
            "home_pitcher": home_p.get("fullName","TBD"),
            "top_hr_candidates": all_batters[:3],
            "lineups_posted": len(lineup_away)>0 or len(lineup_home)>0,
            "weather": {
                "label": weather_label,
                "desc": weather_desc,
                "temp": temp,
                "wind_speed": wind_speed,
                "wind_dir": wind_dir,
                "bonus": weather_bonus
            }
        })

    return {"games":games_out,"date":today,"loading":False}

@app.post("/refresh-cache")
def refresh_cache():
    _cache["ready"] = False
    threading.Thread(target=load_savant_data, daemon=True).start()
    return {"status":"Cache refresh started"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
