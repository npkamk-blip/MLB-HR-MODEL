from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import pandas as pd
from pybaseball import batting_stats, pitching_stats
from datetime import date
import threading
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SEASON = 2025
MLB_API = "https://statsapi.mlb.com/api/v1"

# Cache so we don't hammer Baseball Savant on every request
_cache = {}

def get_savant_batting():
    if "batting" in _cache:
        return _cache["batting"]
    try:
        df = batting_stats(SEASON, qual=30)
        df = df.rename(columns={
            "Name": "name",
            "Barrel%": "barrel_pct",
            "EV": "exit_velo",
            "LA": "launch_angle",
            "Hard%": "hard_hit_pct",
            "ISO": "iso",
            "HR": "hr",
            "FB%": "fb_pct",
            "HR/FB": "hr_fb_pct",
            "Pull%": "pull_pct",
        })
        keep = ["name", "barrel_pct", "exit_velo", "launch_angle",
                "hard_hit_pct", "iso", "hr", "fb_pct", "hr_fb_pct", "pull_pct"]
        df = df[[c for c in keep if c in df.columns]]
        _cache["batting"] = df
        return df
    except Exception as e:
        print(f"Savant batting error: {e}")
        return pd.DataFrame()

def get_savant_pitching():
    if "pitching" in _cache:
        return _cache["pitching"]
    try:
        df = pitching_stats(SEASON, qual=20)
        df = df.rename(columns={
            "Name": "name",
            "HR/9": "hr_per9",
            "HR/FB": "hr_fb_pct",
            "GB%": "gb_pct",
            "FB%": "fb_pct",
            "Hard%": "hard_hit_pct",
            "ERA": "era",
            "WHIP": "whip",
        })
        keep = ["name", "hr_per9", "hr_fb_pct", "gb_pct",
                "fb_pct", "hard_hit_pct", "era", "whip"]
        df = df[[c for c in keep if c in df.columns]]
        _cache["pitching"] = df
        return df
    except Exception as e:
        print(f"Savant pitching error: {e}")
        return pd.DataFrame()

def score_batter(b_row, p_row):
    score = 0
    reasons = []

    # Barrel rate — 30% weight
    barrel = float(b_row.get("barrel_pct", 0) or 0)
    if barrel > 0:
        s = min(barrel / 15.0, 1.0) * 30
        score += s
        if barrel > 10:
            reasons.append(f"Barrel rate {barrel:.1f}%")

    # ISO — 20% weight
    iso = float(b_row.get("iso", 0) or 0)
    if iso > 0:
        s = min(iso / 0.280, 1.0) * 20
        score += s
        if iso > 0.180:
            reasons.append(f"ISO .{int(iso*1000)}")

    # HR/FB rate — 15% weight
    hr_fb = float(b_row.get("hr_fb_pct", 0) or 0)
    if hr_fb > 0:
        s = min(hr_fb / 25.0, 1.0) * 15
        score += s
        if hr_fb > 15:
            reasons.append(f"HR/FB {hr_fb:.1f}%")

    # Exit velo — 15% weight
    ev = float(b_row.get("exit_velo", 0) or 0)
    if ev > 0:
        s = min((ev - 85) / 15.0, 1.0) * 15
        score += s
        if ev > 92:
            reasons.append(f"Avg EV {ev:.1f} mph")

    # Pitcher vulnerability — 20% weight
    if p_row is not None:
        p_hr9 = float(p_row.get("hr_per9", 0) or 0)
        p_hr_fb = float(p_row.get("hr_fb_pct", 0) or 0)
        p_gb = float(p_row.get("gb_pct", 50) or 50)

        vuln = 0
        if p_hr9 > 0:
            vuln += min(p_hr9 / 2.5, 1.0) * 10
            if p_hr9 > 1.5:
                reasons.append(f"SP allows {p_hr9:.1f} HR/9")
        if p_hr_fb > 0:
            vuln += min(p_hr_fb / 20.0, 1.0) * 5
        if p_gb < 40:
            vuln += 5
            reasons.append("SP is fly ball pitcher")
        score += vuln

    return round(min(score, 99)), reasons

def fuzzy_match(name, df):
    if df.empty or "name" not in df.columns:
        return None
    name_lower = name.lower()
    for _, row in df.iterrows():
        if str(row["name"]).lower() in name_lower or name_lower in str(row["name"]).lower():
            return row
    parts = name_lower.split()
    if len(parts) >= 2:
        last = parts[-1]
        for _, row in df.iterrows():
            if last in str(row["name"]).lower():
                return row
    return None

@app.get("/")
def root():
    return {"status": "MLB HR Model API running"}

@app.get("/games")
async def get_games():
    today = date.today().isoformat()
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{MLB_API}/schedule?sportId=1&date={today}&hydrate=team,probablePitcher,lineupType")
        data = r.json()

    dates = data.get("dates", [])
    if not dates:
        return {"games": [], "date": today}

    bat_df = get_savant_batting()
    pit_df = get_savant_pitching()

    games_out = []
    for game in dates[0].get("games", []):
        if game.get("status", {}).get("abstractGameState") == "Final":
            continue

        game_id = game["gamePk"]
        away_team = game["teams"]["away"]["team"]["name"]
        home_team = game["teams"]["home"]["team"]["name"]
        away_pitcher = game["teams"]["away"].get("probablePitcher", {})
        home_pitcher = game["teams"]["home"].get("probablePitcher", {})
        game_time = game.get("gameDate", "")

        # Fetch lineups
        lineup_away, lineup_home = [], []
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{MLB_API}/game/{game_id}/boxscore")
                box = r.json()
            teams = box.get("teams", {})

            def extract_lineup(side):
                players = teams.get(side, {}).get("players", {})
                return sorted(
                    [p for p in players.values() if p.get("battingOrder") and int(p["battingOrder"]) <= 900],
                    key=lambda x: int(x["battingOrder"])
                )[:9]

            lineup_away = extract_lineup("away")
            lineup_home = extract_lineup("home")
        except:
            pass

        all_batters = []
        for batter in lineup_away:
            name = batter.get("person", {}).get("fullName", "")
            b_row = fuzzy_match(name, bat_df)
            p_row = fuzzy_match(home_pitcher.get("fullName", ""), pit_df)
            score, reasons = score_batter(b_row.to_dict() if b_row is not None else {}, p_row.to_dict() if p_row is not None else None)
            all_batters.append({"name": name, "team": away_team, "score": score, "reasons": reasons, "opp_pitcher": home_pitcher.get("fullName", "TBD")})

        for batter in lineup_home:
            name = batter.get("person", {}).get("fullName", "")
            b_row = fuzzy_match(name, bat_df)
            p_row = fuzzy_match(away_pitcher.get("fullName", ""), pit_df)
            score, reasons = score_batter(b_row.to_dict() if b_row is not None else {}, p_row.to_dict() if p_row is not None else None)
            all_batters.append({"name": name, "team": home_team, "score": score, "reasons": reasons, "opp_pitcher": away_pitcher.get("fullName", "TBD")})

        all_batters.sort(key=lambda x: x["score"], reverse=True)
        top3 = all_batters[:3]

        games_out.append({
            "game_id": game_id,
            "away": away_team,
            "home": home_team,
            "time": game_time,
            "away_pitcher": away_pitcher.get("fullName", "TBD"),
            "home_pitcher": home_pitcher.get("fullName", "TBD"),
            "top_hr_candidates": top3,
            "lineups_posted": len(lineup_away) > 0 or len(lineup_home) > 0
        })

    return {"games": games_out, "date": today}

@app.post("/refresh-cache")
def refresh_cache():
    _cache.clear()
    get_savant_batting()
    get_savant_pitching()
    return {"status": "cache refreshed"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
