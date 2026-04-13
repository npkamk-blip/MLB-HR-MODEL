<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<title>Sharp — MLB Model</title>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Barlow:wght@400;500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0a0a;--surface:#111;--s2:#161616;--s3:#0e0e0e;--card2:#131313;
  --border:#252525;--border2:#1a1a1a;
  --accent:#c8f135;--blue:#4da6ff;--text:#f0f0f0;
  --muted:#888;--muted2:#555;--danger:#ff4d4d;--warn:#ffaa33;--purple:#c084fc;
  --g:#39d353;--y:#ffaa33;--r:#ff4d4d;
}
body{background:var(--bg);color:var(--text);font-family:'Barlow',sans-serif;font-size:14px}
header{border-bottom:1px solid var(--border);padding:10px 20px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;background:var(--bg)}
.logo{font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:22px;color:var(--accent);text-transform:uppercase;letter-spacing:.05em}
.logo-sub{font-size:11px;color:var(--muted);border-left:1px solid var(--border);padding-left:10px;text-transform:uppercase;letter-spacing:.1em}
.hdr-r{display:flex;align-items:center;gap:10px}
.date-nav{display:flex;align-items:center;gap:6px}
.date-btn{background:none;border:1px solid var(--border);border-radius:6px;color:var(--muted);font-size:18px;width:26px;height:26px;display:flex;align-items:center;justify-content:center;cursor:pointer;padding:0}
.date-btn:hover{border-color:var(--accent);color:var(--accent)}
.date-lbl{font-family:'Barlow Condensed',sans-serif;font-size:12px;color:var(--muted);letter-spacing:.06em;text-transform:uppercase;white-space:nowrap}
.refresh-btn{background:var(--accent);color:#0a0a0a;border:none;border-radius:6px;font-family:'Barlow Condensed',sans-serif;font-size:13px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:6px 14px;cursor:pointer}
.refresh-btn:hover{opacity:.85}
.refresh-btn:disabled{opacity:.4;cursor:not-allowed}
.upd{font-size:11px;color:var(--muted2)}
.tabs{display:flex;border-bottom:1px solid var(--border);background:var(--surface);padding:0 20px;position:sticky;top:49px;z-index:99}
.tab{background:none;border:none;border-bottom:2px solid transparent;color:var(--muted);font-family:'Barlow Condensed',sans-serif;font-size:13px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;padding:10px 18px;cursor:pointer;transition:all .15s;margin-bottom:-1px}
.tab:hover{color:var(--text)}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab-panel{display:none}
.tab-panel.active{display:block}
.container{max-width:1300px;margin:0 auto;padding:16px}
.loading{text-align:center;padding:60px 20px;color:var(--muted);font-family:'Barlow Condensed',sans-serif;font-size:18px;letter-spacing:.1em}
.dots{display:inline-flex;gap:6px;margin-left:10px}
.dots span{width:6px;height:6px;background:var(--accent);border-radius:50%;animation:dot .8s infinite alternate}
.dots span:nth-child(2){animation-delay:.2s}
.dots span:nth-child(3){animation-delay:.4s}
@keyframes dot{from{opacity:.2}to{opacity:1}}
.err{background:rgba(255,77,77,.08);border:1px solid rgba(255,77,77,.2);border-radius:8px;padding:16px;color:var(--danger);margin:20px 0}
.no-data{text-align:center;padding:40px;color:var(--muted2);font-size:13px}
.section-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.section-title{font-family:'Barlow Condensed',sans-serif;font-size:16px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--accent)}
.section-count{font-size:11px;color:var(--muted2)}
.tbl-wrap{overflow-x:auto;border-radius:10px;border:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:12px}
thead th{background:var(--s2);padding:8px 10px;text-align:left;font-family:'Barlow Condensed',sans-serif;font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border);white-space:nowrap;cursor:pointer;user-select:none}
thead th:hover{color:var(--text)}
thead th.sort-asc::after{content:' ▲';color:var(--accent)}
thead th.sort-desc::after{content:' ▼';color:var(--accent)}
thead th.num{text-align:right}
tbody tr{border-bottom:1px solid var(--border2);cursor:pointer;transition:background .1s}
tbody tr:hover{background:var(--s2)}
tbody tr.detail-row{cursor:default;background:var(--s3)}
tbody tr.detail-row:hover{background:var(--s3)}
td{padding:8px 10px;vertical-align:middle}
td.num{text-align:right;font-family:'Barlow Condensed',sans-serif;font-size:13px;font-weight:600}
.player-name{font-weight:600;font-size:13px}
.player-sub{font-size:10px;color:var(--muted);margin-top:1px}
.hand-tag{display:inline-block;font-size:9px;font-weight:700;padding:1px 5px;border-radius:4px;text-transform:uppercase}
.rhb{background:rgba(200,241,53,.08);color:var(--accent);border:1px solid rgba(200,241,53,.2)}
.lhb{background:rgba(77,166,255,.12);color:var(--blue);border:1px solid rgba(77,166,255,.25)}
.rhp{background:rgba(200,241,53,.08);color:var(--accent);border:1px solid rgba(200,241,53,.2)}
.lhp{background:rgba(77,166,255,.12);color:var(--blue);border:1px solid rgba(77,166,255,.25)}
.sc{font-family:'Barlow Condensed',sans-serif;font-size:13px;font-weight:600}
.sc-g{color:var(--g)}
.sc-y{color:var(--y)}
.sc-r{color:var(--r)}
.sc-n{color:var(--muted2)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:3px;vertical-align:middle}
.dot-g{background:var(--g)}
.dot-y{background:var(--y)}
.dot-r{background:var(--r)}
.dot-n{background:var(--muted2)}
.prob-big{font-family:'Barlow Condensed',sans-serif;font-size:16px;font-weight:800}
.p-elite{color:var(--accent)}
.p-high{color:var(--g)}
.p-mid{color:var(--y)}
.p-low{color:var(--muted)}
.l8d-badge{display:inline-block;font-size:10px;font-weight:700;padding:2px 7px;border-radius:10px}
.l8d-hot{background:rgba(57,211,83,.12);color:var(--g);border:1px solid rgba(57,211,83,.25)}
.l8d-ok{background:rgba(255,170,51,.1);color:var(--y);border:1px solid rgba(255,170,51,.2)}
.l8d-cold{background:#1a1a1a;color:var(--muted2);border:1px solid var(--border2)}
.detail-wrap{padding:12px 16px;background:var(--s3)}
.dstat-wrap{background:var(--surface);border:1px solid var(--border2);border-radius:8px;overflow:hidden;margin-bottom:10px}
.drow{display:grid;grid-template-columns:90px 1fr 1fr 1fr;gap:0;border-bottom:1px solid var(--border2);padding:6px 10px;align-items:center}
.drow:last-child{border-bottom:none}
.drow-hdr{background:var(--s2)}
.drow-hdr .drow-s,.drow-hdr .drow-l,.drow-hdr .drow-sp{font-size:9px;font-weight:700;color:var(--muted2);text-transform:uppercase;letter-spacing:.06em;text-align:center}
.drow-lbl{font-size:11px;color:var(--muted);font-weight:500}
.drow-s,.drow-l,.drow-sp{text-align:center}
.drow-note{font-size:9px;color:var(--muted2);grid-column:span 1;text-align:right}
.dmodel-row{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.dmodel-block{background:var(--surface);border:1px solid var(--border2);border-radius:6px;padding:5px 10px;display:flex;flex-direction:column;align-items:center;min-width:60px}
.dmodel-lbl{font-size:8px;color:var(--muted2);text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px}
.detail-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px}
.detail-block{background:var(--surface);border:1px solid var(--border2);border-radius:8px;padding:10px 12px}
.detail-block-title{font-size:9px;font-weight:700;color:var(--muted2);text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}
.chip-row{display:flex;flex-wrap:wrap;gap:4px}
.chip{display:inline-flex;flex-direction:column;align-items:center;background:#1a1a1a;border:1px solid var(--border2);border-radius:6px;padding:5px 8px;min-width:54px}
.chip-lbl{font-size:8px;color:var(--muted2);text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px}
.chip-val{font-family:'Barlow Condensed',sans-serif;font-size:13px;font-weight:700}
.chip-val.g{color:var(--g)}
.chip-val.y{color:var(--y)}
.chip-val.r{color:var(--r)}
.chip-val.n{color:var(--muted2)}
.edge-pos{color:var(--g);font-weight:700}
.edge-neg{color:var(--r)}
.edge-neu{color:var(--muted2)}
.consistency-badge{display:inline-block;font-size:9px;font-weight:700;padding:2px 6px;border-radius:10px}
.con-solid{background:rgba(57,211,83,.1);color:var(--g);border:1px solid rgba(57,211,83,.2)}
.con-mod{background:rgba(255,170,51,.1);color:var(--y);border:1px solid rgba(255,170,51,.2)}
.con-small{background:rgba(255,170,51,.08);color:var(--warn);border:1px solid rgba(255,170,51,.15)}
.con-none{background:#1a1a1a;color:var(--muted2);border:1px solid var(--border2)}
.team-a{color:var(--blue)}
.team-h{color:var(--warn)}
.at-sep{color:var(--muted2);margin:0 4px}
.ou-over{color:var(--r)}
.ou-under{color:var(--blue)}
.ou-push{color:var(--muted)}
.exp-runs-big{font-family:'Barlow Condensed',sans-serif;font-size:15px;font-weight:700}
.search-row{display:flex;gap:8px;margin-bottom:20px}
.search-input{flex:1;background:var(--surface);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:'Barlow',sans-serif;font-size:14px;padding:10px 14px;outline:none}
.search-input:focus{border-color:var(--accent)}
.search-btn{background:var(--accent);color:#0a0a0a;border:none;border-radius:8px;font-family:'Barlow Condensed',sans-serif;font-size:13px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:10px 20px;cursor:pointer}
.r-chip{background:var(--s2);border:1px solid var(--border);border-radius:8px;padding:8px 12px;min-width:64px;text-align:center;display:inline-block;margin:3px}
.r-chip-lbl{font-size:9px;color:var(--muted2);text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px}
.r-chip-val{font-family:'Barlow Condensed',sans-serif;font-size:15px;font-weight:700}
.r-section{margin-bottom:16px}
.r-section-title{font-size:10px;font-weight:700;color:var(--muted2);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}
.research-result{background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden}
.r-header{background:var(--s2);padding:14px 18px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.r-name{font-family:'Barlow Condensed',sans-serif;font-size:22px;font-weight:800}
.r-body{padding:16px 18px}
</style>
</head>
<body>
<header>
  <div style="display:flex;align-items:center;gap:12px">
    <div class="logo">Sharp</div>
    <div class="logo-sub">MLB Model</div>
  </div>
  <div class="hdr-r">
    <div class="date-nav">
      <button class="date-btn" onclick="changeDate(-1)">&#8249;</button>
      <span class="date-lbl" id="dateLbl"></span>
      <button class="date-btn" onclick="changeDate(1)">&#8250;</button>
    </div>
    <button class="refresh-btn" id="refreshBtn" onclick="load()">Refresh</button>
    <span class="upd" id="upd"></span>
  </div>
</header>
<div class="tabs">
  <button class="tab active" onclick="switchTab('batters',this)">Batters</button>
  <button class="tab" onclick="switchTab('pitchers',this)">Pitchers</button>
  <button class="tab" onclick="switchTab('games',this)">Games</button>
  <button class="tab" onclick="switchTab('research',this)">Research</button>
</div>

<div id="batters-panel" class="tab-panel active">
  <div class="container">
    <div id="batter-main"><div class="loading">Loading<div class="dots"><span></span><span></span><span></span></div></div></div>
  </div>
</div>
<div id="pitchers-panel" class="tab-panel">
  <div class="container">
    <div id="pitcher-main"><div class="no-data">Load data first</div></div>
  </div>
</div>
<div id="games-panel" class="tab-panel">
  <div class="container">
    <div id="game-main"><div class="no-data">Load data first</div></div>
  </div>
</div>
<div id="research-panel" class="tab-panel">
  <div class="container">
    <div class="search-row">
      <input class="search-input" id="searchInput" placeholder="Search player name..." onkeydown="if(event.key==='Enter')searchPlayer()">
      <button class="search-btn" onclick="searchPlayer()">Search</button>
    </div>
    <div id="research-result"></div>
  </div>
</div>

<script>
var API='https://web-production-c0e4.up.railway.app';
var allGames=[];
var selectedDate=new Date();
var sortCol='hr_prob',sortDir=-1;
var pitSortCol='exp_k',pitSortDir=-1;

function formatDateForAPI(d){var y=d.getFullYear(),m=String(d.getMonth()+1).padStart(2,'0'),day=String(d.getDate()).padStart(2,'0');return y+'-'+m+'-'+day;}
function fmtDateLabel(d){return d.toLocaleDateString('en-US',{weekday:'short',month:'short',day:'numeric'});}
function changeDate(n){selectedDate=new Date(selectedDate);selectedDate.setDate(selectedDate.getDate()+n);document.getElementById('dateLbl').textContent=fmtDateLabel(selectedDate);load();}
function fmtTime(iso){if(!iso)return'--';try{var d=new Date(iso);return d.toLocaleTimeString('en-US',{hour:'numeric',minute:'2-digit',timeZone:'America/New_York'});}catch(e){return'--';}}

function switchTab(id,btn){
  document.querySelectorAll('.tab-panel').forEach(function(p){p.classList.remove('active');});
  document.querySelectorAll('.tab').forEach(function(t){t.classList.remove('active');});
  document.getElementById(id+'-panel').classList.add('active');
  btn.classList.add('active');
}

function probCls(p){return p>=20?'p-elite':p>=15?'p-high':p>=10?'p-mid':'p-low';}
function barrelCls(v){return v>=12?'sc-g':v>=6?'sc-y':'sc-r';}
function laCls(v){return(v>=25&&v<=35)?'sc-g':(v>=20&&v<=40)?'sc-y':'sc-r';}
function isoCls(v){return v>=0.200?'sc-g':v>=0.150?'sc-y':'sc-r';}
function fmtIso(v){return v>0?'.'+String(Math.round(v*1000)).padStart(3,'0'):'--';}
function chip(lbl,val,cls){return '<div class="chip"><div class="chip-lbl">'+lbl+'</div><div class="chip-val '+cls+'">'+val+'</div></div>';}
function barrelCls2(v){return v>=12?'g':v>=6?'y':'r';}
function laCls2(v){return(v>=25&&v<=35)?'g':(v>=20&&v<=40)?'y':'r';}
function isoCls2(v){return v>=0.200?'g':v>=0.150?'y':'r';}
function kCls(v){return v>=30?'r':v>=22?'y':'g';}

function divergenceFlag(season,l8d,higherBetter){
  if(!l8d||!season||season===0)return{val:season,flag:''};
  var rel=Math.abs(l8d-season)/season;
  if(rel<0.50)return{val:season,flag:''};
  var goingUp=l8d>season;
  var isHot=higherBetter?goingUp:!goingUp;
  return{val:l8d,flag:isHot?'🔥':'❄️'};
}
function laDivergence(season,l8d){
  if(!l8d||!season||season===0)return{val:season,flag:''};
  var rel=Math.abs(l8d-season)/Math.max(Math.abs(season),1);
  if(rel<0.50)return{val:season,flag:''};
  var sweet=30,ds=Math.abs(season-sweet),dl=Math.abs(l8d-sweet);
  return{val:l8d,flag:dl<ds?'🔥':'❄️'};
}
function batPlatoonDot(isoVsHand,isoOverall){
  if(!isoVsHand||!isoOverall||isoOverall===0)return'n';
  var r=isoVsHand/isoOverall;
  return r>=1.15?'g':r>=0.90?'y':'r';
}
function pitPlatoonDot(slgVsBat,slgOverall){
  if(!slgVsBat||!slgOverall||slgOverall===0)return'n';
  var r=slgVsBat/slgOverall;
  return r>=1.10?'g':r>=0.90?'y':'r';
}
function wxIcon(label,temp,wind){
  var l=(label||'').toLowerCase();
  if(l.includes('dome'))return'&#127967;';
  if(l.includes('rain'))return'&#127783;';
  if(l.includes('out')&&wind>=10)return'&#127782;&#8593;';
  if(l.includes('in')&&wind>=10)return'&#127782;&#8595;';
  if(temp<50)return'&#129398;';
  if(temp>=80)return'&#9728;';
  return'&#9925;';
}
function wxTextCls(label){
  var l=(label||'').toLowerCase();
  if(l.includes('out'))return'color:var(--g)';
  if(l.includes('in'))return'color:var(--r)';
  return'color:var(--muted)';
}

function buildBatterTable(games){
  var batters=[];
  games.forEach(function(g){
    var wx=g.weather||{};
    var awayP=g.away_pitcher||{};
    var homeP=g.home_pitcher||{};
    var all=(g.away_lineup||[]).concat(g.home_lineup||[]);
    if(!all.length)all=g.top_hr_candidates||[];
    all.forEach(function(b){
      if((b.hr_prob||0)<8)return;
      var bd=b.breakdown||{};
      var pitP=b.team===g.away?homeP:awayP;
      batters.push({
        name:b.name,team:b.team,hand:b.bat_hand||'R',
        opp:b.opp_pitcher||'TBD',opp_hand:b.opp_p_hand||'R',
        pitcher_name:pitP.name||b.opp_pitcher||'TBD',
        barrel_s:b.season?b.season.barrel||0:0,
        barrel_8d:b.l8d?b.l8d.barrel||0:0,
        la_s:b.season?b.season.la||0:0,
        la_8d:b.l8d?b.l8d.la||0:0,
        iso_vs_hand:bd.iso_vs_hand||0,
        iso_overall:bd.iso_overall||0,
        l8d_hr:b.l8d_hr_count||0,
        l8d_pa:b.l8d?b.l8d.pa||0:0,
        wx_label:wx.label||'--',wx_temp:wx.temp||70,wx_wind:wx.wind_speed||0,
        bat_platoon_mult:bd.bat_platoon_mult||1,
        pit_platoon_mult:bd.pit_platoon_mult||1,
        slg_vs_bat:bd.pit_slg_vs_bat||0,
        slg_overall:bd.pit_slg_overall||0,
        hr9_vs_bat:bd.hr9_split||0,
        hr_prob:b.hr_prob||0,
        _full:b,_pitP:pitP,_wx:wx,
      });
    });
  });
  batters.sort(function(a,b){return sortDir*((b[sortCol]||0)-(a[sortCol]||0));});

  if(!batters.length){document.getElementById('batter-main').innerHTML='<div class="no-data">No batters above 8% HR probability today.</div>';return;}

  var h='<div class="section-hdr"><div class="section-title">HR Probabilities</div><div class="section-count">'+batters.length+' batters &middot; 8%+ threshold</div></div>';
  h+='<div class="tbl-wrap"><table>';
  h+='<thead><tr>';
  h+=bth('Batter','name',false);h+=bth('Hand','hand',false);h+=bth('Pitcher','pitcher_name',false);h+=bth('P-Hand','opp_hand',false);
  h+=bth('P-Split SLG','slg_vs_bat',true);h+=bth('HR/9 vs Hand','hr9_vs_bat',true);h+=bth('Batter ISO','iso_vs_hand',true);
  h+=bth('Barrel%','barrel_s',true);h+=bth('LA','la_s',true);h+=bth('L8D HR','l8d_hr',true);
  h+=bth('Weather','wx_label',false);h+=bth('HR%','hr_prob',true);
  h+='</tr></thead><tbody>';

  batters.forEach(function(b,i){
    var bd=b._full.breakdown||{};
    var brl=divergenceFlag(b.barrel_s,b.barrel_8d,true);
    var la=laDivergence(b.la_s,b.la_8d);
    var batDot=batPlatoonDot(b.iso_vs_hand,b.iso_overall);
    var pitDot=pitPlatoonDot(b.slg_vs_bat,b.slg_overall);
    var hr9C=b.hr9_vs_bat>=1.4?'sc-g':b.hr9_vs_bat>=1.0?'sc-y':b.hr9_vs_bat>0?'sc-r':'sc-n';
    var l8dBadge=b.l8d_hr>=3?'l8d-hot':b.l8d_hr>=1?'l8d-ok':'l8d-cold';
    var pc=probCls(b.hr_prob);
    var handTag='<span class="hand-tag '+(b.hand==='L'?'lhb':'rhb')+'">'+b.hand+'</span>';
    var oppHTag='<span class="hand-tag '+(b.opp_hand==='L'?'lhp':'rhp')+'">'+b.opp_hand+'</span>';
    h+='<tr onclick="toggleBatRow('+i+')">';
    h+='<td><div class="player-name">'+b.name+'</div><div class="player-sub">'+b.team+'</div></td>';
    h+='<td>'+handTag+'</td>';
    h+='<td><div class="player-name">'+(b.pitcher_name||'TBD')+'</div></td>';
    h+='<td>'+oppHTag+'</td>';
    // P-Split SLG (pitcher weak vs batter hand) — min 20 PA threshold, inverted dot
    var pitSplgDisp=b.slg_vs_bat>0&&b._full.breakdown&&(b._full.breakdown.split_ip_vs_bat||0)>=5?b.slg_vs_bat.toFixed(3):'--';
    h+='<td class="num"><span class="dot dot-'+pitDot+'"></span><span class="sc-n">'+pitSplgDisp+'</span></td>';
    h+='<td class="num sc '+hr9C+'">'+(b.hr9_vs_bat>0?b.hr9_vs_bat.toFixed(2):'--')+'</td>';
    // Batter ISO vs pitcher hand
    h+='<td class="num"><span class="dot dot-'+batDot+'"></span><span class="sc '+isoCls(b.iso_vs_hand)+'">'+(b.iso_vs_hand>0?fmtIso(b.iso_vs_hand):b.iso_overall>0?fmtIso(b.iso_overall):'--')+'</span></td>';
    h+='<td class="num"><span class="sc '+barrelCls(brl.val)+'">'+(brl.val>0?brl.val.toFixed(1)+'%':'--')+'</span>'+(brl.flag?'<span style="font-size:10px">'+brl.flag+'</span>':'')+'</td>';
    h+='<td class="num"><span class="sc '+laCls(la.val)+'">'+(la.val>0?la.val.toFixed(1)+'&#176;':'--')+'</span>'+(la.flag?'<span style="font-size:10px">'+la.flag+'</span>':'')+'</td>';
    h+='<td class="num"><span class="l8d-badge '+l8dBadge+'">'+b.l8d_hr+' HR</span></td>';
    h+='<td style="font-size:11px;'+wxTextCls(b.wx_label)+'">'+wxIcon(b.wx_label,b.wx_temp,b.wx_wind)+' '+(b.wx_temp||'--')+'&#176;<span style="color:var(--muted2);font-size:10px;margin-left:2px">'+(b.wx_wind>0?b.wx_wind+'mph':'calm')+'</span></td>';
    h+='<td class="num"><span class="prob-big '+pc+'">'+b.hr_prob+'%</span></td>';
    h+='</tr>';
    h+='<tr class="detail-row" id="bdet-'+i+'" style="display:none"><td colspan="12">'+buildBatDetail(b)+'</td></tr>';
  });
  h+='</tbody></table></div>';
  document.getElementById('batter-main').innerHTML=h;
}

function bth(lbl,col,isNum){
  var cls=isNum?'num':'';
  var sc=sortCol===col?(sortDir===-1?' sort-desc':' sort-asc'):'';
  return '<th class="'+cls+sc+'" onclick="sortBat(\''+col+'\')">'+lbl+'</th>';
}
function sortBat(col){if(sortCol===col)sortDir*=-1;else{sortCol=col;sortDir=-1;}buildBatterTable(allGames);}
function toggleBatRow(i){var d=document.getElementById('bdet-'+i);if(d)d.style.display=d.style.display==='none'?'table-row':'none';}

function buildBatDetail(b){
  var s=b._full.season||{},l=b._full.l8d||{},bd=b._full.breakdown||{},pitP=b._pitP||{};
  var oppHand=b.opp_hand||b._full.opp_p_hand||'R';
  var splitLabel='vs '+oppHand+'HP';
  var hasL8D=l.pa>0;
  var hasSplit=bd.split_pa>0;
  var l8dHR=b.l8d_hr||b._full.l8d_hr_count||0;

  // Stat row helper: label | season | l8d | split
  function statRow(lbl, sVal, sCls, l8dVal, l8dCls, spVal, spCls, note){
    var r='<div class="drow">';
    r+='<div class="drow-lbl">'+lbl+'</div>';
    r+='<div class="drow-s"><span class="chip-val '+sCls+'">'+sVal+'</span></div>';
    r+='<div class="drow-l"><span class="chip-val '+(hasL8D?l8dCls:'n')+'">'+( hasL8D?l8dVal:'--')+'</span></div>';
    r+='<div class="drow-sp"><span class="chip-val '+(hasSplit?spCls:'n')+'">'+( hasSplit?spVal:'--')+'</span></div>';
    if(note)r+='<div class="drow-note">'+note+'</div>';
    r+='</div>';
    return r;
  }

  var h='<div class="detail-wrap">';

  // ── Stat table ──
  h+='<div class="dstat-wrap">';
  // Header
  h+='<div class="drow drow-hdr"><div class="drow-lbl"></div><div class="drow-s">Season</div><div class="drow-l">L8D</div><div class="drow-sp">'+splitLabel+'</div></div>';

  // Barrel%
  var brlDiv=hasL8D&&s.barrel>0?Math.abs(l.barrel-s.barrel)/s.barrel:0;
  var brlFlag=brlDiv>=0.50?(l.barrel>s.barrel?'🔥':'❄️'):'';
  h+=statRow('Barrel%',
    s.barrel>0?s.barrel.toFixed(1)+'%':'--', barrelCls2(s.barrel),
    l.barrel>0?l.barrel.toFixed(1)+'%'+brlFlag:'--', barrelCls2(l.barrel),
    bd.barrel_use>0?bd.barrel_use.toFixed(1)+'%':'--', barrelCls2(bd.barrel_use));

  // Launch Angle
  var laDiv=hasL8D&&s.la>0?Math.abs(l.la-s.la)/Math.max(Math.abs(s.la),1):0;
  var laFlag='';
  if(laDiv>=0.50){var sweet=30,ds=Math.abs(s.la-sweet),dl=Math.abs(l.la-sweet);laFlag=dl<ds?'🔥':'❄️';}
  h+=statRow('Launch Angle',
    s.la>0?s.la.toFixed(1)+'°':'--', laCls2(s.la),
    l.la>0?l.la.toFixed(1)+'°'+laFlag:'--', laCls2(l.la),
    '--', 'n');

  // ISO
  var isoDiv=hasL8D&&s.iso>0?Math.abs(l.iso-s.iso)/s.iso:0;
  var isoFlag=isoDiv>=0.50?(l.iso>s.iso?'🔥':'❄️'):'';
  var spISO=bd.iso_vs_hand>0?fmtIso(bd.iso_vs_hand):'--';
  h+=statRow('ISO',
    s.iso>0?fmtIso(s.iso):'--', isoCls2(s.iso),
    l.iso>0?fmtIso(l.iso)+isoFlag:'--', isoCls2(l.iso),
    spISO, isoCls2(bd.iso_vs_hand));

  // K%
  h+=statRow('K%',
    s.k>0?s.k.toFixed(1)+'%':'--', kCls(s.k),
    l.pa>0&&b._full.l8d&&b._full.l8d.k_pct>0?b._full.l8d.k_pct.toFixed(1)+'%':'--', kCls(b._full.l8d&&b._full.l8d.k_pct||0),
    bd.split_pa>0&&bd.k_s>0?bd.k_s.toFixed(1)+'%':'--', kCls(bd.k_s||0));

  // HR
  h+=statRow('HR',
    String(s.hr||0), s.hr>=5?'g':s.hr>=2?'y':'n',
    String(l8dHR), l8dHR>=3?'g':l8dHR>=1?'y':'n',
    bd.split_hr>0?String(bd.split_hr):'--', bd.split_hr>=3?'g':bd.split_hr>=1?'y':'n',
    hasSplit?'('+bd.split_pa+' PA)':'');

  // SLG (split only — season SLG shown in season chips below, split is the key one)
  var spSLG=bd.split_slg>0&&bd.split_pa>=20?'.'+String(Math.round(bd.split_slg*1000)).padStart(3,'0'):'--';
  var l8dSLG=l.slg>0?'.'+String(Math.round(l.slg*1000)).padStart(3,'0'):'--';
  var sSLG=s.slg>0?'.'+String(Math.round(s.slg*1000)).padStart(3,'0'):'--';
  h+=statRow('SLG', sSLG,'n', l8dSLG,'n', spSLG,'n');

  h+='</div>'; // dstat-wrap

  // ── Model multipliers row ──
  h+='<div class="dmodel-row">';
  h+='<div class="dmodel-block"><span class="dmodel-lbl">Base</span><span class="chip-val n">'+(bd.base_rate>0?bd.base_rate.toFixed(1)+'%':'--')+'</span></div>';
  h+='<div class="dmodel-block"><span class="dmodel-lbl">Barrel×</span><span class="chip-val '+(bd.barrel_mult>=1.5?'g':bd.barrel_mult>=1?'y':'r')+'">'+(bd.barrel_mult>0?bd.barrel_mult.toFixed(2)+'x':'--')+'</span></div>';
  h+='<div class="dmodel-block"><span class="dmodel-lbl">LA×</span><span class="chip-val '+(bd.la_mult>=1?'g':bd.la_mult>=0.9?'y':'r')+'">'+(bd.la_mult>0?bd.la_mult.toFixed(2)+'x':'--')+'</span></div>';
  h+='<div class="dmodel-block"><span class="dmodel-lbl">Pit×</span><span class="chip-val '+(bd.pit_vuln_mult>=1.3?'g':bd.pit_vuln_mult>=1?'y':'r')+'">'+(bd.pit_vuln_mult>0?bd.pit_vuln_mult.toFixed(2)+'x':'--')+'</span></div>';
  h+='<div class="dmodel-block"><span class="dmodel-lbl">BatPlat×</span><span class="chip-val '+(bd.bat_platoon_mult>=1.15?'g':bd.bat_platoon_mult>=0.9?'y':'r')+'">'+(bd.bat_platoon_mult>0?bd.bat_platoon_mult.toFixed(2)+'x':'--')+'</span></div>';
  h+='<div class="dmodel-block"><span class="dmodel-lbl">PitPlat×</span><span class="chip-val '+(bd.pit_platoon_mult>=1.15?'g':bd.pit_platoon_mult>=0.9?'y':'r')+'">'+(bd.pit_platoon_mult>0?bd.pit_platoon_mult.toFixed(2)+'x':'--')+'</span></div>';
  h+='<div class="dmodel-block"><span class="dmodel-lbl">Park×</span><span class="chip-val '+(bd.park_factor>=1.10?'g':bd.park_factor>=0.95?'y':'r')+'">'+(bd.park_factor>0?bd.park_factor.toFixed(2)+'x':'--')+'</span></div>';
  h+='<div class="dmodel-block"><span class="dmodel-lbl">Hot×</span><span class="chip-val '+(bd.hot_cold_mult>=1.1?'g':bd.hot_cold_mult<=0.9?'r':'y')+'">'+(bd.hot_cold_mult>0?bd.hot_cold_mult.toFixed(2)+'x':'--')+'</span></div>';
  h+='<div class="dmodel-block"><span class="dmodel-lbl">K%×</span><span class="chip-val '+(bd.k_mult>=1?'g':bd.k_mult>=0.94?'y':'r')+'">'+(bd.k_mult>0?bd.k_mult.toFixed(2)+'x':'--')+'</span></div>';
  h+='</div>';

  // ── Pitcher + pitch matchup ──
  h+='<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:8px">';
  if(pitP&&pitP.name){
    h+='<div class="detail-block" style="flex:1;min-width:200px"><div class="detail-block-title">'+pitP.name+' ('+pitP.hand+'HP)</div><div class="chip-row">';
    h+=chip('ERA',pitP.era?pitP.era.toFixed(2):'--',pitP.era>=4.5?'r':pitP.era>=3.5?'y':'g');
    h+=chip('HR/9',pitP.hr9?pitP.hr9.toFixed(2):'--',pitP.hr9>=1.4?'g':pitP.hr9>=1?'y':'r');
    h+=chip('HH%',pitP.hard_hit_pct?pitP.hard_hit_pct+'%':'--',pitP.hard_hit_pct>=42?'r':pitP.hard_hit_pct>=36?'y':'g');
    var slgKey='vs_'+(b.hand==='L'?'L':'R')+'_slg';
    var pitSplgV=pitP[slgKey];
    h+=chip('SLG vs '+b.hand,pitSplgV&&bd.split_ip_vs_bat>=5?'.'+String(Math.round(pitSplgV*1000)).padStart(3,'0'):'--','n');
    h+='</div></div>';
  }
  if(bd.pitch_breakdown&&bd.pitch_breakdown.length){
    h+='<div class="detail-block" style="flex:2;min-width:240px"><div class="detail-block-title">Pitch Matchup</div><div class="chip-row">';
    bd.pitch_breakdown.forEach(function(p){
      var rv=p.bat_rv||0;var rvc=rv>=2?'g':rv<=-2?'r':'n';
      h+=chip(p.name+' '+p.usage+'%',(rv>0?'+':'')+rv,rvc);
    });
    h+='</div></div>';
  }
  h+='</div>';

  h+='</div>'; // detail-wrap
  return h;
}

function buildPitcherTable(games){
  var pitchers=[];
  games.forEach(function(g){
    var wx=g.weather||{};
    [[g.away_pitcher,g.home],[g.home_pitcher,g.away]].forEach(function(arr){
      var p=arr[0],opp=arr[1];
      if(!p||!p.name||p.name==='TBD')return;
      pitchers.push({
        name:p.name,hand:p.hand||'R',opp_team:opp,
        opp_lineup_k:p.opp_lineup_k||0,avg_ip:p.avg_ip||5.0,
        k9:p.k9||0,exp_k:p.exp_k||0,k_prop:p.k_prop||null,
        k_edge:p.k_edge!=null?p.k_edge:null,gs:p.gs||0,
        era:p.era||0,hr9:p.hr9||0,
        wx_label:wx.label||'--',wx_temp:wx.temp||70,wx_wind:wx.wind_speed||0,
        _full:p,
      });
    });
  });
  pitchers.sort(function(a,b){
    if(pitSortCol==='k_edge'){
      var av=a.k_edge!=null?a.k_edge:-99,bv=b.k_edge!=null?b.k_edge:-99;
      return pitSortDir*(bv-av);
    }
    return pitSortDir*((b[pitSortCol]||0)-(a[pitSortCol]||0));
  });
  if(!pitchers.length){document.getElementById('pitcher-main').innerHTML='<div class="no-data">No pitcher data.</div>';return;}
  var h='<div class="section-hdr"><div class="section-title">Pitcher Strikeouts</div><div class="section-count">'+pitchers.length+' starters</div></div>';
  h+='<div class="tbl-wrap"><table>';
  h+='<thead><tr>';
  h+=pth('Pitcher','name',false);h+=pth('Hand','hand',false);h+=pth('Vs Team','opp_team',false);
  h+=pth('OPP K%','opp_lineup_k',true);h+=pth('Avg IP','avg_ip',true);h+=pth('K/9','k9',true);
  h+=pth('Exp Ks','exp_k',true);h+=pth('Prop Line','k_prop',true);h+=pth('Edge','k_edge',true);
  h+=pth('Starts','gs',false);
  h+='</tr></thead><tbody>';
  pitchers.forEach(function(p,i){
    var hTag='<span class="hand-tag '+(p.hand==='L'?'lhp':'rhp')+'">'+p.hand+'HP</span>';
    var oppKCls=p.opp_lineup_k>=25?'sc-g':p.opp_lineup_k>=20?'sc-y':'sc-r';
    var k9Cls=p.k9>=10?'sc-g':p.k9>=8?'sc-y':'sc-r';
    var edgeCls=p.k_edge!=null?(p.k_edge>=1?'edge-pos':p.k_edge<=-1?'edge-neg':'edge-neu'):'edge-neu';
    var edgeDisp=p.k_edge!=null?(p.k_edge>0?'+':'')+p.k_edge:'--';
    var propDisp=p.k_prop?p.k_prop.line.toFixed(1):'--';
    var conBadge=p.gs<=0?'<span class="consistency-badge con-none">No Data</span>':p.gs<=2?'<span class="consistency-badge con-small">'+p.gs+' GS</span>':p.gs<=5?'<span class="consistency-badge con-mod">'+p.gs+' GS</span>':'<span class="consistency-badge con-solid">'+p.gs+' GS</span>';
    h+='<tr onclick="togglePitRow('+i+')">';
    h+='<td><div class="player-name">'+p.name+'</div></td>';
    h+='<td>'+hTag+'</td>';
    h+='<td style="font-size:12px;color:var(--muted)">'+p.opp_team+'</td>';
    h+='<td class="num sc '+oppKCls+'">'+(p.opp_lineup_k>0?p.opp_lineup_k.toFixed(1)+'%':'--')+'</td>';
    h+='<td class="num sc-n">'+(p.avg_ip>0?p.avg_ip.toFixed(1):'--')+'</td>';
    h+='<td class="num sc '+k9Cls+'">'+(p.k9>0?p.k9.toFixed(1):'--')+'</td>';
    h+='<td class="num"><span style="font-family:\'Barlow Condensed\',sans-serif;font-size:18px;font-weight:800;color:var(--accent)">'+(p.exp_k>0?p.exp_k.toFixed(1):'--')+'</span></td>';
    h+='<td class="num" style="font-family:\'Barlow Condensed\',sans-serif;font-size:14px;font-weight:600">'+propDisp+'</td>';
    h+='<td class="num '+edgeCls+'">'+edgeDisp+'</td>';
    h+='<td>'+conBadge+'</td>';
    h+='</tr>';
    h+='<tr class="detail-row" id="pdet-'+i+'" style="display:none"><td colspan="10">'+buildPitDetail(p._full)+'</td></tr>';
  });
  h+='</tbody></table></div>';
  document.getElementById('pitcher-main').innerHTML=h;
}
function pth(lbl,col,isNum){
  var cls=isNum?'num':'';
  var sc=pitSortCol===col?(pitSortDir===-1?' sort-desc':' sort-asc'):'';
  return '<th class="'+cls+sc+'" onclick="sortPit(\''+col+'\')">'+lbl+'</th>';
}
function sortPit(col){if(pitSortCol===col)pitSortDir*=-1;else{pitSortCol=col;pitSortDir=-1;}buildPitcherTable(allGames);}
function togglePitRow(i){var d=document.getElementById('pdet-'+i);if(d)d.style.display=d.style.display==='none'?'table-row':'none';}
function buildPitDetail(p){
  var h='<div class="detail-wrap"><div class="detail-grid">';
  h+='<div class="detail-block"><div class="detail-block-title">Stats</div><div class="chip-row">';
  h+=chip('ERA',p.era?p.era.toFixed(2):'--',p.era>=4.5?'r':p.era>=3.5?'y':'g');
  h+=chip('HR/9',p.hr9?p.hr9.toFixed(2):'--',p.hr9>=1.4?'r':p.hr9>=1?'y':'g');
  h+=chip('HH%',p.hard_hit_pct?p.hard_hit_pct+'%':'--',p.hard_hit_pct>=42?'r':p.hard_hit_pct>=36?'y':'g');
  h+=chip('IP',p.ip_2026?p.ip_2026.toFixed(0):'--','n');
  h+='</div></div>';
  h+='<div class="detail-block"><div class="detail-block-title">Splits</div><div class="chip-row">';
  h+=chip('HR/9 vs L',p.vs_L_hr9!=null?p.vs_L_hr9.toFixed(2):'--',p.vs_L_hr9>=1.4?'r':p.vs_L_hr9>=1?'y':'g');
  h+=chip('HR/9 vs R',p.vs_R_hr9!=null?p.vs_R_hr9.toFixed(2):'--',p.vs_R_hr9>=1.4?'r':p.vs_R_hr9>=1?'y':'g');
  h+=chip('K% vs L',p.vs_L_k!=null?p.vs_L_k+'%':'--',p.vs_L_k>=28?'g':p.vs_L_k>=22?'y':'r');
  h+=chip('K% vs R',p.vs_R_k!=null?p.vs_R_k+'%':'--',p.vs_R_k>=28?'g':p.vs_R_k>=22?'y':'r');
  h+=chip('SLG vs L',p.vs_L_slg!=null?'.'+String(Math.round(p.vs_L_slg*1000)).padStart(3,'0'):'--','n');
  h+=chip('SLG vs R',p.vs_R_slg!=null?'.'+String(Math.round(p.vs_R_slg*1000)).padStart(3,'0'):'--','n');
  h+='</div></div>';
  if(p.top_pitches&&p.top_pitches.length){
    h+='<div class="detail-block"><div class="detail-block-title">Arsenal</div><div class="chip-row">';
    p.top_pitches.forEach(function(pt){h+=chip(pt.name,pt.usage+'%','n');});
    h+='</div></div>';
  }
  if(p.k_prop){
    h+='<div class="detail-block"><div class="detail-block-title">K Prop</div><div class="chip-row">';
    h+=chip('Line',p.k_prop.line.toFixed(1),'n');
    h+=chip('Odds',(p.k_prop.price>0?'+':'')+p.k_prop.price,'n');
    h+=chip('Book',p.k_prop.book||'--','n');
    h+='</div></div>';
  }
  h+='</div></div>';
  return h;
}

function buildGameTable(games){
  if(!games||!games.length){document.getElementById('game-main').innerHTML='<div class="no-data">No games today.</div>';return;}
  var h='<div class="section-hdr"><div class="section-title">Game Totals</div><div class="section-count">'+games.length+' games</div></div>';
  h+='<div class="tbl-wrap"><table>';
  h+='<thead><tr><th>Matchup</th><th>Time</th><th>Away Starter</th><th>Home Starter</th>';
  h+='<th class="num">Exp Runs Away</th><th class="num">Exp Runs Home</th><th class="num">Total</th>';
  h+='<th class="num">Exp HRs</th><th>Weather</th><th class="num">F5 Total</th></tr></thead><tbody>';
  games.forEach(function(g){
    var t=g.totals||{},wx=g.weather||{};
    var awayP=g.away_pitcher||{},homeP=g.home_pitcher||{};
    var total=t.total_exp_runs||0;
    var ouCls=total>=9.5?'ou-over':total<=7.5?'ou-under':'ou-push';
    var ouLbl=total>=9.5?'OVER':total<=7.5?'UNDER':'PUSH';
    var rpgA=t.away_runs_pg||4.5,rpgH=t.home_runs_pg||4.5;
    var eraA=awayP.era||4.20,eraH=homeP.era||4.20;
    var f5A=((eraH/4.20)*rpgA*0.5);
    var f5H=((eraA/4.20)*rpgH*0.5);
    var f5T=(f5A+f5H).toFixed(1);
    var f5Cls=parseFloat(f5T)>=5?'ou-over':parseFloat(f5T)<=3.5?'ou-under':'ou-push';
    h+='<tr>';
    h+='<td style="white-space:nowrap"><span class="team-a">'+g.away+'</span><span class="at-sep">@</span><span class="team-h">'+g.home+'</span></td>';
    h+='<td style="color:var(--muted);font-size:11px">'+fmtTime(g.time)+'</td>';
    h+='<td><div class="player-name">'+awayP.name+'</div><div class="player-sub">'+(awayP.hand||'R')+'HP &middot; ERA '+(awayP.era||'--')+' &middot; HR/9 '+(awayP.hr9||'--')+'</div></td>';
    h+='<td><div class="player-name">'+homeP.name+'</div><div class="player-sub">'+(homeP.hand||'R')+'HP &middot; ERA '+(homeP.era||'--')+' &middot; HR/9 '+(homeP.hr9||'--')+'</div></td>';
    h+='<td class="num"><span class="exp-runs-big team-a">'+(t.away_exp_runs||'--')+'</span></td>';
    h+='<td class="num"><span class="exp-runs-big team-h">'+(t.home_exp_runs||'--')+'</span></td>';
    h+='<td class="num"><span class="exp-runs-big '+ouCls+'">'+total+'</span><br><span style="font-size:9px;font-weight:700;letter-spacing:.06em" class="'+ouCls+'">'+ouLbl+'</span></td>';
    h+='<td class="num" style="color:var(--y);font-family:\'Barlow Condensed\',sans-serif;font-size:14px">'+(t.total_exp_hr||'--')+'</td>';
    h+='<td style="font-size:11px">'+wxIcon(wx.label,wx.temp,wx.wind_speed)+' '+(wx.temp||'--')+'&#176; <span style="color:var(--muted2)">'+(wx.wind_speed>0?wx.wind_speed+'mph':'calm')+'</span></td>';
    h+='<td class="num"><span class="exp-runs-big '+f5Cls+'">'+f5T+'</span></td>';
    h+='</tr>';
  });
  h+='</tbody></table></div>';
  document.getElementById('game-main').innerHTML=h;
}

async function searchPlayer(){
  var name=document.getElementById('searchInput').value.trim();
  if(!name)return;
  var el=document.getElementById('research-result');
  el.innerHTML='<div class="loading">Searching<div class="dots"><span></span><span></span><span></span></div></div>';
  try{
    var dateStr=formatDateForAPI(selectedDate);
    var resp=await fetch(API+'/research?player='+encodeURIComponent(name)+'&date='+dateStr);
    var data=await resp.json();
    if(data.error){el.innerHTML='<div class="err">'+data.error+'</div>';return;}
    var p=data.player||{},m=data.matchup||{};
    var bc=p.season_2026||{},b8d=p.last_8d||{};
    var h='<div class="research-result"><div class="r-header"><div class="r-name">'+p.name+'</div>';
    if(m.pitcher_name)h+='<div style="font-size:12px;color:var(--muted)">vs '+m.pitcher_name+' ('+m.pitcher_hand+'HP)</div>';
    h+='</div><div class="r-body">';
    h+='<div class="r-section"><div class="r-section-title">2026 Season ('+p.pa_2026+' PA)</div>';
    h+=rchip('Barrel%',bc.barrel_pct>0?bc.barrel_pct.toFixed(1)+'%':'--');
    h+=rchip('Launch Angle',bc.launch_angle>0?bc.launch_angle.toFixed(1)+'&#176;':'--');
    h+=rchip('ISO',bc.iso>0?fmtIso(bc.iso):'--');
    h+=rchip('Hard Hit%',bc.hard_hit_pct>0?bc.hard_hit_pct.toFixed(1)+'%':'--');
    h+=rchip('K%',bc.k_pct>0?bc.k_pct.toFixed(1)+'%':'--');
    h+=rchip('HR',String(bc.hr||0));
    h+='</div>';
    if(b8d&&b8d.pa>0){
      h+='<div class="r-section"><div class="r-section-title">Last 8 Days ('+b8d.pa+' PA)</div>';
      h+=rchip('Barrel%',b8d.barrel_pct>0?b8d.barrel_pct.toFixed(1)+'%':'--');
      h+=rchip('LA',b8d.launch_angle>0?b8d.launch_angle.toFixed(1)+'&#176;':'--');
      h+=rchip('ISO',b8d.iso>0?fmtIso(b8d.iso):'--');
      h+=rchip('HR',String(b8d.hr||0));
      h+='</div>';
    }
    var sp=p.splits||{};
    if(sp.vs_lhp||sp.vs_rhp){
      h+='<div class="r-section"><div class="r-section-title">Splits</div>';
      if(sp.vs_lhp&&sp.vs_lhp.pa>0){h+=rchip('ISO vs LHP',sp.vs_lhp.iso>0?fmtIso(sp.vs_lhp.iso):'--');h+=rchip('SLG vs LHP',sp.vs_lhp.slg>0?'.'+String(Math.round(sp.vs_lhp.slg*1000)).padStart(3,'0'):'--');}
      if(sp.vs_rhp&&sp.vs_rhp.pa>0){h+=rchip('ISO vs RHP',sp.vs_rhp.iso>0?fmtIso(sp.vs_rhp.iso):'--');h+=rchip('SLG vs RHP',sp.vs_rhp.slg>0?'.'+String(Math.round(sp.vs_rhp.slg*1000)).padStart(3,'0'):'--');}
      h+='</div>';
    }
    if(m.pitcher_stats){
      var ps=m.pitcher_stats;
      h+='<div class="r-section"><div class="r-section-title">Pitcher — '+m.pitcher_name+'</div>';
      h+=rchip('ERA',ps.era?ps.era.toFixed(2):'--');h+=rchip('HR/9',ps.hr9?ps.hr9.toFixed(2):'--');h+=rchip('HH%',ps.hard_hit_pct?ps.hard_hit_pct+'%':'--');
      h+='</div>';
    }
    h+='</div></div>';
    el.innerHTML=h;
  }catch(e){el.innerHTML='<div class="err">'+e.message+'</div>';}
}
function rchip(lbl,val){return '<div class="r-chip"><div class="r-chip-lbl">'+lbl+'</div><div class="r-chip-val">'+val+'</div></div>';}

async function load(){
  var btn=document.getElementById('refreshBtn'),upd=document.getElementById('upd');
  btn.disabled=true;
  document.getElementById('batter-main').innerHTML='<div class="loading">Loading Sharp Data<div class="dots"><span></span><span></span><span></span></div></div>';
  document.getElementById('pitcher-main').innerHTML='<div class="no-data">Loading...</div>';
  document.getElementById('game-main').innerHTML='<div class="no-data">Loading...</div>';
  try{
    var dateStr=formatDateForAPI(selectedDate);
    var resp=await fetch(API+'/games?date='+dateStr);
    if(!resp.ok)throw new Error('Backend error '+resp.status);
    var data=await resp.json();
    if(data.loading){document.getElementById('batter-main').innerHTML='<div class="err">'+(data.message||'Data loading — try again in 30 seconds.')+'</div>';btn.disabled=false;return;}
    allGames=data.games||[];
    if(!allGames.length){
      var nm='<div class="no-data">No MLB games on '+dateStr+'.</div>';
      ['batter-main','pitcher-main','game-main'].forEach(function(id){document.getElementById(id).innerHTML=nm;});
      btn.disabled=false;return;
    }
    buildBatterTable(allGames);
    buildPitcherTable(allGames);
    buildGameTable(allGames);
    upd.textContent='Updated '+new Date().toLocaleTimeString()+' \u00b7 '+dateStr;
  }catch(e){document.getElementById('batter-main').innerHTML='<div class="err"><strong>Error:</strong> '+e.message+'</div>';}
  btn.disabled=false;
}
document.getElementById('dateLbl').textContent=fmtDateLabel(selectedDate);
load();
</script>
</body>
</html>
