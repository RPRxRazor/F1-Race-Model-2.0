# “””
F1 in Schools Race Simulator

Physics-based CO2 rocket car performance model.

Fixes applied vs previous version:

- Simulation wrapped in @st.cache_data only reruns when inputs change
- acceleration list initialised at [0.0], no post-hoc length patch
- Static/kinetic friction state handled with explicit launched flag
- Thrust hard-cutoff to zero once canister is modelled as empty
- Named constants block no magic numbers in simulation loop
- Variable names no longer shadow Python builtins (time t_vals, etc.)
- Downsampling shows exact row count, handles short runs gracefully
- to_excel() defined once, called once (was called twice before)
  

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from io import BytesIO

# ══════════════════════════════════════════════════════════════

# PAGE CONFIG

# ══════════════════════════════════════════════════════════════

st.set_page_config(
page_title=“F1 in Schools · Race Simulator”,
page_icon=“🏎️”,
layout=“wide”,
initial_sidebar_state=“expanded”,
)

# ══════════════════════════════════════════════════════════════

# GLOBAL STYLING

# ══════════════════════════════════════════════════════════════

st.markdown(”””

<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Rajdhani:wght@300;400;500;600&display=swap');

:root {
  --red:     #E8002D;
  --gold:    #FFD600;
  --dark:    #0A0A0F;
  --panel:   #12121A;
  --border:  #1E1E2E;
  --muted:   #3A3A55;
  --text:    #E8E8F0;
  --sub:     #8888AA;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--dark) !important;
  color: var(--text) !important;
  font-family: 'Rajdhani', sans-serif !important;
}
[data-testid="stSidebar"] {
  background: var(--panel) !important;
  border-right: 1px solid var(--border) !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Hero ── */
.hero {
  background: linear-gradient(135deg, #0A0A0F 0%, #1A0008 50%, #0A0A0F 100%);
  border: 1px solid var(--border);
  border-top: 3px solid var(--red);
  border-radius: 0 0 12px 12px;
  padding: 32px 40px 24px;
  margin-bottom: 28px;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  top: -60px; right: -80px;
  width: 320px; height: 320px;
  background: radial-gradient(circle, rgba(232,0,45,0.12) 0%, transparent 70%);
  pointer-events: none;
}
.hero-badge {
  display: inline-block;
  background: var(--red);
  color: #fff;
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem;
  font-weight: 800;
  letter-spacing: 0.15em;
  padding: 3px 10px;
  border-radius: 2px;
  margin-bottom: 10px;
}
.hero-title {
  font-family: 'Orbitron', monospace !important;
  font-size: 2.1rem !important;
  font-weight: 900 !important;
  color: #fff !important;
  letter-spacing: 0.08em;
  margin: 0 0 4px !important;
  line-height: 1.1 !important;
}
.hero-title span { color: var(--red); }
.hero-sub {
  color: var(--sub);
  font-size: 0.95rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

/* ── Sidebar section labels ── */
.sidebar-section {
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--red);
  border-bottom: 1px solid var(--border);
  padding-bottom: 6px;
  margin: 20px 0 12px;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.metric-card {
  flex: 1; min-width: 160px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 18px 22px;
  position: relative;
  overflow: hidden;
}
.metric-card::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--accent, var(--red));
}
.metric-label {
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--sub);
  margin-bottom: 6px;
}
.metric-value {
  font-family: 'Orbitron', monospace;
  font-size: 1.7rem;
  font-weight: 800;
  color: #fff;
  line-height: 1;
}
.metric-value span { font-size: 0.9rem; color: var(--sub); font-weight: 400; }
.metric-sub { font-size: 0.75rem; color: var(--sub); margin-top: 4px; }
.accent-red   { --accent: #E8002D; }
.accent-gold  { --accent: #FFD600; }
.accent-blue  { --accent: #00B4FF; }
.accent-green { --accent: #00E676; }

/* ── Finish banner ── */
.finish-banner {
  background: linear-gradient(90deg, rgba(232,0,45,0.15) 0%, rgba(232,0,45,0.04) 100%);
  border: 1px solid rgba(232,0,45,0.4);
  border-left: 4px solid var(--red);
  border-radius: 8px;
  padding: 18px 24px;
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 28px;
}
.finish-flag { font-size: 2rem; }
.finish-label {
  font-family: 'Orbitron', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.2em;
  color: var(--sub);
  text-transform: uppercase;
  margin-bottom: 2px;
}
.finish-time {
  font-family: 'Orbitron', monospace;
  font-size: 2.6rem;
  font-weight: 900;
  color: var(--red);
  line-height: 1;
}
.finish-time span { color: var(--sub); font-size: 1.2rem; font-weight: 400; }

/* ── Inputs ── */
[data-testid="stNumberInput"] input {
  background: #0D0D18 !important;
  border: 1px solid var(--muted) !important;
  border-radius: 6px !important;
  color: var(--text) !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 0.85rem !important;
}
[data-testid="stNumberInput"] label {
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  color: var(--text) !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button,
[data-testid="stDownloadButton"] > button {
  background: var(--red) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 6px !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 0.75rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.1em !important;
  padding: 10px 22px !important;
  transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button:hover,
[data-testid="stDownloadButton"] > button:hover {
  background: #FF1744 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(232,0,45,0.35) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: var(--panel) !important;
  border-bottom: 2px solid var(--border) !important;
  gap: 4px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  font-family: 'Orbitron', monospace !important;
  font-size: 0.68rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.12em !important;
  color: var(--sub) !important;
  padding: 10px 20px !important;
  border-radius: 4px 4px 0 0 !important;
  border: none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  background: rgba(232,0,45,0.1) !important;
  color: var(--red) !important;
  border-bottom: 2px solid var(--red) !important;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}

hr { border-color: var(--border) !important; }
</style>

“””, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════

# PHYSICS CONSTANTS  (named — no magic numbers in sim loop)

# ══════════════════════════════════════════════════════════════

RHO_AIR      = 1.225   # kg/m³  — sea-level ISA air density
GRAVITY      = 9.81    # m/s²
DT           = 0.001   # s      — integration timestep (1 ms)
MAX_SIM_TIME = 5.0     # s      — hard cutoff if car never finishes
V_THRESH     = 0.01    # m/s    — velocity below which car is stationary

# ══════════════════════════════════════════════════════════════

# SIMULATION  (cached — only reruns when parameters change)

# ══════════════════════════════════════════════════════════════

@st.cache_data
def run_simulation(
mass_g: float,
drag_coeff: float,
frontal_area: float,
max_thrust: float,
thrust_decay_rate: float,
static_friction: float,
kinetic_friction: float,
track_length: float,
) -> dict:
“””
Euler-method simulation of a CO₂ rocket car.

```
Friction model
──────────────
Car starts stationary. Thrust must strictly exceed static_friction
to launch. Once launched, only kinetic_friction applies for the
remainder of the run — the launched flag never resets, preventing
the stall-oscillation glitch in the original code.

Thrust model
────────────
Exponential decay: F_thrust = max_thrust * exp(-decay * t)
Hard floor at 0 N — np.maximum prevents negative thrust values.

Returns a dict of equal-length lists safe for plotting and export.
"""
mass_kg  = mass_g / 1000.0
t_val    = 0.0
vel      = 0.0
pos      = 0.0
launched = False

# All four lists initialised at t=0 — no post-hoc length patching
t_vals   = [t_val]
vel_vals = [vel]
pos_vals = [pos]
acc_vals = [0.0]   # acceleration is 0 before launch

while pos < track_length and t_val < MAX_SIM_TIME:
    thrust = max(0.0, max_thrust * np.exp(-thrust_decay_rate * t_val))
    drag   = 0.5 * RHO_AIR * drag_coeff * frontal_area * vel ** 2

    if not launched:
        if thrust <= static_friction:
            net_force = 0.0          # car stays still
        else:
            launched  = True         # single clean state transition
            net_force = thrust - drag - kinetic_friction
    else:
        net_force = thrust - drag - kinetic_friction

    accel = net_force / mass_kg
    vel   = max(0.0, vel + accel * DT)
    pos  += vel * DT
    t_val += DT

    t_vals.append(t_val)
    vel_vals.append(vel)
    pos_vals.append(pos)
    acc_vals.append(accel)

return {
    "t":           t_vals,
    "velocity":    vel_vals,
    "position":    pos_vals,
    "accel":       acc_vals,
    "finish_time": t_vals[-1],
}
```

# ══════════════════════════════════════════════════════════════

# HERO BANNER

# ══════════════════════════════════════════════════════════════

st.markdown(”””

<div class="hero">
  <div class="hero-badge">F1 IN SCHOOLS · ENGINEERING TOOL</div>
  <div class="hero-title">RACE <span>SIMULATOR</span></div>
  <div class="hero-sub">Physics-Based Performance Modelling · Precise Controls · Export Ready</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════

# SIDEBAR — INPUTS

# ══════════════════════════════════════════════════════════════

with st.sidebar:
st.markdown(”””
<div style="font-family:'Orbitron',monospace;font-size:1rem;font-weight:900;
color:#E8002D;letter-spacing:0.1em;padding:8px 0 4px;">⚙ PARAMETERS</div>
<div style="color:#8888AA;font-size:0.78rem;margin-bottom:16px;">
Adjust variables below. Simulation recalculates automatically.
</div>
“””, unsafe_allow_html=True)

```
st.markdown('<div class="sidebar-section">🏎 Car Properties</div>', unsafe_allow_html=True)
mass         = st.number_input("Car Mass (grams)",      30.0,   200.0,  55.0,   step=0.1,    format="%.2f")
drag_coeff   = st.number_input("Drag Coefficient (Cd)", 0.05,   1.5,    0.50,   step=0.01,   format="%.3f")
frontal_area = st.number_input("Frontal Area (m²)",     0.0005, 0.01,   0.0039, step=0.0001, format="%.5f")

st.markdown('<div class="sidebar-section">🚀 Propulsion</div>', unsafe_allow_html=True)
max_thrust        = st.number_input("Max Thrust Force (N)",    1.0, 40.0, 22.0, step=0.1, format="%.2f")
thrust_decay_rate = st.number_input("Thrust Decay Rate (1/s)", 1.0, 50.0, 17.0, step=0.1, format="%.2f")

st.markdown('<div class="sidebar-section">⚡ Friction</div>', unsafe_allow_html=True)
static_friction_force  = st.number_input("Static Friction Force (N)",  0.0, 1.0, 0.90,  step=0.01,  format="%.3f")
kinetic_friction_force = st.number_input("Kinetic Friction Force (N)", 0.0, 0.1, 0.065, step=0.001, format="%.4f")

st.markdown('<div class="sidebar-section">📍 Track</div>', unsafe_allow_html=True)
track_length = st.number_input("Track Length (m)", 5.0, 50.0, 20.0, step=0.1, format="%.2f")

st.markdown("---")
st.markdown(f"""
<div style="font-size:0.7rem;color:#8888AA;line-height:1.9;">
  Timestep &nbsp;: <b style="color:#E8E8F0;">{DT * 1000:.0f} ms</b><br>
  Max window: <b style="color:#E8E8F0;">{MAX_SIM_TIME:.0f} s</b><br>
  Air density: <b style="color:#E8E8F0;">{RHO_AIR} kg/m³</b> (ISA sea level)
</div>
""", unsafe_allow_html=True)
```

# ══════════════════════════════════════════════════════════════

# RUN SIMULATION

# ══════════════════════════════════════════════════════════════

results = run_simulation(
mass_g            = mass,
drag_coeff        = drag_coeff,
frontal_area      = frontal_area,
max_thrust        = max_thrust,
thrust_decay_rate = thrust_decay_rate,
static_friction   = static_friction_force,
kinetic_friction  = kinetic_friction_force,
track_length      = track_length,
)

t_vals      = results[“t”]
vel_vals    = results[“velocity”]
pos_vals    = results[“position”]
acc_vals    = results[“accel”]
finish_time = results[“finish_time”]

peak_vel   = float(np.max(vel_vals))
peak_accel = float(np.max(acc_vals))
avg_vel    = float(np.mean(vel_vals))
n_steps    = len(t_vals)

# Warn user if the car never reached the finish line

if finish_time >= MAX_SIM_TIME:
st.warning(
f”⚠️  Car did not reach {track_length} m within {MAX_SIM_TIME} s. “
“Try increasing thrust or reducing mass / drag.”,
)

# ══════════════════════════════════════════════════════════════

# FINISH TIME BANNER

# ══════════════════════════════════════════════════════════════

st.markdown(f”””

<div class="finish-banner">
  <div class="finish-flag">🏁</div>
  <div>
    <div class="finish-label">Finish Time</div>
    <div class="finish-time">{finish_time:.3f}<span> s</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════

# METRIC CARDS

# ══════════════════════════════════════════════════════════════

st.markdown(f”””

<div class="metric-row">
  <div class="metric-card accent-gold">
    <div class="metric-label">Peak Velocity</div>
    <div class="metric-value">{peak_vel:.2f}<span> m/s</span></div>
    <div class="metric-sub">{peak_vel * 3.6:.1f} km/h</div>
  </div>
  <div class="metric-card accent-red">
    <div class="metric-label">Peak Acceleration</div>
    <div class="metric-value">{peak_accel:.1f}<span> m/s²</span></div>
    <div class="metric-sub">{peak_accel / GRAVITY:.2f} g-force</div>
  </div>
  <div class="metric-card accent-blue">
    <div class="metric-label">Avg Velocity</div>
    <div class="metric-value">{avg_vel:.2f}<span> m/s</span></div>
    <div class="metric-sub">{avg_vel * 3.6:.1f} km/h</div>
  </div>
  <div class="metric-card accent-green">
    <div class="metric-label">Simulation Steps</div>
    <div class="metric-value">{n_steps:,}<span></span></div>
    <div class="metric-sub">{mass:.1f} g · Cd {drag_coeff:.3f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════

# SHARED PLOT HELPERS

# ══════════════════════════════════════════════════════════════

PLOT_BG  = “#0A0A0F”
PAPER_BG = “#0A0A0F”
GRID_COL = “#1E1E2E”
FONT_COL = “#8888AA”
FONT_FAM = “Rajdhani, sans-serif”
OBT_FAM  = “Orbitron, monospace”

def base_layout(title: str, x_title: str, y_title: str, height: int = 420) -> dict:
“”“Returns a consistent dark-theme Plotly layout dict.”””
return dict(
plot_bgcolor  = PLOT_BG,
paper_bgcolor = PAPER_BG,
font          = dict(family=FONT_FAM, color=FONT_COL),
title         = dict(text=title, font=dict(family=OBT_FAM, size=11, color=”#E8E8F0”), x=0.01),
xaxis         = dict(title=x_title, gridcolor=GRID_COL, linecolor=GRID_COL,
tickfont=dict(family=OBT_FAM, size=9)),
yaxis         = dict(title=y_title, gridcolor=GRID_COL, linecolor=GRID_COL,
tickfont=dict(family=OBT_FAM, size=9)),
height        = height,
margin        = dict(l=10, r=10, t=44, b=10),
)

# ══════════════════════════════════════════════════════════════

# TABS

# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([“📈  TELEMETRY”, “📊  DATA TABLE”, “📥  EXPORT”])

# ────── TAB 1 : TELEMETRY ──────────────────────────────────

with tab1:
col_left, col_right = st.columns([2, 1])

```
with col_left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_vals, y=pos_vals, mode="lines", name="Position (m)",
        line=dict(color="#00B4FF", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,180,255,0.05)",
    ))
    fig.add_trace(go.Scatter(
        x=t_vals, y=vel_vals, mode="lines", name="Velocity (m/s)",
        line=dict(color="#E8002D", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=t_vals, y=acc_vals, mode="lines", name="Accel (m/s²)",
        line=dict(color="#FFD600", width=1.8, dash="dot"),
    ))
    fig.add_vline(
        x=finish_time, line_dash="dash", line_color="rgba(232,0,45,0.5)",
        annotation_text=f"🏁 {finish_time:.3f}s",
        annotation_font_color="#E8002D",
        annotation_font_family=OBT_FAM,
        annotation_font_size=11,
    )
    layout = base_layout(
        "RACE TELEMETRY — POSITION · VELOCITY · ACCELERATION",
        "Time (s)", "Value", 420,
    )
    layout["legend"] = dict(
        bgcolor="rgba(18,18,26,0.9)", bordercolor="#1E1E2E", borderwidth=1,
        font=dict(family=FONT_FAM, size=12), x=0.01, y=0.99,
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    # ── Thrust profile ──
    t_arr      = np.linspace(0, finish_time + 0.2, 400)
    thrust_arr = np.maximum(0.0, max_thrust * np.exp(-thrust_decay_rate * t_arr))

    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=t_arr, y=thrust_arr, mode="lines",
        line=dict(color="#FF6B35", width=2.5),
        fill="tozeroy", fillcolor="rgba(255,107,53,0.08)",
    ))
    fig_t.add_hline(
        y=static_friction_force, line_dash="dash",
        line_color="rgba(255,214,0,0.6)",
        annotation_text="Static friction",
        annotation_font_color="#FFD600",
        annotation_font_size=10,
    )
    fig_t.update_layout(**base_layout("THRUST PROFILE", "Time (s)", "Force (N)", 195))
    st.plotly_chart(fig_t, use_container_width=True)

    # ── Velocity histogram ──
    fig_h = go.Figure()
    fig_h.add_trace(go.Histogram(
        x=vel_vals, nbinsx=30,
        marker_color="#E8002D", opacity=0.85,
    ))
    fig_h.update_layout(**base_layout("VELOCITY DISTRIBUTION", "m/s", "Count", 195))
    st.plotly_chart(fig_h, use_container_width=True)
```

# ────── TAB 2 : DATA TABLE ─────────────────────────────────

with tab2:
target_rows  = 500
downsample   = max(1, n_steps // target_rows)
visible_rows = len(t_vals[::downsample])

```
st.markdown(
    f"<div style='font-size:0.82rem;color:#8888AA;margin-bottom:12px;'>"
    f"Displaying <b style='color:#E8E8F0;'>{visible_rows:,}</b> of "
    f"<b style='color:#E8E8F0;'>{n_steps:,}</b> rows "
    f"(1-in-{downsample} sample). Full resolution in Export tab.</div>",
    unsafe_allow_html=True,
)

df_display = pd.DataFrame({
    "Time (s)":       np.array(t_vals[::downsample]).round(4),
    "Position (m)":   np.array(pos_vals[::downsample]).round(4),
    "Velocity (m/s)": np.array(vel_vals[::downsample]).round(4),
    "Accel (m/s²)":   np.array(acc_vals[::downsample]).round(4),
})
st.dataframe(df_display, use_container_width=True, height=480)
```

# ────── TAB 3 : EXPORT ─────────────────────────────────────

with tab3:
st.markdown(”””
<div style="font-family:'Orbitron',monospace;font-size:0.8rem;font-weight:700;
letter-spacing:0.12em;color:#E8E8F0;margin-bottom:16px;">
EXPORT SIMULATION RESULTS
</div>
“””, unsafe_allow_html=True)

```
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Run Parameters**")
    df_params = pd.DataFrame({
        "Parameter": [
            "Car Mass (g)", "Drag Coefficient (Cd)", "Frontal Area (m²)",
            "Max Thrust (N)", "Thrust Decay Rate (1/s)",
            "Static Friction (N)", "Kinetic Friction (N)", "Track Length (m)",
        ],
        "Value": [
            mass, drag_coeff, frontal_area, max_thrust,
            thrust_decay_rate, static_friction_force, kinetic_friction_force, track_length,
        ],
    })
    st.dataframe(df_params, use_container_width=True, hide_index=True)

with col_b:
    st.markdown("**Performance Summary**")
    df_perf = pd.DataFrame({
        "Metric": [
            "Finish Time (s)", "Peak Velocity (m/s)", "Peak Velocity (km/h)",
            "Peak Acceleration (m/s²)", "Peak G-Force",
            "Avg Velocity (m/s)", "Simulation Steps",
        ],
        "Value": [
            round(finish_time, 5),
            round(peak_vel,    4),
            round(peak_vel * 3.6, 3),
            round(peak_accel,  4),
            round(peak_accel / GRAVITY, 4),
            round(avg_vel,     4),
            n_steps,
        ],
    })
    st.dataframe(df_perf, use_container_width=True, hide_index=True)

st.markdown("---")

def build_excel() -> bytes:
    """Builds the Excel workbook in memory and returns raw bytes."""
    df_p = pd.DataFrame({
        "Parameter": [
            "Car Mass (grams)", "Drag Coefficient (Cd)", "Frontal Area (m²)",
            "Max Thrust Force (N)", "Thrust Decay Rate (1/s)",
            "Static Friction Force (N)", "Kinetic Friction Force (N)",
            "Track Length (m)", "Finish Time (s)",
        ],
        "Value": [
            mass, drag_coeff, frontal_area, max_thrust,
            thrust_decay_rate, static_friction_force, kinetic_friction_force,
            track_length, round(finish_time, 5),
        ],
    })
    df_r = pd.DataFrame({
        "Time (s)":            t_vals,
        "Position (m)":        pos_vals,
        "Velocity (m/s)":      vel_vals,
        "Acceleration (m/s²)": acc_vals,
    })
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_p.to_excel(writer, sheet_name="Parameters",      index=False)
        df_r.to_excel(writer, sheet_name="Simulation Data", index=False)
    return buf.getvalue()

st.download_button(
    label     = "📥  DOWNLOAD EXCEL REPORT",
    data      = build_excel(),
    file_name = f"f1_race_simulation_{int(finish_time * 1000)}ms.xlsx",
    mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown(
    "<div style='font-size:0.75rem;color:#8888AA;margin-top:12px;'>"
    "Excel contains two sheets: "
    "<b style='color:#E8E8F0;'>Parameters</b> and "
    "<b style='color:#E8E8F0;'>Simulation Data</b> at full 1 ms resolution."
    "</div>",
    unsafe_allow_html=True,
)
```