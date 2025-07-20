import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="F1 Race Simulator Precise", layout="wide")
st.title("🏎️ F1 in Schools Race Simulator (Precise Controls + Export)")

# Inputs
mass = st.number_input("Car Mass (grams)", 30.0, 200.0, 55.0, step=0.1, format="%.2f")
drag_coeff = st.number_input("Drag Coefficient (Cd)", 0.05, 1.5, 0.50, step=0.01, format="%.3f")
frontal_area = st.number_input("Frontal Area (m²)", 0.0005, 0.01, 0.0039, step=0.0001, format="%.5f")
max_thrust = st.number_input("Max Thrust Force (N)", 1.0, 40.0, 22.0, step=0.1, format="%.2f")
thrust_decay_rate = st.number_input("Thrust Decay Rate (1/s)", 1.0, 50.0, 17.0, step=0.1, format="%.2f")
static_friction_force = st.number_input("Static Friction Force (N)", 0.0, 1.0, 0.90, step=0.01, format="%.3f")
kinetic_friction_force = st.number_input("Kinetic Friction Force (N)", 0.0, 0.1, 0.065, step=0.001, format="%.4f")
track_length = st.number_input("Track Length (m)", 5.0, 50.0, 20.0, step=0.1, format="%.2f")

# Constants
mass_kg = mass / 1000
rho = 1.225
g = 9.81
dt = 0.001
max_time = 5.0

# Simulation
t = 0.0
v = 0.0
x = 0.0

time = [t]
velocity = [v]
position = [x]
acceleration = []

while x < track_length and t < max_time:
    thrust = max_thrust * np.exp(-thrust_decay_rate * t)

    drag_force = 0.5 * rho * drag_coeff * frontal_area * v**2

    if v < 0.01:
        friction_force = static_friction_force
        if thrust < static_friction_force:
            net_force = 0.0
        else:
            net_force = thrust - drag_force - kinetic_friction_force
    else:
        friction_force = kinetic_friction_force
        net_force = thrust - drag_force - friction_force

    a = net_force / mass_kg

    v += a * dt
    if v < 0:
        v = 0.0
    x += v * dt
    t += dt

    time.append(t)
    velocity.append(v)
    position.append(x)
    acceleration.append(a)

acceleration.insert(0, acceleration[0])

finish_time = time[-1]
st.success(f"🏁 Finish Time: **{finish_time:.3f} seconds**")

fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=position, mode='lines', name='Position (m)', line=dict(color='royalblue')))
fig.add_trace(go.Scatter(x=time, y=velocity, mode='lines', name='Velocity (m/s)', line=dict(color='firebrick')))
fig.add_trace(go.Scatter(x=time, y=acceleration, mode='lines', name='Acceleration (m/s²)', line=dict(color='green', dash='dot')))

fig.update_layout(
    title="📈 Race Simulation (Precise Controls)",
    xaxis_title="Time (s)",
    yaxis_title="Value",
    height=550,
    legend=dict(x=0.01, y=0.99)
)

st.plotly_chart(fig, use_container_width=True)

# Export to Excel
def to_excel():
    params = {
        "Parameter": [
            "Car Mass (grams)", "Drag Coefficient (Cd)", "Frontal Area (m²)", "Max Thrust Force (N)",
            "Thrust Decay Rate (1/s)", "Static Friction Force (N)", "Kinetic Friction Force (N)", "Track Length (m)",
            "Finish Time (s)"
        ],
        "Value": [
            mass, drag_coeff, frontal_area, max_thrust,
            thrust_decay_rate, static_friction_force, kinetic_friction_force, track_length,
            round(finish_time, 5)
        ]
    }
    df_params = pd.DataFrame(params)

    df_results = pd.DataFrame({
        "Time (s)": time,
        "Position (m)": position,
        "Velocity (m/s)": velocity,
        "Acceleration (m/s²)": acceleration
    })

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_params.to_excel(writer, sheet_name='Parameters', index=False)
        df_results.to_excel(writer, sheet_name='Simulation Data', index=False)

    return output.getvalue()

if st.button("📥 Export Results to Excel"):
    excel_data = to_excel()
    st.download_button(
        label="Download Excel file",
        data=excel_data,
        file_name=f"f1_race_simulation_results_{int(finish_time*1000)}ms.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
