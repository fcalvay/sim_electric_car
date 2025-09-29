#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 11:13:45 2025

@author: florentcalvayrac
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# =========================
# Constantes & helpers
# =========================
RHO_AIR = 1.225
G = 9.81
T_REF = 25.0
KMH2MS = 1/3.6

st.set_page_config(page_title="Simulateur VE — batteries, moteur, pneus", layout="wide")
st.title("🚗⚡ Simulateur VE — batteries, moteur, pneus (léger → poids lourd)")

# ---- OCV(SOC) (approx Li-ion NMC/NCA) ----
SOC_TAB = np.array([0.00, 0.10, 0.20, 0.50, 0.80, 0.90, 1.00])
OCV_TAB = np.array([3.00, 3.40, 3.55, 3.70, 3.90, 4.00, 4.15])
def ocv_cell(soc, U_nom):
    v = np.interp(np.clip(soc, 0, 1), SOC_TAB, OCV_TAB)
    v += (U_nom - np.interp(0.5, SOC_TAB, OCV_TAB))
    return v

def r_cell_multiplier_soc(soc):
    x = np.abs(np.clip(soc, 0, 1) - 0.5) / 0.5
    return 1.0 + 0.4*(x**2)

def r_cell_multiplier_temp(temp_c):
    beta = 0.015
    return float(np.exp(beta*(T_REF - temp_c)))

def pack_params(n_s, n_p, U_cell_nom, C_cell_Ah, R_cell_mohm_25C, m_cell, soc, temp_c):
    U_cell_ocv = ocv_cell(soc, U_cell_nom)
    R_cell_eff = (R_cell_mohm_25C*1e-3) * r_cell_multiplier_soc(soc) * r_cell_multiplier_temp(temp_c)
    U_pack = n_s * U_cell_ocv
    C_Ah   = n_p * C_cell_Ah
    E_kWh  = U_pack * C_Ah / 1000.0
    R_pack = n_s * R_cell_eff / n_p
    m_pack = n_s * n_p * m_cell
    return U_pack, C_Ah, E_kWh, R_pack, m_pack, U_cell_ocv, R_cell_eff

# ---- Rendements (pédago) ----
def eta_motor_var(P_mech, P_mech_max):
    if P_mech_max <= 0: return 0.9
    x = np.clip(P_mech / P_mech_max, 0.0, 1.0)
    return float(np.clip(0.70 + 0.30 * (1 - (x-0.7)**2), 0.70, 0.97))
def eta_inverter_var(P_elec):
    x = np.tanh(P_elec / 10000.0)
    return float(np.clip(0.94 + 0.04 * x, 0.92, 0.985))

# ---- P_max arbre via Thévenin (pack+Rmoteur) ----
def max_shaft_power_from_thevenin(U_ocv, R_pack, R_motor, eta_motor_peak=0.94):
    R_tot = max(R_pack + R_motor, 1e-9)
    P_shaft = eta_motor_peak * U_ocv**2 / (4.0 * R_tot)
    I_opt   = U_ocv / (2.0 * R_tot)
    return P_shaft, I_opt

# ---- Puissances & efforts véhicule ----
def power_required(v, m, Cx, S, Crr):
    P_drag = 0.5 * RHO_AIR * Cx * S * v**3
    P_roll = m * G * Crr * v
    return P_drag + P_roll

def vmax_from_power(P_shaft_max, m, Cx, S, Crr):
    v = 30.0
    for _ in range(80):
        f  = power_required(v, m, Cx, S, Crr) - P_shaft_max
        df = 1.5*RHO_AIR*Cx*S*v*v + m*G*Crr
        v  = max(v - f/max(df,1e-9), 0.0)
    return v

# =========================
# 0–100 robuste (forces)
# =========================
def integrate_0_100_advanced(
    P_shaft_peak, T_motor_max, omega_base, gear_ratio, r_wheel,
    m_total, Cx, S, Crr,
    a_max_g=1.0, dt=0.02,
    soc0=0.6, C_Ah_pack=200.0,
    n_s=200, n_p=60,
    U_cell_nom=3.65, R_cell_mohm_25C=12.0, m_cell=0.07, temp_c=25.0
):
    v_target = 27.78
    rho_air  = 1.225
    g        = 9.81
    eps_v    = 0.10

    v = 0.0; t = 0.0
    soc = float(np.clip(soc0, 0.01, 1.0))
    Q   = max(C_Ah_pack, 1e-9) * soc
    U_pack_nom = max(1.0, n_s * U_cell_nom)

    ts, vs, Us, Is, SOCs, Pmecs = [], [], [], [], [], []
    F_roll_const = m_total * g * Crr
    last_v = 0.0; stall_time = 0.0

    while v < v_target and t < 200.0:
        omega_wheel = v / max(r_wheel, 1e-6)
        omega_motor = omega_wheel * gear_ratio
        # Transition naturelle : T = min(Tmax, P/ω)
        Tm = min(T_motor_max, P_shaft_peak / max(omega_motor, 1e-6))

        T_wheel  = Tm * gear_ratio
        F_torque = T_wheel / max(r_wheel, 1e-6)
        F_power  = P_shaft_peak / max(v, eps_v)

        # Adhérence pneus (globale via var globale MU_TYRE_EFFECTIVE)
        try:
            F_grip = MU_TYRE_EFFECTIVE * m_total * g
        except NameError:
            F_grip = float('inf')

        F_trac = max(0.0, min(F_torque, F_power, F_grip))
        F_aero = 0.5 * rho_air * Cx * S * v*v
        F_loss = F_roll_const + F_aero

        F_net = max(0.0, F_trac - F_loss)
        a     = min(F_net / max(m_total,1e-9), a_max_g * g)

        P_mech = F_trac * max(v, eps_v)
        eta_m   = eta_motor_var(P_mech, max(P_shaft_peak, 1.0))
        P_elec  = P_mech / max(eta_m, 1e-6)
        eta_inv = eta_inverter_var(P_elec)
        P_dc    = P_elec / max(eta_inv, 1e-6)

        I = P_dc / U_pack_nom
        U_load = U_pack_nom

        Q  = max(0.0, Q - I * dt / 3600.0)
        soc = Q / max(C_Ah_pack, 1e-9)

        v += a * dt
        t += dt

        ts.append(t); vs.append(v); Us.append(U_load); Is.append(I); SOCs.append(soc); Pmecs.append(P_mech)

        if soc <= 0.02: break
        if v - last_v < 1e-4:
            stall_time += dt
            if stall_time > 1.5: break
        else:
            stall_time = 0.0; last_v = v

    t100 = t if v >= v_target else np.inf
    return t100, {"t":np.array(ts),"v":np.array(vs),"U":np.array(Us),
                  "I":np.array(Is),"SOC":np.array(SOCs),"Pmech":np.array(Pmecs)}

# =========================
# Courant de croisière avec R_moteur
# =========================
def cruise_current_total(U_pack, R_pack, R_motor, P_mech_req, P_aux_W, eta_core):
    """
    Résout: I*(U_pack - I*(R_pack + R_motor)) = P_mech_req/eta_core + P_aux_W
    """
    R_tot = max(R_pack + R_motor, 1e-12)
    U = max(U_pack, 1e-9)
    C = (P_mech_req / max(eta_core, 1e-6)) + max(P_aux_W, 0.0)
    disc = U*U - 4.0*R_tot*C
    if disc <= 0:
        return np.nan
    return (U - np.sqrt(disc)) / (2.0*R_tot)

def peukert_time_hours(C_Ah_pack, I_pack, k=1.03):
    if not np.isfinite(I_pack) or I_pack <= 1e-9:
        return np.nan
    return (C_Ah_pack**k) / (I_pack**k)

# =========================
# UI
# =========================
with st.sidebar:
    st.header("Cellules (préréglages)")
    preset = st.selectbox("Type de cellule", ["21700 réaliste", "18650 réaliste", "Personnalisé"])
    if preset == "21700 réaliste":
        U_cell_nom = 3.65; C_cell = 5.0; R_cell_mohm_25C = 12.0; m_cell = 0.070
    elif preset == "18650 réaliste":
        U_cell_nom = 3.60; C_cell = 3.0; R_cell_mohm_25C = 18.0; m_cell = 0.045
    else:
        U_cell_nom = st.slider("U_cell nominal (V)", 3.2, 4.2, 3.65, 0.01)
        C_cell     = st.slider("C_cell (Ah)", 2.0, 10.0, 5.0, 0.1)
        R_cell_mohm_25C = st.slider("R_cell @25°C (mΩ)", 2.0, 30.0, 12.0, 0.5)
        m_cell     = st.slider("m_cell (kg)", 0.035, 0.12, 0.070, 0.001)

    st.header("Architecture pack")
    n_s = st.slider("Cellules en série (n_s)", 50, 220, 200, 1)
    n_p = st.slider("Branches en parallèle (n_p)", 1, 600, 60, 1)
    soc_perf = st.slider("SOC pour perfs (%)", 10, 100, 60, 1) / 100.0
    temp_c   = st.slider("Température batterie (°C)", -20, 50, 25, 1)
    interconn_pct = st.slider("Interconnexions (% masse batterie)", 0, 20, 6, 1) / 100.0

    st.header("Chaîne de traction / moteur")
    PkW_slider = st.slider("Puissance moteur (kW)", 50, 1000, 300, 10)
    rpm_base   = st.slider("Vitesse base moteur (tr/min)", 3000, 15000, 6500, 100)
    gear_ratio = st.slider("Réducteur (i)", 5.0, 20.0, 9.0, 0.1)
    r_wheel    = st.slider("Rayon de roue (m)", 0.28, 0.60, 0.33, 0.005)
    use_torque_cap = st.checkbox("Plafonner le couple par densité (T ≤ densité×kW)", value=True)
    torque_density = st.slider("Couple par kW (N·m/kW)", 2.0, 8.0, 3.5, 0.1)
    a_max_g    = st.slider("Accélération max (g)", 0.5, 1.0, 1.0, 0.05)
    eta_motor_peak = st.slider("η_m (pic, pour P_max Thévenin)", 0.88, 0.97, 0.94, 0.005)

    st.header("Véhicule & environnement (léger → poids lourd)")
    m_body = st.slider("Masse carrosserie/châssis (kg)", 800, 38000, 1000, 50)
    Cx     = st.slider("Cx", 0.18, 1.00, 0.24, 0.01)
    S      = st.slider("Surface frontale S (m²)", 1.8, 12.0, 2.2, 0.1)
    Crr    = st.slider("Coeff. roulement Crr (base)", 0.006, 0.020, 0.011, 0.001)

    st.header("Pneus & motricité")
    tire_width_mm = st.slider("Largeur pneus (mm)", 155, 445, 235, 5)
    driven_wheels = st.selectbox("Roues motrices", ["2", "4"])
    drive_axle_share = st.slider("Part de charge sur essieu moteur", 0.40, 0.75, 0.55, 0.01)
    mu0_base = st.slider("µ de base route", 0.60, 1.20, 1.00, 0.01)

    st.header("Accessoires & autonomie")
    P_aux_kW = st.slider("Accessoires (kW)", 0.0, 5.0, 0.8, 0.1)
    v_cruise_kmh = st.slider("Vitesse croisière (km/h)", 30, 130, 90, 5)
    soc_start = st.slider("SOC départ autonomie (%)", 20, 100, 90, 1) / 100.0
    k_peukert = st.slider("Exposant Peukert k", 1.00, 1.10, 1.03, 0.01)

# =========================
# Calculs pack, moteur, pneus
# =========================
U_pack, C_Ah, E_kWh, R_pack, m_pack, U_cell_ocv, R_cell_eff = pack_params(
    n_s, n_p, U_cell_nom, C_cell, R_cell_mohm_25C, m_cell, soc_perf, temp_c
)
m_inter = m_pack * interconn_pct

# Moteur : masse & R_motor diminuent avec puissance nominale
def motor_scaling(P_kW, m0=20.0, alpha=0.45, R0=0.03, Pref_kW=50.0, beta=0.7):
    P_kW = max(P_kW, 5.0)
    m_motor = m0 + alpha * P_kW
    R_motor = R0 * (Pref_kW / P_kW)**beta
    return m_motor, float(np.clip(R_motor, 0.003, 0.08))

m_motor, R_motor = motor_scaling(PkW_slider)

# Capacité pack vs tirette puissance
P_pack_raw, _ = max_shaft_power_from_thevenin(U_pack, R_pack, R_motor, eta_motor_peak)
P_cap_by_slider = PkW_slider * 1000.0
P_shaft_max = max(0.0, min(P_pack_raw, P_cap_by_slider))

# RPM base agit : T_max = P / ω_base (plafond optionnel par densité)
omega_base = (rpm_base/60.0) * 2*np.pi
T_motor_max = P_shaft_max / max(omega_base, 1e-6)
if use_torque_cap:
    T_motor_max = min(T_motor_max, torque_density * PkW_slider)

# Masses totales
m_total = m_body + m_pack + m_inter + m_motor

# Pneus : µ effectif et Crr_eff (largeur & rayon)
w_ref = 205.0; Fz_ref = 4000.0
beta_width = 0.15; alpha_load = 0.06
n_drive = 4 if driven_wheels == "4" else 2
Fz_driven = m_total * G * drive_axle_share
Fz_per_wheel = max(Fz_driven / n_drive, 1.0)
mu_width = (tire_width_mm / w_ref) ** beta_width
mu_load  = (Fz_per_wheel / Fz_ref) ** (-alpha_load)
MU_TYRE_EFFECTIVE = float(np.clip(mu0_base * mu_width * mu_load, 0.60, 1.30))

k_rr_width  = 0.15
k_rr_radius = 0.20
r_ref = 0.33
Crr_eff = Crr * (1.0 + k_rr_width * ((tire_width_mm / w_ref) - 1.0)) * (r_ref / max(r_wheel,1e-6))**k_rr_radius
Crr_eff = float(max(0.003, Crr_eff))

# =========================
# 0–100 & vitesse max
# =========================
t_0_100, traces_0100 = integrate_0_100_advanced(
    P_shaft_peak=P_shaft_max,
    T_motor_max=T_motor_max,
    omega_base=omega_base,
    gear_ratio=gear_ratio,
    r_wheel=r_wheel,
    m_total=m_total,
    Cx=Cx, S=S, Crr=Crr_eff,
    a_max_g=a_max_g, dt=0.02,
    soc0=soc_perf, C_Ah_pack=C_Ah,
    n_s=n_s, n_p=n_p, U_cell_nom=U_cell_nom, R_cell_mohm_25C=R_cell_mohm_25C,
    m_cell=m_cell, temp_c=temp_c
)
vmax_ms = vmax_from_power(P_shaft_max, m_total, Cx, S, Crr_eff)
vmax_kmh = vmax_ms*3.6

# =========================
# Autonomie à vitesse constante (avec R_moteur dans I)
# =========================
v_cruise = v_cruise_kmh*KMH2MS
P_mech_req = power_required(v_cruise, m_total, Cx, S, Crr_eff)
eta_m_mid, eta_inv_mid = 0.92, 0.96
eta_core = eta_m_mid * eta_inv_mid
I_cruise = cruise_current_total(U_pack, R_pack, R_motor, P_mech_req, P_aux_kW*1000.0, eta_core)
t_hours = peukert_time_hours(C_Ah, I_cruise, k_peukert)
autonomy_km = v_cruise_kmh * t_hours if np.isfinite(t_hours) else np.nan

# =========================
# Affichages
# =========================
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("U_cell (OCV @SOC perfs)", f"{ocv_cell(soc_perf,U_cell_nom):.2f} V")
c2.metric("R_cell eff.", f"{(R_cell_mohm_25C*1e-3)*r_cell_multiplier_soc(soc_perf)*r_cell_multiplier_temp(temp_c)*1e3:.1f} mΩ")
c3.metric("Wh/cell", f"{ocv_cell(soc_perf,U_cell_nom)*C_cell:.1f} Wh")
c4.metric("U_pack", f"{U_pack:.0f} V")
c5.metric("Capacité pack", f"{C_Ah:.0f} Ah")

c6,c7,c8,c9,c10 = st.columns(5)
c6.metric("Énergie pack", f"{E_kWh:.1f} kWh")
c7.metric("R_pack", f"{R_pack*1e3:.1f} mΩ")
c8.metric("Masse batterie", f"{m_pack:.0f} kg")
c9.metric("Interconnexions", f"{(m_inter):.0f} kg")
c10.metric("Masse moteur", f"{m_motor:.0f} kg")

c11,c12,c13,c14,c15 = st.columns(5)
c11.metric("Masse totale", f"{m_total:.0f} kg")
c12.metric("Puissance (tirette)", f"{PkW_slider:.0f} kW")
c13.metric("P_max arbre (limité)", f"{P_shaft_max/1000:.0f} kW")
c14.metric("0–100 km/h", f"{t_0_100:.1f} s" if np.isfinite(t_0_100) else "—")
c15.metric("Vitesse max", f"{vmax_kmh:.0f} km/h")

c16,c17,c18 = st.columns(3)
c16.metric("µ effectif pneus", f"{MU_TYRE_EFFECTIVE:.2f}")
c17.metric("Crr effectif", f"{Crr_eff:.3f}")
c18.metric("Autonomie @ cruise", f"{autonomy_km:.0f} km" if np.isfinite(autonomy_km) else "—")

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["Puissance vs vitesse", "SOC→OCV / R_cell", "0–100 : v(t)"])

with tab1:
    vs = np.linspace(0.5, max(vmax_ms*1.2, 60*KMH2MS), 260)
    P_req_curve = power_required(vs, m_total, Cx, S, Crr_eff)
    P_avail_curve = np.full_like(vs, P_shaft_max)
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(vs/ KMH2MS, P_req_curve/1000, label="P_demandée(v) (kW)")
    ax1.plot(vs/ KMH2MS, P_avail_curve/1000, label="P_disponible max (kW)")
    ax1.axvline(vmax_kmh, linestyle="--", label=f"Vmax ≈ {vmax_kmh:.0f} km/h")
    ax1.axvline(v_cruise_kmh, linestyle=":",  label=f"Cruise {v_cruise_kmh:.0f} km/h")
    ax1.set_xlabel("Vitesse (km/h)"); ax1.set_ylabel("Puissance (kW)")
    ax1.set_title("Puissance demandée vs disponible (Crr_eff)")
    ax1.grid(True); ax1.legend()
    st.pyplot(fig1, clear_figure=True)

with tab2:
    socs = np.linspace(0, 1, 200)
    ocvs = ocv_cell(socs, U_cell_nom)
    rcell = (R_cell_mohm_25C*1e-3) * r_cell_multiplier_soc(socs) * r_cell_multiplier_temp(temp_c)
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(socs*100, ocvs)
    ax2.set_xlabel("SOC (%)"); ax2.set_ylabel("OCV cellule (V)")
    ax2.set_title("Courbe SOC → OCV (cellule)")
    ax2.grid(True)
    st.pyplot(fig2, clear_figure=True)

    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    ax3.plot(socs*100, rcell*1e3)
    ax3.set_xlabel("SOC (%)"); ax3.set_ylabel("R_cell eff. (mΩ)")
    ax3.set_title(f"R interne cellule vs SOC @ {temp_c:.0f}°C")
    ax3.grid(True)
    st.pyplot(fig3, clear_figure=True)

with tab3:
    fig4, ax4 = plt.subplots(figsize=(7, 4.5))
    if "t" in traces_0100:
        ax4.plot(traces_0100["t"], traces_0100["v"]/KMH2MS)
        ax4.set_xlabel("Temps (s)"); ax4.set_ylabel("Vitesse (km/h)")
        ax4.set_title("0–100 km/h : v(t)")
        ax4.grid(True)
    st.pyplot(fig4, clear_figure=True)

with st.expander("Notes"):
    st.markdown(r"""
- **RPM base** agit directement sur le couple max : \(T_\text{max} = P_\text{max}/\omega_\text{base}\).
- **Pneus** : µ effectif augmente avec la largeur mais diminue avec la charge par roue (loi de charge).  
  **Crr_eff** ↑ avec la largeur et ↓ légèrement si le rayon de roue ↑.
- **Moteur plus gros ⇒ R_moteur ↓** : c’est **pris en compte dans l’autonomie** via l’équation
  \(I(U - I(R_\text{pack}+R_\text{moteur})) = P_\text{mec}/\eta_\text{core} + P_\text{aux}\).
- **0–100** : modèle forces (couple vs puissance), adhérence pneus, aéro+roulement, a ≤ 1 g.
- Modèle **pédagogique** (ordres de grandeur, non calibré constructeur).
""")