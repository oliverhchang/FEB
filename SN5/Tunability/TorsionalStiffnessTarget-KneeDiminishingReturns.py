import numpy as np
import matplotlib.pyplot as plt

# ─── Vehicle Parameters ───────────────────────────────────────────────
p = {
    'wheelbase': 1.55, 'a_s': 0.7525, 'h_G': 0.279,
    'z_F': 0.115, 'z_R': 0.165, 'd_sF': 0.150, 'd_sR': 0.130, 'z_uF': 0.12,
    'm': 310.0, 'm_uF': 12.0,
    'k_f': 275.0, 'k_r': 292.0,
    'k_f_max': 450.0, 'k_r_max': 500.0,
}

total_sprung = p['m'] - p['m_uF'] * 2
b_s = p['wheelbase'] - p['a_s']
p['m_sF'] = total_sprung * b_s / p['wheelbase']
p['m_sR'] = total_sprung * p['a_s'] / p['wheelbase']

k_ref   = p['k_f'] + p['k_r']
lam_min = p['k_f']     / (p['k_f']     + p['k_r_max'])
lam_max = p['k_f_max'] / (p['k_f_max'] + p['k_r'])


# ─── LLTD ─────────────────────────────────────────────────────────────
def lltd(lam, mu, p):
    d = lam**2 - lam - mu
    d = np.where(np.abs(d) < 1e-9, np.sign(d) * 1e-9, d) if isinstance(d, np.ndarray) else (1e-9 if d == 0 else d)
    t1 = ((lam**2 - (mu + 1)*lam) / d) * (p['d_sF']*p['m_sF']) / (p['h_G']*p['m'])
    t2 = (mu*lam / d)                  * (p['d_sR']*p['m_sR']) / (p['h_G']*p['m'])
    t3 = (p['z_F']*p['m_sF'])          / (p['h_G']*p['m'])
    t4 = (p['z_uF']*p['m_uF'])         / (p['h_G']*p['m'])
    return t1 - t2 + t3 + t4


# ─── Sweep ────────────────────────────────────────────────────────────
kc = np.logspace(np.log10(50), np.log10(8000), 1000)
mu = kc / k_ref

lltd_rigid_min = lltd(lam_min, 1e9, p)
lltd_rigid_max = lltd(lam_max, 1e9, p)

err_min = (lltd(lam_min, mu, p) - lltd_rigid_min) / lltd_rigid_min * 100
err_max = (lltd(lam_max, mu, p) - lltd_rigid_max) / lltd_rigid_max * 100
total_err = err_min - err_max   # always positive, monotone decreasing


# ─── Curvature ────────────────────────────────────────────────────────
def curvature(x, y, log_x=True):
    xn = np.log10(x) if log_x else x.copy()
    xn = (xn - xn.min()) / (xn.max() - xn.min())
    yn = (y  - y.min())  / (y.max()  - y.min())
    dx, dy   = np.gradient(xn), np.gradient(yn)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    num = np.abs(dx*ddy - dy*ddx)
    den = np.where((dx**2 + dy**2)**1.5 < 1e-12, 1e-12, (dx**2 + dy**2)**1.5)
    return num / den

kappa    = curvature(kc, total_err)
knee_idx = np.argmax(kappa)
kc_knee  = kc[knee_idx]
err_knee = total_err[knee_idx]
kc_spec  = np.ceil(kc_knee / 100) * 100

print(f"Knee:  {kc_knee:.0f} Nm/deg  ({err_knee:.1f}% error)")
print(f"Spec:  {kc_spec:.0f} Nm/deg")


# ─── Plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("SN5 Torsional Stiffness Target — Knee of Diminishing Returns", fontsize=12)

# --- Panel 1: Total error range ---
ax = axes[0]
ax.plot(kc, total_err, color='C0')
ax.axvline(kc_knee, color='C3', linestyle='--', linewidth=1)
ax.scatter([kc_knee], [err_knee], color='C3', zorder=5)
ax.set_xscale('log')
ax.set_xlabel('k_c  [Nm/deg]')
ax.set_ylabel('Total LLTD error range  [%]')
ax.set_title('Error vs Chassis Stiffness')
ax.grid(True, which='both', linestyle=':', alpha=0.4)

# --- Panel 2: Curvature ---
ax = axes[1]
ax.plot(kc, kappa, color='C0')
ax.axvline(kc_knee, color='C3', linestyle='--', linewidth=1,
           label=f'Knee: {kc_knee:.0f} Nm/deg')
ax.scatter([kc_knee], [kappa[knee_idx]], color='C3', zorder=5)
ax.set_xscale('log')
ax.set_xlabel('k_c  [Nm/deg]')
ax.set_ylabel('Curvature  κ  [normalized]')
ax.set_title('Curvature of Error Curve')
ax.legend(fontsize=9)
ax.grid(True, which='both', linestyle=':', alpha=0.4)

# --- Panel 3: LLTD tuning window ---
ax = axes[2]
ax.fill_between(kc, lltd(lam_min, mu, p)*100, lltd(lam_max, mu, p)*100,
                alpha=0.15, color='C0', label='Tuning window')
ax.plot(kc, lltd(lam_max, mu, p)*100, color='C1', label=f'λ={lam_max:.2f} (max RSD)')
ax.plot(kc, lltd(lam_min, mu, p)*100, color='C0', label=f'λ={lam_min:.2f} (min RSD)')
ax.axhline(lltd_rigid_max*100, color='C1', linestyle='--', linewidth=0.8, alpha=0.6)
ax.axhline(lltd_rigid_min*100, color='C0', linestyle='--', linewidth=0.8, alpha=0.6,
           label='Rigid limits')
ax.axvline(kc_knee, color='C3', linestyle='--', linewidth=1,
           label=f'Spec: {kc_spec:.0f} Nm/deg')
ax.set_xscale('log')
ax.set_xlabel('k_c  [Nm/deg]')
ax.set_ylabel('LLTD χ  [%]')
ax.set_title('LLTD Tuning Window')
ax.legend(fontsize=8)
ax.grid(True, which='both', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.show()