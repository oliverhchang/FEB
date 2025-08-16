import numpy as np
import matplotlib.pyplot as plt

k_f = 240.8517989  # Front roll stiffness
k_r = 176.0499249  # Rear roll stiffness
d_sF = 0.1  # Distance between front sprung mass and front roll center
d_sR = 0.1  # Distance between rear sprung mass and rear roll center
m_sF = 155  # Front sprung mass
m_sR = 155  # Rear sprung mass
h_G = 0.25  # Height of car CG
m = 310  # Total mass of car
z_F = 0.11  # Height of front roll center
z_uF = 0.08  # Height of front unsprung mass
m_uF = 12  # Front unsprung mass

lambda_values = np.linspace(0, 1, 100)
kc_values = [50, 100, 200, 400, 600, 1000, 1500, 2000, 10000]

lambda_values = np.linspace(0.2, 0.8, 100)


for k_c in kc_values:
    mu = k_c / (k_f + k_r)
    lltd_values = []
    for lambda_ in lambda_values:
        front_roll_stiffness_term = ((lambda_ ** 2 - (mu + 1) * lambda_) / (lambda_ ** 2 - lambda_ - mu)) * (d_sF * m_sF) / (h_G * m)
        rear_roll_stiffness_term = - (mu * lambda_ / (lambda_ ** 2 - lambda_ - mu)) * (d_sR * m_sR) / (h_G * m)
        front_roll_center_height_term = (z_F * m_sF) / (h_G * m)
        front_unsprung_mass_term = (z_uF * m_uF) / (h_G * m)

        x = front_roll_stiffness_term + rear_roll_stiffness_term + front_roll_center_height_term + front_unsprung_mass_term
        lltd_values.append(x)

    plt.plot(lambda_values, lltd_values, label=f'kc = {k_c} Nm/deg')

plt.xlabel("Roll Stiffness Distribution (λ)")
plt.ylabel("Lateral Load Transfer Distribution (LLTD)")
plt.title("LLTD vs Roll Stiffness Distribution for Different Torsional Stiffness")
plt.legend()
plt.grid(True)
plt.xlim(0.2, 0.8)
plt.show()
