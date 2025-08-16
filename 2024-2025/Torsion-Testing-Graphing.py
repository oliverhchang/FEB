import matplotlib.pyplot as plt
import numpy as np

# Torsion test data
# Displacement in degrees and torque in Newton-meters

# Test 1
disp_test1 = [0, 0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.84, 0.96, 1.08, 1.19]
torque_test1 = [0, 123, 246, 369, 492, 615, 738, 861, 984, 1108, 1232]

# Test 2
disp_test2 = [0, 0.12, 0.24, 0.36, 0.48, 0.59, 0.71, 0.83, 0.95, 1.07, 1.18]
torque_test2 = [0, 113, 226, 339, 452, 565, 678, 791, 904, 1017, 1161]

# Test 3
disp_test3 = [0, 0.14, 0.28, 0.42, 0.56, 0.70, 0.84, 0.98, 1.12, 1.26, 1.40]
torque_test3 = [0, 138, 276, 414, 552, 690, 828, 966, 1104, 1242, 1326]

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(disp_test1, torque_test1, 'bo-', label='Test 1 (TR = 1018 Nm/deg)', alpha=0.8, linewidth=2, markersize=6)
plt.plot(disp_test2, torque_test2, 'rs-', label='Test 2 (TR = 940 Nm/deg)', alpha=0.8, linewidth=2, markersize=6)
plt.plot(disp_test3, torque_test3, 'g^-', label='Test 3 (TR = 983 Nm/deg)', alpha=0.8, linewidth=2, markersize=6)

# Linear Trends
fit1 = np.polyfit(disp_test1, torque_test1, 1)
poly1 = np.poly1d(fit1)
plt.plot(disp_test1, poly1(disp_test1), 'b--', alpha=0.5, linewidth=1)

fit2 = np.polyfit(disp_test2, torque_test2, 1)
poly2 = np.poly1d(fit2)
plt.plot(disp_test2, poly2(disp_test2), 'r--', alpha=0.5, linewidth=1)

fit3 = np.polyfit(disp_test3, torque_test3, 1)
poly3 = np.poly1d(fit3)
plt.plot(disp_test3, poly3(disp_test3), 'g--', alpha=0.5, linewidth=1)

# Legend
plt.xlabel('Angular Displacement (degrees)', fontsize=14, fontweight='bold')
plt.ylabel('Torque (Nm)', fontsize=14, fontweight='bold')
plt.title('Torque vs Angular Displacement', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
plt.legend(loc='lower right', fontsize=12, framealpha=0.9)

plt.xlim(-0.05, 1.5)
plt.ylim(-50, 1400)
plt.tight_layout()
plt.show()