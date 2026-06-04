#!/usr/bin/env python3
"""
Verify J2 precession of RAAN and Argument of Periapsis
Compares theoretical rates with simulated values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Constants
MU_EARTH = 398600.4418  # km^3/s^2
J2_EARTH = 0.00108248
RE_EARTH = 6371  # km

# Read CSV data
csv_file = "Results_full_orbit/Results_20260509/sat_propagation_j2_file.csv"
try:
    data = pd.read_csv(csv_file)
    print(f"✓ Loaded satellite propagation data from {csv_file}")
except FileNotFoundError:
    print(f"✗ Error: Could not find {csv_file}")
    print("Make sure you're in the workspace root directory")
    exit(1)

# Convert columns to standard names (strip whitespace)
data.columns = data.columns.str.strip()

# Extract orbital elements
time = data['Time (sec)'].values
a = data['Semi-Major Axis (km)'].values
e = data['Eccentricity'].values
i = np.radians(data['Inclination (deg)'].values)  # Convert to radians
raan = np.radians(data['RAAN (deg)'].values)  # Convert to radians
aop = np.radians(data['Argument of Periapsis (deg)'].values)  # Convert to radians

print(f"\n{'='*60}")
print("SATELLITE ORBITAL PARAMETERS")
print(f"{'='*60}")
print(f"Semi-major axis (a):     {a[0]:.2f} km")
print(f"Eccentricity (e):        {e[0]:.6f}")
print(f"Inclination (i):         {np.degrees(i[0]):.2f}°")
print(f"Orbit period:            {2*np.pi*np.sqrt(a[0]**3/MU_EARTH):.2f} seconds")
print(f"Mean motion (n):         {np.sqrt(MU_EARTH/a[0]**3):.6f} rad/s")

# Calculate mean motion
n = np.sqrt(MU_EARTH / a**3)  # rad/s

# Calculate theoretical J2 precession rates
# RAAN precession: dΩ/dt = -3/2 * n * J2 * (RE/a)^2 * cos(i) / (1-e^2)^2
raan_precession_rate = (-3/2) * n * J2_EARTH * (RE_EARTH/a)**2 * np.cos(i) / (1 - e**2)**2
raan_precession_rate_deg_per_day = np.degrees(raan_precession_rate) * 86400  # deg/day

# AoP precession: dω/dt = 3/4 * n * J2 * (RE/a)^2 * (5*cos²(i) - 1) / (1-e^2)^2
aop_precession_rate = (3/4) * n * J2_EARTH * (RE_EARTH/a)**2 * (5*np.cos(i)**2 - 1) / (1 - e**2)**2
aop_precession_rate_deg_per_day = np.degrees(aop_precession_rate) * 86400  # deg/day

# Average theoretical rates (since a and e are nearly constant)
avg_raan_rate = np.mean(raan_precession_rate_deg_per_day)
avg_aop_rate = np.mean(aop_precession_rate_deg_per_day)

# Calculate orbital period for deg/period conversion
orbital_period = 2*np.pi*np.sqrt(a[0]**3/MU_EARTH)
orbital_period_days = orbital_period / 86400

print(f"\n{'='*60}")
print("THEORETICAL J2 PRECESSION RATES (from formulas)")
print(f"{'='*60}")
print(f"dΩ/dt (RAAN):            {avg_raan_rate:+.6f}°/day")
print(f"                         {avg_raan_rate/360:+.6f} rev/day")
print(f"                         {avg_raan_rate*orbital_period_days:+.6f}°/period")
print(f"dω/dt (Argument of Periapsis): {avg_aop_rate:+.6f}°/day")
print(f"                         {avg_aop_rate/360:+.6f} rev/day")
print(f"                         {avg_aop_rate*orbital_period_days:+.6f}°/period")

# Calculate observed precession rates from simulation
# Remove duplicate time steps (where orbital elements don't change)
unique_indices = np.where(np.diff(time) > 0)[0]
unique_indices = np.append(unique_indices, len(time) - 1)

time_unique = time[unique_indices]
raan_unique = raan[unique_indices]
aop_unique = aop[unique_indices]

# Use linear regression to fit precession rates (more robust than finite differences)
# This handles the small angle changes better
from scipy.stats import linregress

# For RAAN, unwrap the angle to handle discontinuities
raan_unwrapped = np.degrees(raan_unique).copy()
for i in range(1, len(raan_unwrapped)):
    # Handle 360 degree wrap
    if raan_unwrapped[i] - raan_unwrapped[i-1] > 180:
        raan_unwrapped[i] -= 360
    elif raan_unwrapped[i] - raan_unwrapped[i-1] < -180:
        raan_unwrapped[i] += 360

# For AoP, similar unwrapping
aop_unwrapped = np.degrees(aop_unique).copy()
for i in range(1, len(aop_unwrapped)):
    if aop_unwrapped[i] - aop_unwrapped[i-1] > 180:
        aop_unwrapped[i] -= 360
    elif aop_unwrapped[i] - aop_unwrapped[i-1] < -180:
        aop_unwrapped[i] += 360

# Perform linear regression in deg/sec
slope_raan, intercept_raan, r_value_raan, p_value_raan, std_err_raan = linregress(time_unique, raan_unwrapped)
slope_aop, intercept_aop, r_value_aop, p_value_aop, std_err_aop = linregress(time_unique, aop_unwrapped)

# Convert from deg/sec to deg/day
avg_observed_raan_rate = slope_raan * 86400
avg_observed_aop_rate = slope_aop * 86400

# Also calculate instantaneous rates for plotting
draan_deg_day = np.zeros_like(time_unique)
daop_deg_day = np.zeros_like(time_unique)

for i in range(1, len(time_unique)):
    dt = (time_unique[i] - time_unique[i-1]) / 86400  # in days
    draan_deg_day[i] = (raan_unwrapped[i] - raan_unwrapped[i-1]) / dt
    daop_deg_day[i] = (aop_unwrapped[i] - aop_unwrapped[i-1]) / dt

print(f"\n{'='*60}")
print("OBSERVED PRECESSION RATES (from simulation)")
print(f"{'='*60}")
print(f"dΩ/dt (RAAN):            {avg_observed_raan_rate:+.6f}°/day")
print(f"                         {avg_observed_raan_rate/360:+.6f} rev/day")
print(f"                         {avg_observed_raan_rate*orbital_period_days:+.6f}°/period")
print(f"                         R² = {r_value_raan**2:.6f} (linear fit quality)")
print(f"dω/dt (Argument of Periapsis): {avg_observed_aop_rate:+.6f}°/day")
print(f"                         {avg_observed_aop_rate/360:+.6f} rev/day")
print(f"                         {avg_observed_aop_rate*orbital_period_days:+.6f}°/period")
print(f"                         R² = {r_value_aop**2:.6f} (linear fit quality)")

# Calculate errors
raan_error = avg_observed_raan_rate - avg_raan_rate
aop_error = avg_observed_aop_rate - avg_aop_rate

raan_error_percent = (raan_error / avg_raan_rate) * 100 if avg_raan_rate != 0 else 0
aop_error_percent = (aop_error / avg_aop_rate) * 100 if avg_aop_rate != 0 else 0

print(f"\n{'='*60}")
print("ERROR ANALYSIS")
print(f"{'='*60}")
print(f"RAAN error:              {raan_error:+.6f}°/day ({raan_error_percent:+.2f}%)")
print(f"AoP error:               {aop_error:+.6f}°/day ({aop_error_percent:+.2f}%)")

# Create detailed comparison table
print(f"\n{'='*60}")
print("DETAILED PRECESSION COMPARISON")
print(f"{'='*60}")
print(f"{'Time (hrs)':<12} {'RAAN (°)':<15} {'AoP (°)':<15} {'Theo Rate':<15} {'Obs Rate (RAAN)':<18}")
print(f"{'-'*75}")

theo_raan_rate_at_0 = np.degrees(raan_precession_rate[0]) * 86400
theo_aop_rate_at_0 = np.degrees(aop_precession_rate[0]) * 86400

for idx in range(0, len(time_unique), max(1, len(time_unique)//10)):
    hours = time_unique[idx] / 3600
    print(f"{hours:<12.2f} {raan_unwrapped[idx]:<15.6f} {aop_unwrapped[idx]:<15.6f} "
          f"{theo_raan_rate_at_0:<15.6f} {draan_deg_day[idx]:<18.6f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('J2 Precession Verification: RAAN and Argument of Periapsis', fontsize=14, fontweight='bold')

# Plot 1: RAAN over time
ax = axes[0, 0]
ax.plot(time_unique/3600, np.degrees(raan_unique), 'b-', linewidth=2, label='Simulated')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('RAAN (degrees)')
ax.set_title('Right Ascension of Ascending Node')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Argument of Periapsis over time
ax = axes[0, 1]
ax.plot(time_unique/3600, np.degrees(aop_unique), 'r-', linewidth=2, label='Simulated')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Argument of Periapsis (degrees)')
ax.set_title('Argument of Periapsis')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 3: RAAN precession rate
ax = axes[1, 0]
# Plot fitted line
time_fit = np.array([time_unique[0], time_unique[-1]]) / 3600
raan_fit = intercept_raan + slope_raan * np.array([time_unique[0], time_unique[-1]])
ax.plot(time_fit, [avg_observed_raan_rate, avg_observed_raan_rate], 'b-', linewidth=2.5, label=f'Linear Fit: {avg_observed_raan_rate:.4f}°/day (R²={r_value_raan**2:.4f})')
ax.axhline(y=avg_raan_rate, color='g', linestyle='--', linewidth=2, label=f'Theoretical Avg: {avg_raan_rate:.4f}°/day')
if len(draan_deg_day) > 1:
    ax.scatter(time_unique[1:]/3600, draan_deg_day[1:], c='b', alpha=0.4, s=20, label='Instantaneous rates')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('dΩ/dt (degrees/day)')
ax.set_title('RAAN Precession Rate')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# Plot 4: AoP precession rate
ax = axes[1, 1]
# Plot fitted line
aop_fit = intercept_aop + slope_aop * np.array([time_unique[0], time_unique[-1]])
ax.plot(time_fit, [avg_observed_aop_rate, avg_observed_aop_rate], 'r-', linewidth=2.5, label=f'Linear Fit: {avg_observed_aop_rate:.4f}°/day (R²={r_value_aop**2:.4f})')
ax.axhline(y=avg_aop_rate, color='g', linestyle='--', linewidth=2, label=f'Theoretical Avg: {avg_aop_rate:.4f}°/day')
if len(daop_deg_day) > 1:
    ax.scatter(time_unique[1:]/3600, daop_deg_day[1:], c='r', alpha=0.4, s=20, label='Instantaneous rates')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('dω/dt (degrees/day)')
ax.set_title('Argument of Periapsis Precession Rate')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('j2_precession_verification.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: j2_precession_verification.png")

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

if abs(raan_error_percent) < 5:
    raan_status = "✓ PASS"
else:
    raan_status = "⚠ FAIL" if abs(raan_error_percent) > 10 else "⚠ MARGINAL"
    
if abs(aop_error_percent) < 5:
    aop_status = "✓ PASS"
else:
    aop_status = "⚠ FAIL" if abs(aop_error_percent) > 10 else "⚠ MARGINAL"

print(f"RAAN Precession:       {raan_status}")
print(f"  Theoretical:         {avg_raan_rate:+.6f}°/day ({avg_raan_rate*orbital_period_days:+.6f}°/period)")
print(f"  Observed (fit):      {avg_observed_raan_rate:+.6f}°/day ({avg_observed_raan_rate*orbital_period_days:+.6f}°/period)")
print(f"  Error:               {raan_error:+.6f}°/day ({raan_error_percent:+.2f}%)")
print(f"  Linear Fit Quality:  R² = {r_value_raan**2:.6f}")

print(f"\nArgument of Periapsis: {aop_status}")
print(f"  Theoretical:         {avg_aop_rate:+.6f}°/day ({avg_aop_rate*orbital_period_days:+.6f}°/period)")
print(f"  Observed (fit):      {avg_observed_aop_rate:+.6f}°/day ({avg_observed_aop_rate*orbital_period_days:+.6f}°/period)")
print(f"  Error:               {aop_error:+.6f}°/day ({aop_error_percent:+.2f}%)")
print(f"  Linear Fit Quality:  R² = {r_value_aop**2:.6f}")

# Additional analysis
print(f"\n{'='*60}")
print("INTERPRETATION")
print(f"{'='*60}")
if avg_raan_rate < 0:
    print(f"✓ RAAN precesses CLOCKWISE (negative): Expected for i > 63.43°")
else:
    print(f"✓ RAAN precesses COUNTER-CLOCKWISE (positive)")
    
if avg_aop_rate < 0:
    print(f"✓ Argument of Periapsis precesses CLOCKWISE (negative)")
else:
    print(f"✓ Argument of Periapsis precesses COUNTER-CLOCKWISE (positive)")

total_duration_days = (time_unique[-1] - time_unique[0]) / 86400
total_raan_change = raan_unwrapped[-1] - raan_unwrapped[0]
total_aop_change = aop_unwrapped[-1] - aop_unwrapped[0]

print(f"\nSimulation Duration:   {total_duration_days:.6f} days ({(time_unique[-1] - time_unique[0]):.1f} seconds)")
print(f"Total RAAN change:     {total_raan_change:+.6f}°")
print(f"Total AoP change:      {total_aop_change:+.6f}°")

# Print J2 formulas for reference
print(f"\n{'='*60}")
print("J2 PERTURBATION FORMULAS (Used in Verification)")
print(f"{'='*60}")
print("RAAN Precession Rate:")
print("  dΩ/dt = -3/2 * n * J2 * (RE/a)² * cos(i) / (1-e²)²")
print("\nArgument of Periapsis Precession Rate:")
print("  dω/dt = 3/4 * n * J2 * (RE/a)² * (5*cos²(i) - 1) / (1-e²)²")
print("\nWhere:")
print("  n = mean motion (rad/s)")
print("  J2 = Earth's J2 coefficient (0.00108248)")
print("  RE = Earth's equatorial radius (6371 km)")
print("  a = semi-major axis (km)")
print("  e = eccentricity")
print("  i = inclination (radians)")
print(f"{'='*60}")
