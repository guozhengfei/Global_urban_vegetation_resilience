import numpy as np
import matplotlib;
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rc('font', family='Arial')
plt.rc('lines', linewidth=1.0)  # Increase default line width
plt.rc('axes', linewidth=1.0)   # Increase axes line width
plt.rc('grid', linewidth=1.0)   # Increase grid line width
plt.tick_params(width=1.0, labelsize=14)
plt.rc('xtick.major', size=4, width=1.0)    # Increase length and width of major ticks
plt.rc('ytick.major', size=4, width=1.0)    # Increase length and width of major ticks
plt.close()

import pandas as pd
import os

# Create time series
t = np.linspace(0, 140, 140)

# Function to create the disturbance and recovery pattern
def create_pattern(min_val, recovery_rate, noise_level=0.06):
    # Initial fluctuation
    y = np.random.normal(0, noise_level, len(t))
    
    # Disturbance and recovery
    disturbance_center = 40
    recovery_center = 40
    
    # Create disturbance
    disturbance = min_val * np.exp(-(t - disturbance_center)**2 / 50)
    disturbance[disturbance_center:] = 0
    
    # Create recovery
    recovery = np.where(t > recovery_center,
                       min_val * np.exp(-(t - recovery_center) * recovery_rate),
                       0)
    recovery[0:disturbance_center] = 0
    return y + disturbance + recovery

# Create three patterns
# pattern1 = create_pattern(min_val=-0.5, recovery_rate=0.15)*0.1  # Quick recovery
# pattern2 = create_pattern(min_val=-1.0, recovery_rate=0.13)*0.1  # Medium recovery
pattern3 = create_pattern(min_val=-1.0, recovery_rate=0.20)*0.1  # Slow recovery

# Create figure
plt.figure(figsize=(11 * 0.72, 3.5))

# Plot patterns
# plt.plot(t, pattern1, label='High resilience', color='#2166ac', linewidth=2)
# plt.plot(t, pattern2, label='Midum resilience', color='#67a9cf', linewidth=2)
plt.plot((t[::6]/6)[2:25], (pattern3[::6]*1.5)[2:25]+0.35, label='Low resilience', color='#b2182b', linewidth=2)

# Customize plot
# plt.ylim(-0.24, 0.04)
plt.xlabel('Year')
plt.ylabel('Detrend growing-season kNDVI')

# Adjust layout and save
plt.tight_layout()
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
plt.savefig(os.path.join(current_dir, '4_Figures', 'Disturbance_events_example.png'), dpi=900)
plt.show()