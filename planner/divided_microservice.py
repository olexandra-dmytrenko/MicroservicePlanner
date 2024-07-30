import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = np.random.randint(0, 101, 24)

# Create a figure with 3 subplots horizontally
fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Original data with red fill above 50
ax1.bar(range(24), data, color='blue')
ax1.bar(range(24), np.maximum(0, data - 50), bottom=50, color='red')
ax1.axhline(y=50, color='black', linestyle='--')
ax1.set_title('Original Data')
ax1.set_ylim(0, 100)

# Plot 3: 100 - values > 50
data_above_50 = np.where(data > 50, 100 - data, 0)
ax3.bar(range(24), data_above_50, color='blue')
ax3.axhline(y=50, color='black', linestyle='--')
ax3.set_title('100 - Values ≤ 50')
ax3.set_ylim(0, 100)

# Plot 2: Values ≤ 50
data_below_50 = np.where(data <= 50, data, 0)
ax2.bar(range(24), data_below_50, color='red')
ax2.set_title('Values > 50')
ax2.axhline(y=50, color='black', linestyle='--')
ax2.set_ylim(0, 100)

# Common settings for all subplots
for ax in (ax1, ax2, ax3):
    ax.set_xlabel('Hour')
    ax.set_ylabel('Load')
    ax.set_xticks(range(0, 24, 4))
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()