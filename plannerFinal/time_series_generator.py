import numpy as np
import matplotlib.pyplot as plt


class TimeSeriesGenerator:
    def __init__(self, microservices=None):
        if microservices is None:
            self.microservices = []
        else:
            self.microservices = microservices


# Set random seed for reproducibility
np.random.seed(42)


# Helper function to add noise to the data and round to integers
def add_noise_and_round(data, noise_level=0.1):
    noisy_data = data + np.random.normal(0, noise_level, data.shape)
    return np.clip(np.round(noisy_data).astype(int), 0, 100)


# Helper function to create gradual peaks
def gradual_peak(length, peak_height, peak_center, peak_width):
    x = np.arange(length)
    return peak_height * np.exp(-((x - peak_center) ** 2) / (2 * peak_width ** 2))


# 1. Stably loaded (2 microservices)
stable_load = np.full((2, 24), 50)
stable_load = add_noise_and_round(stable_load, 5)

# 2. Big peak in working hours (4 microservices)
working_hours_peak = np.zeros((4, 24))
for i in range(4):
    working_hours_peak[i] = gradual_peak(24, 80, 13, 4)
working_hours_peak = add_noise_and_round(working_hours_peak, 10)

# 3. Peaks after 12 am and before 6 am (2 microservices)
night_peak = np.zeros((2, 24))
for i in range(2):
    night_peak[i] = gradual_peak(24, 70, 3, 3) + gradual_peak(24, 70, 23, 3)
night_peak = add_noise_and_round(night_peak, 10)

# 4. Peaks from 5 pm till night (2 microservices)
evening_peak = np.zeros((2, 24))
for i in range(2):
    evening_peak[i] = gradual_peak(24, 75, 20, 4)
evening_peak = add_noise_and_round(evening_peak, 10)

# 5. Random peaks (2 microservices)
random_peaks = np.zeros((2, 24))
for i in range(2):
    num_peaks = np.random.randint(3, 6)
    for _ in range(num_peaks):
        peak_center = np.random.randint(0, 24)
        peak_height = np.random.randint(50, 80)
        random_peaks[i] += gradual_peak(24, peak_height, peak_center, 2)
random_peaks = add_noise_and_round(random_peaks, 10)

# Combine all patterns
all_patterns = np.vstack((stable_load, working_hours_peak, night_peak, evening_peak, random_peaks))

# Generate complements (opposite time load)
complements = 100 - all_patterns

# Combine original patterns and complements
microservices = np.vstack((all_patterns, complements))

# Ensure all values are integers between 0 and 100
microservices = np.clip(microservices, 0, 100).astype(int)

# Plot the data with colors for each type of microservice
plt.figure(figsize=(15, 10))
colors = ['blue', 'green', 'red', 'purple', 'orange']
labels = ['Stable', 'Working Hours Peak', 'Night Peak', 'Evening Peak', 'Random Peaks']
hours = np.arange(24)  # Generate 24 hours of data

for i, (color, label) in enumerate(zip(colors, labels)):
    if i == 0:
        plt.plot(hours, microservices[i], color=color, label=f'{label} (Original)')
        plt.plot(hours, microservices[i + 12], color=color, linestyle='--', label=f'{label} (Complement)')
    elif i == 1:
        for j in range(4):
            plt.plot(hours, microservices[i + j], color=color, label=f'{label} (Original)' if j == 0 else '')
            plt.plot(hours, microservices[i + j + 12], color=color, linestyle='--',
                     label=f'{label} (Complement)' if j == 0 else '')
    else:
        for j in range(2):
            plt.plot(hours, microservices[i * 2 + j + 2], color=color, label=f'{label} (Original)' if j == 0 else '')
            plt.plot(hours, microservices[i * 2 + j + 14], color=color, linestyle='--',
                     label=f'{label} (Complement)' if j == 0 else '')

plt.title('Processor Load for 24 Microservices')
plt.xlabel('Hour of Day')
plt.ylabel('Processor Load (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Save the data to a file
np.savetxt('microservices_load.csv', microservices, delimiter=',', fmt='%d')
print("Data saved to 'microservices_load.csv'")

# Print the first few rows of the data
print("\nFirst few rows of the data:")
print(microservices[:24])
