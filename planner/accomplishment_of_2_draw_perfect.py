import matplotlib.pyplot as plt
import numpy as np

# Sample data as dictionaries
#    'Time_of_Day': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00'],
freq1 = [10, 15, 20, 25, 30, 30, 40, 45, 50, 65, 60, 60, 70, 75, 80, 85, 90, 80, 60, 50, 38, 30, 30, 20]

freq2 = [80, 75, 70, 65, 60, 60, 50, 45, 40, 25, 30, 30, 20, 15, 10, 5,   0, 10, 30, 40, 52, 60, 60, 70]

freq3 = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

vect1 = np.array(freq1)
vect2 = np.array(freq2)
vect3 = np.array(freq3)

vect_sum = (vect1 + vect2 + vect3)
vect_sum_max = max(vect_sum)

freq_Unused = (np.full(24, vect_sum_max) - vect_sum).tolist()

time_of_day = range(0, len(freq1))

#time_of_day = np.linspace(0, len(freq1), len(freq1))

# Plot the bars
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(time_of_day, freq1, width=1.0, color='lightblue', edgecolor='black')
ax1.bar(time_of_day, freq2, width=1.0, color='lightcoral', edgecolor='black', bottom=freq1)
ax1.bar(time_of_day, freq3, width=1.0, color='lightyellow', edgecolor='black', bottom=[i + j for i, j in zip(freq1, freq2)])
#ax1.bar(time_of_day, freq_Unused, width=1.0, color='lightgreen', edgecolor='black', bottom=vect_sum.tolist())
#ax1.bar(time_of_day, freq3, width=1.0, color='lightred', edgecolor='black') #, bottom=vect_sum.tolist())

# Set the y-axis labels and ticks
ax1.set_yticks(np.linspace(0, vect_sum_max, int(vect_sum_max / 20)))
#ax1.set_yticklabels(vect_sum)
ax1.set_ylabel('Використання ресурсу')

# Set the x-axis label and title
ax1.set_xticks(time_of_day)
ax1.set_xlabel('Час, 24 години')
ax1.set_title('Використання процесорного ресурсу у відповідний час, ідеальний випадок')

# Add a legend
ax1.legend(['Процес 1', 'Процес 2', 'Безпечний надлишковий діапазон ±10%'])

# Create a secondary y-axis at the top
#ax2 = ax1.twiny()
#ax2.set_xticks(time_of_day)


# Hide the tick marks and labels for the secondary y-axis
#ax2.xaxis.set_visible(False)

# Show the plot
plt.grid(True)
plt.show()
