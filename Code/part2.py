import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from part1 import detect_qrs_af2, HR_calculate

print("*************part2***************")

signal =  np.loadtxt('108-b.txt', delimiter='\t', skiprows=1)

# Extract leads
lead_I = signal[:, 0]
lead_II = signal[:, 1]
lead_III = signal[:, 2]

fs=500
t = np.arange(len(lead_I)) / fs

ig, axs = plt.subplots(3, 1, figsize=(10, 10))
axs[0].plot(t, lead_I, label='Original signal')
axs[0].set_title('Lead I')
axs[0].set_xlabel('Time (sec)')
axs[0].set_ylabel('Amplitude (mV)')
axs[0].legend()

axs[1].plot(t, lead_II, label='Original signal')
axs[1].set_title('Lead II')
axs[1].set_xlabel('Time (sec)')
axs[1].set_ylabel('Amplitude (mV)')
axs[1].legend()

axs[2].plot(t, lead_III, label='Original signal')
axs[2].set_title('Lead III')
axs[2].set_xlabel('Time (sec)')
axs[2].set_ylabel('Amplitude (mV)')
axs[2].legend()

plt.tight_layout()
plt.show()


# Apply filtering to the lead signal
fs = 500  # Sampling frequency
nyq = 0.5 * fs
lowcut = 0.5  # Low cutoff frequency 0.5 Hz
highcut = 100  # High cutoff frequency 100 Hz
order = 4  # Filter order
b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
filtered_lead_I = lfilter(b, a, lead_I)

t = np.arange(len(filtered_lead_I)) / fs

r_peaks_indices_I = detect_qrs_af2(filtered_lead_I, fs)

# Plotting the ECG signal and marking the detected R-peaks
plt.figure(figsize=(20, 6))
plt.plot(t, filtered_lead_I, label='ECG signal')
plt.scatter(t[r_peaks_indices_I], filtered_lead_I[r_peaks_indices_I], color='red', s=50, label='R peaks detection', zorder=3)
plt.title('Part2: Lead I QRS Detection')
plt.xlabel('Time [sec]')
plt.ylabel('Lead I [mV]')
plt.legend()
plt.grid(True)
plt.show()

HR_values = HR_calculate(r_peaks_indices_I)

HR_values = HR_values[np.isfinite(HR_values)]

# Calculate mean and standard deviation of HR
mean_hr = np.mean(HR_values)
std_hr = np.std(HR_values)

print(f"Mean HR: {mean_hr:.2f} BPM")
print(f"Standard Deviation of HR: {std_hr:.2f} BPM")

# Plotting 15 HR values with mean and standard deviation
plt.figure(figsize=(10, 6))
samples = np.arange(1, 16)
plt.plot(samples, HR_values[30:45], marker='o', linestyle='-', label='HR Values')
plt.axhline(mean_hr, color='r', linestyle='--', label=f'Mean HR: {mean_hr:.2f} BPM')
plt.axhline(mean_hr + std_hr, color='g', linestyle=':', label=f'Mean HR ± Std: {std_hr:.2f} BPM')
plt.axhline(mean_hr - std_hr, color='g', linestyle=':')
plt.title('Heart Rate for 15 Consecutive Beats')
plt.xlabel('Beat Number')
plt.ylabel('Heart Rate (BPM)')
plt.legend()
plt.show()



########

r_peaks_indices_I = detect_qrs_af2(filtered_lead_I, fs)

# window arround each R peak (each heart beat)
pre_r = int(0.2 * 1000)  # 200ms before R peak
post_r = int(0.3 * 1000)  # 300ms aftrer R peak

# heart cycles cutting
heart_cycles = [filtered_lead_I[idx-pre_r:idx+post_r] for idx in r_peaks_indices_I if idx-pre_r > 0 and idx+post_r < len(filtered_lead_I)]


# cheching that all the cycles are the same length
min_length = min(len(cycle) for cycle in heart_cycles)
heart_cycles = np.array([cycle[:min_length] for cycle in heart_cycles])


# average cycle calculation
average_cycle = np.mean(heart_cycles, axis=0)


plt.plot(average_cycle)
plt.title('Average Heart Cycle')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.show()

plt.figure(figsize=(15, 10))

# average cycle
plt.subplot(3, 1, 1)
plt.plot(average_cycle)
plt.title('Average Heart Cycle')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

# First Detected Heart Cycle
plt.subplot(3, 1, 2)
plt.plot(heart_cycles[0])
plt.title('First Detected Heart Cycle')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

# Last Detected Heart Cycle
plt.subplot(3, 1, 3)
plt.plot(heart_cycles[-1])
plt.title('Last Detected Heart Cycle')
plt.tight_layout()
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.show()

# isoelectric part 100 ms.
iso_first_HB = heart_cycles[0][50:100]
iso_last_HB = heart_cycles[-1][50:100]
iso_avg_HB =  average_cycle[50:100]

# avg voltage and std for isoelectic area 

mean_iso_first_HB = np.mean(iso_first_HB)
mean_iso_last_HB = np.mean(iso_last_HB)
mean_iso_avg_HB = np.mean(iso_avg_HB)

std_iso_first_HB = np.std(iso_first_HB)
std_iso_last_HB = np.std(iso_last_HB)
std_iso_avg_HB = np.std(iso_avg_HB)

print("mean_iso_first_HB", mean_iso_first_HB)
print("mean_iso_last_HB", mean_iso_last_HB)
print("mean_iso_avg_HB", mean_iso_avg_HB)

print("std_iso_first_HB", std_iso_first_HB)
print("std_iso_last_HB", std_iso_last_HB)
print("std_iso_avg_HB", std_iso_avg_HB)



R_first = np.max(heart_cycles[0])
R_last = np.max(heart_cycles[-1])
R_avg = np.max(average_cycle)
 
print("first SNR", R_first/std_iso_first_HB)
print("last SNR",  R_last/std_iso_last_HB)
print("avarage SNR",  R_avg/std_iso_avg_HB)

SNR_avg_cycles = average_cycle/std_iso_avg_HB
plt.plot(average_cycle)
plt.title('Average Heart Cycle')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.show()

plt.plot(SNR_avg_cycles)
plt.title('SNR Average Heart Cycle')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.show()


SNR_last_cycle = heart_cycles[-1]/std_iso_last_HB
SNR_first_cycle = heart_cycles[0]/std_iso_first_HB

# Plot SNR AVG Heart Cycle
plt.plot(SNR_avg_cycles, label='Average Heart Cycle')

# Plot SNR First Heart Cycle
plt.plot(SNR_first_cycle, label='First Heart Cycle')

# Plot SNR Last Heart Cycle
plt.plot(SNR_last_cycle, label='Last Heart Cycle')

plt.title('SNR Heart Cycles Comparison')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

plt.legend()

plt.show()