import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.stats import ttest_ind


print("*************part1***************")
# Load the ECG signal from the text file
signal =  np.loadtxt('108-a.txt', delimiter='\t', skiprows=1)


# Extract each column of the ECG signal into a separate lead signal
lead_I = signal[:, 0]
lead_II = signal[:, 1]
lead_III = signal[:, 2]


# Apply filtering to each lead signal
fs = 500  # Sampling frequency
nyq = 0.5 * fs
lowcut = 0.5  # Low cutoff frequency 0.5 Hz
highcut = 100  # High cutoff frequency 100 Hz
order = 4  # Filter order
b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
filtered_lead_I = lfilter(b, a, lead_I)
filtered_lead_II = lfilter(b, a, lead_II)
filtered_lead_III = lfilter(b, a, lead_III)

# Create time array
t = np.arange(len(lead_I)) / fs


fig, axs = plt.subplots(3, 1, figsize=(10, 10))
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

plt.xlim(0, 1.5)

# Plot the lead I signal
t = np.arange(lead_I.size) / fs
plt.plot(t, lead_I)
axs[0].set_title('Lead I')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('lead I Zoomed in')
plt.show()

plt.xlim(0, 1.5)

# Plot the lead II signal
t = np.arange(lead_II.size) / fs
plt.plot(t, lead_II)
axs[0].set_title('Lead II')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('lead II Zoomed in')
plt.show()

plt.xlim(0, 1.5)

# Plot the lead III signal
t = np.arange(lead_III.size) / fs
plt.plot(t, lead_III)
axs[0].set_title('Lead III')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('lead III Zoomed in')

plt.show()


# Plot filtered leads
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
axs[0].plot(t, filtered_lead_I, label='Filtered signal')
axs[0].set_title('Filtered Lead I')
axs[0].set_xlabel('Time (sec)')
axs[0].set_ylabel('Amplitude (mV)')
axs[0].legend()


axs[1].plot(t, filtered_lead_II, label='Filtered signal')
axs[1].set_title('Filtered Lead II')
axs[1].set_xlabel('Time (sec)')
axs[1].set_ylabel('Amplitude (mV)')
axs[1].legend()


axs[2].plot(t, filtered_lead_III, label='Filtered signal')
axs[2].set_title('Filtered Lead III')
axs[2].set_xlabel('Time (sec)')
axs[2].set_ylabel('Amplitude (mV)')
axs[2].legend()

plt.tight_layout()
plt.show()

plt.xlim(0, 1.5)

# Plot the lead I signal
t = np.arange(lead_I.size) / fs
plt.plot(t, filtered_lead_I)
axs[0].set_title('Filtered Lead I zoomed in')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('lead I Zoomed in')
plt.show()

plt.xlim(0, 1.5)

# Plot the lead II signal
t = np.arange(lead_II.size) / fs
plt.plot(t, filtered_lead_II)
axs[0].set_title('Filtered Lead II zoomed in')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('lead II Zoomed in')
plt.show()

plt.xlim(0, 1.5)

# Plot the lead III signal
t = np.arange(lead_III.size) / fs
plt.plot(t, filtered_lead_III)
axs[0].set_title('Filtered Lead III zoomed in')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.title('lead III Zoomed in')

plt.show()

def plot_fft(signal, fs, title, ax):
    # Compute the FFT and the frequency bins
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), 1/fs)
    
    # Shift both the fft values and the frequency bins
    fft_vals_shifted = np.fft.fftshift(fft_vals)
    fft_freq_shifted = np.fft.fftshift(fft_freq)
    
    # Plot the FFT on the provided axis
    ax.plot(fft_freq_shifted, np.abs(fft_vals_shifted))
    ax.set_title('FFT of ' + title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')


# Plot for original signals
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
plot_fft(lead_I, fs, 'Lead I', axs[0])
plot_fft(lead_II, fs, 'Lead II', axs[1])
plot_fft(lead_III, fs, 'Lead III', axs[2])
plt.tight_layout()
plt.show()

# Plot for filtered signals
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
plot_fft(filtered_lead_I, fs, 'Filtered Lead I', axs[0])
plot_fft(filtered_lead_II, fs, 'Filtered Lead II', axs[1])
plot_fft(filtered_lead_III, fs, 'Filtered Lead III', axs[2])
plt.tight_layout()
plt.show()


def HR_calculate(r_peaks):
    r_peak_times = r_peaks / fs
    rr_intervals = np.diff(r_peak_times)
    heart_rate_bpm = 60 / rr_intervals
    
    return heart_rate_bpm

    
# Define the function for AF2 QRS detection
def detect_qrs_af2(ecg_signal, fs):
    
    # 1. Set amplitude threshold
    amplitude_threshold = 0.4 * max(ecg_signal)

    # 2. Rectify the signal
    YO = np.where(ecg_signal >= 0, ecg_signal, -ecg_signal)

    # 3. Apply low level clipper
    Y1 = np.where(YO >= amplitude_threshold, YO, 0)
    

    # 4. Calculate the first derivative of the clipped signal
    Y2 = np.diff(Y1, n=1)
    Y2 = np.append(Y2, 0)  # Append a zero to maintain the array size

    # 5. Detect QRS (R-peaks in this case)
    QRS_threshold = 0.7 * max(Y2)  # Set a relative threshold based on the max of the first derivative
    r_peaks = np.where(Y2 > QRS_threshold)[0]
    
    
    # Adjust R-peaks to the nearest local maximum
    for i, r_peak in enumerate(r_peaks):
        # Search window to find the local maximum around the detected peak
        window = ecg_signal[max(0, r_peak - int(fs*0.050)): min(len(ecg_signal), r_peak + int(fs*0.050))]
        if len(window) == 0:
            continue
        local_max = np.argmax(window)
        r_peaks[i] = r_peak - int(fs*0.050) + local_max if window.any() else r_peak

    return r_peaks


# Apply the QRS detection algorithm
r_peaks_indices_I = detect_qrs_af2(lead_I, fs)

# Plotting the ECG signal and marking the detected R-peaks
plt.figure(figsize=(20, 6))
plt.plot(t, lead_I, label='ECG signal')
plt.scatter(t[r_peaks_indices_I], lead_I[r_peaks_indices_I], color='red', s=50, label='R peaks detection', zorder=3)
plt.title('Lead I QRS Detection')
plt.xlabel('Time [sec]')
plt.ylabel('Lead I [mV]')
plt.legend()
plt.grid(True)
plt.show()


r_peaks_indices_II = detect_qrs_af2(lead_II, fs)

# Plotting the ECG signal and marking the detected R-peaks
plt.figure(figsize=(20, 6))
plt.plot(t, lead_II, label='ECG signal')
plt.scatter(t[r_peaks_indices_II], lead_II[r_peaks_indices_II], color='red', s=50, label='R peaks detection', zorder=3)
plt.title('Lead II Seated QRS Detection')
plt.xlabel('Time [sec]')
plt.ylabel('Lead II [mV]')
plt.legend()
plt.grid(True)
plt.show()

r_peaks_indices_III = detect_qrs_af2(lead_II, fs)

# Plotting the ECG signal and marking the detected R-peaks
plt.figure(figsize=(20, 6))
plt.plot(t, lead_III, label='ECG signal')
plt.scatter(t[r_peaks_indices_III], lead_III[r_peaks_indices_III], color='red', s=50, label='R peaks detection', zorder=3)
plt.title('Lead III Seated QRS Detection')
plt.xlabel('Time [sec]')
plt.ylabel('Lead III [mV]')
plt.legend()
plt.grid(True)
plt.show()


seated_end = 30 * fs  # 30 seconds
standing_start = 30 * fs  # 30 seconds
standing_end = 40 * fs  # 40 seconds
heavy_breathing_start = 40 * fs  # 40 seconds
heavy_breathing_end = 50 * fs  # 50 seconds

# Segment the ECG signal for each activity phase
lead_I_seated = lead_I[:seated_end]
lead_I_standing = lead_I[standing_start:standing_end]
lead_I_heavy_breathing = lead_I[heavy_breathing_start:heavy_breathing_end]

# Adjusted time arrays for plotting
t_seated = np.arange(len(lead_I_seated)) / fs
t_standing = np.arange(len(lead_I_standing)) / fs + 30  # Start from 30s
t_heavy_breathing = np.arange(len(lead_I_heavy_breathing)) / fs + 40  # Start from 40s

# Function to detect R-peaks and calculate heart rate, adjusted for segment analysis
def analyze_segment(ecg_segment, fs, segment_name):
    r_peaks_indices = detect_qrs_af2(ecg_segment, fs)
    HR_values = HR_calculate(r_peaks_indices)
    
    
    # Plotting the segment and R-peaks
    plt.figure(figsize=(10, 4))
    t_segment = np.arange(len(ecg_segment)) / fs
    if segment_name == "standing":
        t_segment += 30
    elif segment_name == "heavy breathing":
        t_segment += 40
    plt.plot(t_segment, ecg_segment, label='ECG signal')
    plt.scatter(t_segment[r_peaks_indices], ecg_segment[r_peaks_indices], color='red', s=50, label='R peaks', zorder=3)
    plt.title(f'ECG lead I: {segment_name}')
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude [mV]')
    plt.legend()
    plt.grid(True)
    plt.show()
    return HR_values

# Analyze each segment
seated_HR_values = analyze_segment(lead_I_seated, fs, "seated")
standing_HR_values = analyze_segment(lead_I_standing, fs, "standing")
heavy_breathing_HR_values = analyze_segment(lead_I_heavy_breathing, fs, "heavy breathing")

average_HR_seating = np.mean(seated_HR_values)
std_HR_seating = np.std(seated_HR_values)
print("mean HR seating", average_HR_seating)
print("std HR seating", std_HR_seating)
average_HR_standing = np.mean(standing_HR_values)
std_HR_standing = np.std(standing_HR_values)
print("mean HR standing", average_HR_standing)
print("std HR standing", std_HR_standing)
average_HR_breathing = np.mean(heavy_breathing_HR_values)
std_HR_breathing = np.std(heavy_breathing_HR_values)
print("mean HR heavy breathing", average_HR_breathing)
print("std HR heavy breathing", std_HR_breathing)


# t- test is measing.
t_stat, p_value = ttest_ind(heavy_breathing_HR_values, standing_HR_values)

print(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")
if p_value < 0.05:
    print("There is a statistically significant difference between the HR of heavy breathing and standing.")
else:
    print("There is no statistically significant difference between the HR of heavy breathing and standing.")
    
