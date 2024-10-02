
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


print("*************part3***************")
data = np.loadtxt('108-c.txt', skiprows=1)

stethoscope = data[:, 0]
ecg = data[:, 1]

fs = 500.0  
dt = 1/fs
N = len(stethoscope)
t = np.arange(0, N*dt, dt)
lowcut = 0.5
highcut = 100.0
nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq

# filter creation
b, a = butter(3, [low, high], btype='band')

# filtering
stethoscope_filtered = filtfilt(b, a, stethoscope)
ecg_filtered = filtfilt(b, a, ecg)


def calculate_bpm(peaks, fs):
    intervals_in_samples = np.diff(peaks)
    intervals_in_seconds = intervals_in_samples / fs
    
    # Calculate BPM
    bpm = 60 / intervals_in_seconds
    
    return bpm

def find_hs_peaks(signal, peak_indices, window_size, threshold):
    peaks = []
    peak_values = []
    for i in peak_indices:
        window = signal[i:i + window_size]
        if window.size > 0:
            max_val = np.max(window)
            
            max_val = np.max(window)
            if max_val >= threshold:
                max_idx = np.argmax(window)
                peaks.append(i + max_idx)
                peak_values.append(max_val)
    return peaks, peak_values


def detect_r_peaks(ecg_signal, fs):
    amplitude_threshold = 0.4 * np.max(ecg_signal)

    YO = np.where(ecg_signal >= 0, ecg_signal, -ecg_signal)

    # Apply low level clipper
    Y1 = np.where(YO >= amplitude_threshold, YO, 0)

    # Calculate the first derivative of the clipped signal
    Y2 = np.diff(Y1, n=1)
    Y2 = np.append(Y2, 0)  

    # Detect QRS (R-peaks in this case)
    QRS_threshold = 0.7 * np.max(Y2)
    r_peaks_indices = np.where(Y2 > QRS_threshold)[0]

    # Adjust R-peaks to the nearest local maximum
    for i, r_peak in enumerate(r_peaks_indices):
        # Search window to find the local maximum around the detected peak
        window = ecg_signal[max(0, r_peak - int(fs*0.050)): min(len(ecg_signal), r_peak + int(fs*0.050))]
        if len(window) == 0:
            continue
        local_max = np.argmax(window)
        r_peaks_indices[i] = r_peak - int(fs*0.050) + local_max if window.any() else r_peak

    # Create a binary array for R-peaks
    R_Peaks = np.zeros_like(ecg_signal)
    R_Peaks[r_peaks_indices] = 1

    return R_Peaks

def detect_heart_sounds(hs_signal, fs):

    # Normalize HS signal
    max_abs_value = np.max(np.abs(hs_signal))
    hs_norm = hs_signal / max_abs_value
    N = len(hs_signal)
    Ehs = np.zeros_like(hs_signal)
    for i in range (1, N):
        Ehs[i] = - (hs_norm[i]**2 * np.log(hs_norm[i]**2))
    
    time_gate = np.zeros_like(Ehs)
    
    # Calculate normalized average three-order Shannon energy
    Pha = (Ehs - np.mean(Ehs)) / np.std(Ehs)
    
    # Set soft threshold based on envelope amplitude
    Th = 0.2 * np.max(Pha)  # Replace with adaptive thresholding if needed
    
    
    for i in range (1, len(Ehs)):
        if Ehs[i] > Th: 
            time_gate[i] = 1
        else:
            time_gate[i] = 0
     
    R_Peaks = detect_r_peaks(ecg_filtered, fs)
    
    R_peak = np.where(R_Peaks)[0]
    
    s1_indexs = np.zeros(len(Ehs))
    s1_vals = np.zeros(len(Ehs))
    dt_R_to_s1= np.zeros(len(R_peak))
    
    s1_peaks, s1_values = find_hs_peaks(Ehs, R_peak, 100, 0)
    for i, peak in enumerate(s1_peaks):
        s1_indexs[peak] = 1
        s1_vals[peak] = s1_values[i]
        dt_R_to_s1[i] = (peak - R_peak[i]) * dt
        
    
    s2_indexs = np.zeros(len(Ehs))
    s2_vals = np.zeros(len(Ehs))
    dt_R_to_s2 = np.zeros(len(R_peak))
    
    s2_peaks, s2_values = find_hs_peaks(Ehs, R_peak + 100, 150, 0)
    for i, peak in enumerate(s2_peaks):
        s2_indexs[peak] = 1
        s2_vals[peak] = s2_values[i]
        dt_R_to_s2[i] = (peak - R_peak[i]) * dt

    print("mean dt_R_to_s2", np.mean(dt_R_to_s2))
    print("mean dt_R_to_s1", np.mean(dt_R_to_s1))
    print("std dt_R_to_s2", np.std(dt_R_to_s2))
    print("std dt_R_to_s1", np.std(dt_R_to_s1))


    s1 = np.full_like(s1_indexs, np.nan)
    s1[s1_indexs == 1] = stethoscope_filtered.max() + 0.3*stethoscope_filtered.max()
    
    s2 = np.full_like(s2_indexs, np.nan)
    s2[s2_indexs == 1] = stethoscope_filtered.max() + 0.3*stethoscope_filtered.max()

    return s1, s2

        
fs = 500
s1_peaks, s2_peaks = detect_heart_sounds(stethoscope_filtered, fs)

t = np.arange(len(stethoscope_filtered)) / fs

plt.figure('Rest S1,S2 Detection')

# First subplot
plt.figure(figsize=(15, 7))
plt.subplot(2, 1, 1)
plt.plot(t, stethoscope_filtered)
plt.xlabel('Time [sec]')
plt.ylabel('Voltage [mV]')
plt.title('S1,S2 Detection-Rest Part')
plt.xlim([0, 20])
plt.scatter(t, s1_peaks, color='gold')
plt.scatter(t, s2_peaks, color='violet')
plt.legend(['Stethoscope', 'S1', 'S2'], loc='upper right')

#Second subplot
plt.figure(figsize=(15, 7))
plt.subplot(2, 1, 2)
plt.plot(t, ecg_filtered)
plt.xlabel('Time [sec]')
plt.ylabel('Voltage [mV]')
plt.title('ECG Signal-Rest Part')
plt.xlim([0, 20])
plt.legend(['ECG'])

plt.show()

plt.figure('Past activity S1,S2 Detection')

# First subplot
plt.figure(figsize=(15, 7))
plt.subplot(2, 1, 1)
plt.plot(t, stethoscope_filtered)
plt.xlabel('Time [sec]')
plt.ylabel('Voltage [mV]')
plt.title('S1,S2 Detection-post activity Part')
plt.xlim([20, 40])
plt.scatter(t, s1_peaks, color='gold')
plt.scatter(t, s2_peaks, color='violet')
plt.legend(['Stethoscope', 'S1', 'S2'], loc='upper right')


# Second subplot
plt.figure(figsize=(15, 7))
plt.subplot(2, 1, 2)
plt.plot(t, ecg_filtered)
plt.xlabel('Time [sec]')
plt.ylabel('Voltage [mV]')
plt.title('ECG Signal-Post active Part')
plt.xlim([20, 40])
plt.legend(['ECG'])

plt.show()


samples_for_20_sec = 20 * fs

ecg_rest = ecg_filtered[:samples_for_20_sec]
ecg_post_active = ecg_filtered[samples_for_20_sec:2*samples_for_20_sec]

stethoscope_rest = stethoscope_filtered[:samples_for_20_sec]
stethoscope_post_active = stethoscope_filtered[samples_for_20_sec:2*samples_for_20_sec]


print("\nactive\n")
s1_peaks_rest, s2_peaks_rest = detect_heart_sounds(stethoscope_rest, fs)

np.set_printoptions(threshold=np.inf)
s1_indexes = np.where(~np.isnan(s1_peaks_rest))[0]
s2_indexes = np.where(~np.isnan(s2_peaks_rest))[0]

dt_s1_to_s2 = (s2_indexes - s1_indexes) / fs
print("mean dt_s1_to_s2",np.mean(dt_s1_to_s2))
print("std dt_s1_to_s2", np.std(dt_s1_to_s2))

s2_indexes_shift = s2_indexes[:-1]
s1_indexes_shift = s1_indexes[1:]

dt_s2_to_s1 = (s1_indexes_shift - s2_indexes_shift) / fs
print("mean dt_s2_to_s1",np.mean(dt_s2_to_s1))
print("std dt_s2_to_s1", np.std(dt_s2_to_s1))

print("\nrest\n")
s1_peaks_post_active, s2_peaks_post_active = detect_heart_sounds(stethoscope_post_active, fs)

np.set_printoptions(threshold=np.inf)
s1_indexes_post = np.where(~np.isnan(s1_peaks_post_active))[0]
s2_indexes_post = np.where(~np.isnan(s2_peaks_post_active))[0]

dt_s1_to_s2 = (s2_indexes_post - s1_indexes_post) / fs
print("mean dt_s1_to_s2",np.mean(dt_s1_to_s2))
print("std dt_s1_to_s2", np.std(dt_s1_to_s2))

s2_indexes_shift = s2_indexes_post[:-1]
s1_indexes_shift = s1_indexes_post[1:]

dt_s2_to_s1 = (s1_indexes_shift - s2_indexes_shift) / fs
print("mean dt_s2_to_s1",np.mean(dt_s2_to_s1))
print("std dt_s2_to_s1", np.std(dt_s2_to_s1))


R_Peaks_rest = detect_r_peaks(ecg_rest, fs)
R_Peaks_post_active = detect_r_peaks(ecg_post_active, fs)

R_Peaks_rest = np.where(R_Peaks_rest)[0]
R_Peaks_post_active = np.where(R_Peaks_post_active)[0]

s1_peaks_rest = np.where(s1_peaks_rest)[0]
s2_peaks_rest = np.where(s2_peaks_rest)[0]
s1_peaks_post_active = np.where(s1_peaks_post_active)[0]
s2_peaks_post_active = np.where(s2_peaks_post_active)[0]

BPM_rest = calculate_bpm(R_Peaks_rest, fs)
BPM_post_active = calculate_bpm(R_Peaks_post_active, fs)

avg_BPM_rest = np.mean(BPM_rest)
avg_BPM_post_active = np.mean(BPM_post_active)
std_BPM_rest = np.std(BPM_rest)
std_BPM_post_active = np.std(BPM_post_active)

print("\nBPM\n")
print("avg BPM rest", avg_BPM_rest)
print("avg BPM post active", avg_BPM_post_active)
print("std BPM rest", std_BPM_rest)
print("std BPM post active", std_BPM_post_active)

