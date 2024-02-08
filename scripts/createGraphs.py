from tools import pickle2dict
import numpy as np
import matplotlib.pyplot as plt

def low_pass_filter(signal, cutoff_frequency, sampling_rate):
    # Calculate the normalized cutoff frequency
    nyquist_frequency = 0.5 * sampling_rate
    normalized_cutoff = cutoff_frequency / nyquist_frequency

    # Design a low-pass filter kernel (FIR filter)
    num_taps = 101  # Adjust the number of taps as needed
    filter_kernel = np.sinc(2 * normalized_cutoff * (np.arange(num_taps) - (num_taps - 1) / 2))

    # Normalize the filter kernel to have unity gain at zero frequency
    filter_kernel /= np.sum(filter_kernel)

    # Apply the filter using convolution
    filtered_signal = np.convolve(signal, filter_kernel, mode='same')

    return filtered_signal



# Generate a sample signal (e.g., a sine wave)
sampling_rate = 1000  # in Hz
time = np.arange(0, 1, 1/sampling_rate)
t_s = np.arange(0, 1000)
#signal = np.sin(2 * np.pi * 5 * time) + 0.5 * np.sin(2 * np.pi * 20 * time)

#load dict
dict_data = pickle2dict("exp_target.pickle")
signal = dict_data["x_position"][:-1]
target_signal = dict_data["target_traj"][:-1]
target_rewards = dict_data['target_r'][:-1]

print("signal shape: ", np.shape(signal))
print("target_signal shape: ", target_signal)
# Apply the low-pass filter
cutoff_frequency = 10  # in Hz
filtered_signal = low_pass_filter(signal, cutoff_frequency, sampling_rate)
filtered_target_signal = low_pass_filter(target_signal, cutoff_frequency, sampling_rate)
# 

#load dict
#dict_data = pickle2dict("my_dict.pickle")
print("dict data: ", dict_data.keys())
print("total_reward: ", dict_data["total_reward"])
print("rollouts: ", dict_data["rollouts"])
print("x_position: ", dict_data["x_position"])
# Plot the original and filtered signals
rollouts = dict_data["rollouts"]
x_vel = np.array(rollouts[:-1])
x_vel = x_vel[:,8]
print("x_vel shape: ", np.shape(x_vel))
print("rollouts shape: ", np.shape(rollouts) )
plt.figure(figsize=(10, 6))
plt.grid(True)
plt.title('Locomotion with Cost Function Guidance', fontsize=20)
plt.plot(t_s[:-40], signal[:-40], color="lightgray")
plt.plot(t_s[:-40], filtered_signal[:-40], label='Velocity (m/s)')
plt.plot(t_s[:-40], target_signal[:-40], color="powderblue")
plt.plot(t_s[:-40], filtered_target_signal[:-40], label='Target velocity (m/s)')
plt.xlabel('Time step (t)', fontsize=20)
plt.ylabel('Velocity (m/s)', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.figure(figsize=(10, 6))
plt.title('Locomotion with Cost Function Guidance', fontsize=20)
plt.grid(True)
plt.plot(t_s, dict_data["total_reward"][:-1])
plt.plot(t_s, target_rewards)
plt.legend(fontsize=16)
plt.xlabel('Time step (t)', fontsize=20)
plt.ylabel('Total reward', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
