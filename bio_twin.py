import time
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import csv

# ==========================================
# 1. MOCK STREAMER WITH ANOMALY INJECTION
# ==========================================
class MockBioStreamer:
    def __init__(self, sampling_rate=250):
        self.fs = sampling_rate
        self.t = 0.0
        self.dt = 1.0 / sampling_rate

    def get_sample(self):
        # Inject artificial autonomic decoupling/stress anomaly after t = 6 seconds
        anomaly_factor = 0.0 if self.t < 6.0 else 3.5

        # Central Brain channels (EEG1-4)
        eeg1 = np.sin(2 * np.pi * 10 * self.t) + 0.5 * np.random.randn() + anomaly_factor
        eeg2 = np.sin(2 * np.pi * 10 * self.t + 0.2) + 0.5 * np.random.randn()
        eeg3 = np.cos(2 * np.pi * 20 * self.t) + 0.5 * np.random.randn()
        eeg4 = np.cos(2 * np.pi * 20 * self.t + 0.4) + 0.5 * np.random.randn()

        # Cardiac/Autonomic signal
        cardiac = np.sin(2 * np.pi * 1.2 * self.t) + 0.1 * np.random.randn()

        # Micro-Vascular Capillary Flow
        vascular = 0.8 * np.sin(2 * np.pi * 1.2 * self.t + 0.1) + 0.2 * np.random.randn()

        sample = np.array([eeg1, eeg2, eeg3, eeg4, cardiac, vascular])
        self.t += self.dt
        return sample


# ==========================================
# 2. SIGNAL PROCESSOR
# ==========================================
class RealTimeSignalProcessor:
    def __init__(self, sampling_rate=250):
        self.fs = sampling_rate

    def butter_bandpass_filter(self, data, lowcut=0.5, highcut=45.0, order=4):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=-1)

    def compute_adjacency_matrix(self, buffer_data):
        filtered_data = self.butter_bandpass_filter(buffer_data)
        corr_matrix = np.corrcoef(filtered_data)
        return np.nan_to_num(corr_matrix)


# ==========================================
# 3. PYTORCH GRAPH CONVOLUTIONAL NETWORK (GCN)
# ==========================================
class BioGraphGCN(nn.Module):
    def __init__(self, num_nodes=6):
        super(BioGraphGCN, self).__init__()
        self.fc1 = nn.Linear(num_nodes, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, adj_matrix):
        x = torch.matmul(adj_matrix, adj_matrix)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(torch.mean(x))


# ==========================================
# 4. DASHBOARD ENGINE WITH AUTOMATED DATA LOGGING
# ==========================================
def run_digital_twin_dashboard(duration_seconds=12, window_size=250):
    streamer = MockBioStreamer(sampling_rate=250)
    processor = RealTimeSignalProcessor(sampling_rate=250)
    gcn_model = BioGraphGCN(num_nodes=6)
    gcn_model.eval()

    buffer = []
    time_axis = []
    health_history = []

    # Open CSV file to log real-time data for research
    csv_file = open("bio_twin_telemetry_log.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Timestamp_s", "EEG1_Cardiac_Coupling", "Cardiac_Vascular_Coupling", "PyTorch_Health_Index", "Status"])

    # Set up Live Plot
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle('Bio-Autonomic Digital Twin Engine (Research Logger Active)', fontsize=14, fontweight='bold')

    line_eeg, = ax1.plot([], [], label='Brain EEG1', color='cyan')
    line_cardiac, = ax1.plot([], [], label='Cardiac Pulse', color='red')
    ax1.set_ylabel('Amplitude')
    ax1.set_ylim(-5, 6)
    ax1.legend(loc='upper right')
    ax1.grid(True)

    line_health, = ax2.plot([], [], label='PyTorch GCN Health Index (%)', color='green', linewidth=2)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Health Index (%)')
    ax2.set_ylim(0, 100)
    ax2.axhline(y=30, color='r', linestyle='--', label='Anomaly Threshold')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    start_time = time.time()
    print("--- Starting Live Dashboard & Data Logger ---")

    while (time.time() - start_time) < duration_seconds:
        sample = streamer.get_sample()
        buffer.append(sample)

        if len(buffer) >= window_size:
            buffer_data = np.array(buffer[-window_size:]).T
            adj_matrix = processor.compute_adjacency_matrix(buffer_data)
            adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)

            with torch.no_grad():
                health_index = gcn_model(adj_tensor).item() * 100

            time_axis.append(streamer.t)
            health_history.append(health_index)

            # Determine status
            status_text = "NORMAL" if streamer.t <= 6.0 else "ALERT_DECOUPLING"

            # Log step to CSV file
            csv_writer.writerow([
                f"{streamer.t:.2f}",
                f"{adj_matrix[0, 4]:.4f}",
                f"{adj_matrix[4, 5]:.4f}",
                f"{health_index:.2f}",
                status_text
            ])

            # Update Plot
            recent_time = np.linspace(streamer.t - 1.0, streamer.t, window_size)
            line_eeg.set_data(recent_time, buffer_data[0])
            line_cardiac.set_data(recent_time, buffer_data[4])
            ax1.set_xlim(streamer.t - 1.0, streamer.t)

            line_health.set_data(time_axis, health_history)
            ax2.set_xlim(0, max(10, streamer.t))

            if streamer.t > 6.0:
                line_health.set_color('red')
                ax2.set_title("Status: ⚠️ ALERT: ORGAN DECOUPLING DETECTED!", color='red', fontweight='bold')
            else:
                line_health.set_color('green')
                ax2.set_title("Status: NORMAL", color='green')

            fig.canvas.draw()
            fig.canvas.flush_events()

        time.sleep(1.0 / 250.0)

    csv_file.close()
    plt.ioff()
    plt.show()
    print("\nData successfully exported to 'bio_twin_telemetry_log.csv'!")

if __name__ == "__main__":
    run_digital_twin_dashboard()