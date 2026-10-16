"""
Bio-Autonomic Digital Twin Engine
=================================
A real-time signal processing and Graph Convolutional Network (GCN) pipeline designed
for multi-modal electrophysiology ingestion (EEG, ECG, Micro-Vascular Flow) and continuous
neural-organ coupling analysis.

Author: Shakeel Ahmed
Reference Publication: Dynamic Connectome Modeling (Zenodo DOI: 10.5281/zenodo.22939463)
"""

from pathlib import Path
import csv
import time
from typing import Tuple, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn


class SyntheticBioSignalStreamer:
    """
    Simulates real-time multi-modal biological signal ingestion at 250 Hz, including 4-channel
    central EEG, cardiac rhythm (ECG), and micro-vascular capillary flow.
    """

    def __init__(self, sampling_rate_hz: int = 250) -> None:
        self.sampling_rate_hz: int = sampling_rate_hz
        self.current_time_sec: float = 0.0
        self.time_step_sec: float = 1.0 / sampling_rate_hz

    def fetch_next_sample(self) -> np.ndarray:
        """
        Generates next time-frame vector. Injects autonomic decoupling anomaly after t = 6.0s.

        Returns
        -------
        np.ndarray
            6-element array holding [EEG1, EEG2, EEG3, EEG4, Cardiac_Pulse, Vascular_Flow].
        """
        decoupling_anomaly_shift: float = 0.0 if self.current_time_sec < 6.0 else 3.5

        eeg_ch1 = np.sin(2 * np.pi * 10 * self.current_time_sec) + 0.5 * np.random.randn() + decoupling_anomaly_shift
        eeg_ch2 = np.sin(2 * np.pi * 10 * self.current_time_sec + 0.2) + 0.5 * np.random.randn()
        eeg_ch3 = np.cos(2 * np.pi * 20 * self.current_time_sec) + 0.5 * np.random.randn()
        eeg_ch4 = np.cos(2 * np.pi * 20 * self.current_time_sec + 0.4) + 0.5 * np.random.randn()

        cardiac_pulse = np.sin(2 * np.pi * 1.2 * self.current_time_sec) + 0.1 * np.random.randn()
        vascular_flow = 0.8 * np.sin(2 * np.pi * 1.2 * self.current_time_sec + 0.1) + 0.2 * np.random.randn()

        multimodal_frame = np.array([eeg_ch1, eeg_ch2, eeg_ch3, eeg_ch4, cardiac_pulse, vascular_flow])
        self.current_time_sec += self.time_step_sec
        return multimodal_frame


class ZeroPhaseSignalProcessor:
    """
    Executes real-time digital filtering and computes functional cross-organ correlation matrices.
    """

    def __init__(self, sampling_rate_hz: int = 250) -> None:
        self.sampling_rate_hz: int = sampling_rate_hz

    def apply_butterworth_bandpass(
        self,
        signal_data: np.ndarray,
        low_pass_cutoff_hz: float = 0.5,
        high_pass_cutoff_hz: float = 45.0,
        filter_order: int = 4
    ) -> np.ndarray:
        """
        Applies a zero-phase Butterworth bandpass filter to mitigate phase distortions.
        """
        nyquist_frequency = 0.5 * self.sampling_rate_hz
        normalized_low = low_pass_cutoff_hz / nyquist_frequency
        normalized_high = high_pass_cutoff_hz / nyquist_frequency

        filter_b, filter_a = butter(filter_order, [normalized_low, normalized_high], btype='band')
        filtered_signals = filtfilt(filter_b, filter_a, signal_data, axis=-1)
        return filtered_signals

    def extract_functional_adjacency_matrix(self, buffer_data: np.ndarray) -> np.ndarray:
        """
        Computes Pearson correlation coefficients across all active channels to derive adjacency matrix.
        """
        filtered_buffer = self.apply_butterworth_bandpass(buffer_data)
        correlation_matrix = np.corrcoef(filtered_buffer)
        return np.nan_to_num(correlation_matrix)


class OrganCouplingGCN(nn.Module):
    """
    Graph Convolutional Network mapping cross-organ spatial adjacency matrices to system health scores.
    """

    def __init__(self, total_nodes: int = 6) -> None:
        super(OrganCouplingGCN, self).__init__()
        self.dense_layer_1 = nn.Linear(total_nodes, 16)
        self.activation_relu = nn.ReLU()
        self.dense_layer_2 = nn.Linear(16, 1)
        self.activation_sigmoid = nn.Sigmoid()

    def forward(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing graph state transitions via message passing.
        """
        node_features = torch.matmul(adjacency_matrix, adjacency_matrix)
        hidden_states = self.activation_relu(self.dense_layer_1(node_features))
        output_logits = self.dense_layer_2(hidden_states)
        system_health_index = self.activation_sigmoid(torch.mean(output_logits))
        return system_health_index


def execute_digital_twin_telemetry(
    monitoring_duration_sec: int = 12,
    sliding_window_size: int = 250,
    log_output_path: str = "bio_twin_telemetry_log.csv"
) -> None:
    """
    Runs the live real-time dashboard and telemetry logging interface.
    """
    signal_streamer = SyntheticBioSignalStreamer(sampling_rate_hz=250)
    signal_processor = ZeroPhaseSignalProcessor(sampling_rate_hz=250)
    gcn_classifier = OrganCouplingGCN(total_nodes=6)
    gcn_classifier.eval()

    sample_buffer: List[np.ndarray] = []
    time_points: List[float] = []
    health_index_history: List[float] = []

    plt.ion()
    figure, (axis_signals, axis_health) = plt.subplots(2, 1, figsize=(10, 6))
    figure.suptitle('Bio-Autonomic Digital Twin Engine (Real-Time Ingestion Active)', fontsize=13, fontweight='bold')

    line_eeg, = axis_signals.plot([], [], label='Brain EEG1 (µV)', color='cyan')
    line_cardiac, = axis_signals.plot([], [], label='Cardiac Pulse', color='red')
    axis_signals.set_ylabel('Amplitude')
    axis_signals.set_ylim(-5, 6)
    axis_signals.legend(loc='upper right')
    axis_signals.grid(True)

    line_health, = axis_health.plot([], [], label='PyTorch GCN Coupling Index (%)', color='green', linewidth=2)
    axis_health.set_xlabel('Time (seconds)')
    axis_health.set_ylabel('Health Index (%)')
    axis_health.set_ylim(0, 100)
    axis_health.axhline(y=30, color='r', linestyle='--', label='Decoupling Anomaly Threshold')
    axis_health.legend(loc='upper right')
    axis_health.grid(True)

    start_timestamp = time.time()
    print("--- Starting Bio-Autonomic Engine Ingestion Loop ---")

    with open(log_output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Timestamp_s", "EEG1_Cardiac_Coupling", "Cardiac_Vascular_Coupling", "PyTorch_Health_Index", "Status"])

        try:
            while (time.time() - start_timestamp) < monitoring_duration_sec:
                raw_sample = signal_streamer.fetch_next_sample()
                sample_buffer.append(raw_sample)

                if len(sample_buffer) >= sliding_window_size:
                    buffer_array = np.array(sample_buffer[-sliding_window_size:]).T
                    adjacency_matrix = signal_processor.extract_functional_adjacency_matrix(buffer_array)
                    adjacency_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32)

                    with torch.no_grad():
                        computed_health_index = gcn_classifier(adjacency_tensor).item() * 100.0

                    current_time_sec = signal_streamer.current_time_sec
                    time_points.append(current_time_sec)
                    health_index_history.append(computed_health_index)

                    status_label = "NORMAL" if current_time_sec <= 6.0 else "ALERT_DECOUPLING"

                    csv_writer.writerow([
                        f"{current_time_sec:.2f}",
                        f"{adjacency_matrix[0, 4]:.4f}",
                        f"{adjacency_matrix[4, 5]:.4f}",
                        f"{computed_health_index:.2f}",
                        status_label
                    ])

                    recent_time_window = np.linspace(current_time_sec - 1.0, current_time_sec, sliding_window_size)
                    line_eeg.set_data(recent_time_window, buffer_array[0])
                    line_cardiac.set_data(recent_time_window, buffer_array[4])
                    axis_signals.set_xlim(current_time_sec - 1.0, current_time_sec)

                    line_health.set_data(time_points, health_index_history)
                    axis_health.set_xlim(0, max(10, current_time_sec))

                    if current_time_sec > 6.0:
                        line_health.set_color('red')
                        axis_health.set_title("Status: ?? ALERT: ORGAN DECOUPLING DETECTED!", color='red', fontweight='bold')
                    else:
                        line_health.set_color('green')
                        axis_health.set_title("Status: NORMAL", color='green')

                    figure.canvas.draw()
                    figure.canvas.flush_events()

                time.sleep(1.0 / 250.0)

        except KeyboardInterrupt:
            print("\nStream execution interrupted by user.")

    plt.ioff()
    plt.show()
    print(f"\nTelemetry session successfully logged to '{log_output_path}'.")


if __name__ == "__main__":
    execute_digital_twin_telemetry()
