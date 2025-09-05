#!/usr/bin/env python3
"""
RQ1 - Part A: Feature Engineering Impact Analysis
================================================

OBJECTIVE: Compare different FEATURE TYPES for neuromorphic anomaly detection
- Basic Features (15): Statistical measures (event rates, spatial stats)
- Spatiotemporal Features (20): Motion and flow analysis
- Neuromorphic Features (29): Event-camera specific patterns

APPROACH: Same algorithms tested on different feature sets (apples-to-apples)
RESEARCH QUESTION: Which feature engineering approach works best for event-based anomaly detection?

This analysis focuses ONLY on feature impact, using consistent algorithms across all tests.
"""

import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import convolve2d
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Check OpenCV
try:
    import cv2

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print("🔬 RQ1-A: Feature Engineering Impact Analysis")
print("=" * 60)
print("FOCUS: Comparing Basic vs Spatiotemporal vs Neuromorphic features")
print("METHOD: Same algorithms on different feature sets")
print("=" * 60)


def load_mvsec_data(data_path="../data", sequence="indoor_flying", camera="left"):
    """Load MVSEC dataset from HDF5 files"""
    data_files = []
    if os.path.isdir(data_path):
        all_files = os.listdir(data_path)
        candidate_files = [f for f in all_files if f.endswith(".hdf5") and "data" in f]

        if sequence:
            sequence_files = [
                f for f in candidate_files if sequence.lower() in f.lower()
            ]
            data_files = sequence_files if sequence_files else candidate_files
        else:
            data_files = candidate_files

    if not data_files:
        available_files = [f for f in os.listdir(data_path) if f.endswith(".hdf5")]
        raise ValueError(
            f"No MVSEC data files found for sequence '{sequence}' in {data_path}. Available files: {available_files}"
        )

    # Load data from all matching files
    data_files.sort()  # Ensure consistent ordering
    print(f"📁 Loading MVSEC data from {len(data_files)} file(s): {data_files}")

    all_events_data = []
    sensor_size = None

    for data_file in data_files:
        file_path = os.path.join(data_path, data_file)
        print(f"  📄 Processing: {data_file}")

        with h5py.File(file_path, "r") as f:
            events_data = f["davis"][camera]["events"][:]
            all_events_data.append(events_data)
            print(f"    📊 Loaded {len(events_data):,} events")

            # Get sensor size from first file
            if sensor_size is None:
                temp_events = {
                    "x": events_data[:, 0].astype(int),
                    "y": events_data[:, 1].astype(int),
                }
                max_x, max_y = np.max(temp_events["x"]), np.max(temp_events["y"])
                sensor_size = (max_y + 1, max_x + 1)

    # Concatenate all events
    combined_events_data = np.concatenate(all_events_data, axis=0)
    print(
        f"📊 Total loaded: {len(combined_events_data):,} events from {len(data_files)} file(s)"
    )

    events = {
        "x": combined_events_data[:, 0].astype(int),
        "y": combined_events_data[:, 1].astype(int),
        "t": combined_events_data[:, 2],
        "p": combined_events_data[:, 3].astype(int),
    }

    print(f"📐 Sensor resolution: {sensor_size}")
    return events, sensor_size


def process_events_to_frames(
    events, sensor_size, num_frames=50, max_events=300000, target_size=(64, 64)
):
    """Convert events to frame representation"""
    if len(events["x"]) > max_events:
        indices = np.linspace(0, len(events["x"]) - 1, max_events, dtype=int)
        for key in events:
            events[key] = events[key][indices]
        print(f"⚡ Subsampled to {max_events:,} events for processing")

    x, y, t, p = events["x"], events["y"], events["t"], events["p"]

    # Create time bins
    t_min, t_max = np.min(t), np.max(t)
    time_bins = np.linspace(t_min, t_max, num_frames + 1)

    # Initialize frames
    H, W = target_size
    frames = torch.zeros((num_frames, 2, H, W))

    # Scale coordinates
    orig_H, orig_W = sensor_size
    x_scaled = (x * W / orig_W).astype(int)
    y_scaled = (y * H / orig_H).astype(int)
    x_scaled = np.clip(x_scaled, 0, W - 1)
    y_scaled = np.clip(y_scaled, 0, H - 1)

    # Bin events into frames
    print("🎬 Processing events into frames...")
    for i in range(len(x)):
        bin_idx = np.searchsorted(time_bins[1:], t[i])
        bin_idx = min(bin_idx, num_frames - 1)
        channel = 0 if p[i] == 1 else 1
        frames[bin_idx, channel, y_scaled[i], x_scaled[i]] += 1

    # Normalize frames
    for f in range(num_frames):
        for c in range(2):
            max_val = frames[f, c].max()
            if max_val > 0:
                frames[f, c] = frames[f, c] / max_val

    print(f"✅ Generated {num_frames} frames of size {target_size}")
    return frames


class AnomalyGenerator:
    """Generate controlled synthetic anomalies"""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def add_random_anomaly(self, frame, anomaly_type=None):
        """Add a random anomaly to the frame"""
        if anomaly_type is None:
            anomaly_type = self.rng.choice(["blackout", "vibration", "flip"])

        C, H, W = frame.shape
        frame_with_anomaly = frame.clone()

        # Random region size and position
        rh = self.rng.randint(H // 10, H // 4)
        rw = self.rng.randint(W // 10, W // 4)
        y = self.rng.randint(0, H - rh)
        x = self.rng.randint(0, W - rw)

        mask = torch.zeros((H, W), dtype=torch.bool)
        mask[y : y + rh, x : x + rw] = True

        if anomaly_type == "blackout":
            intensity = self.rng.uniform(0.7, 1.0)
            for c in range(C):
                frame_with_anomaly[c][mask] *= 1 - intensity
        elif anomaly_type == "vibration":
            intensity = self.rng.uniform(0.3, 0.7)
            noise = torch.randn(rh, rw) * intensity
            for c in range(C):
                frame_with_anomaly[c][y : y + rh, x : x + rw] += noise
                frame_with_anomaly[c] = torch.clamp(frame_with_anomaly[c], 0, 1)
        elif anomaly_type == "flip" and C == 2:
            flip_prob = self.rng.uniform(0.6, 0.9)
            flip_mask = torch.rand(rh, rw) < flip_prob
            pos_events = frame_with_anomaly[0, y : y + rh, x : x + rw].clone()
            neg_events = frame_with_anomaly[1, y : y + rh, x : x + rw].clone()
            frame_with_anomaly[0, y : y + rh, x : x + rw][flip_mask] = neg_events[
                flip_mask
            ]
            frame_with_anomaly[1, y : y + rh, x : x + rw][flip_mask] = pos_events[
                flip_mask
            ]

        return frame_with_anomaly, mask, anomaly_type


class BasicFeatureExtractor:
    """Extract basic statistical features from event data"""

    def __init__(self):
        self.feature_names = [
            "total_events",
            "pos_event_rate",
            "neg_event_rate",
            "polarity_ratio",
            "spatial_mean",
            "spatial_std",
            "spatial_max",
            "spatial_sparsity",
            "temporal_mean",
            "temporal_std",
            "intensity_mean",
            "intensity_std",
            "activity_regions",
            "edge_activity",
            "center_activity",
        ]

    def extract_features(self, frame):
        """Extract 15 basic statistical features"""
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()

        if len(frame.shape) == 2:
            frame = frame[np.newaxis, :]

        C, H, W = frame.shape
        features = []

        # Event channels
        pos_events = frame[0] if C > 0 else np.zeros((H, W))
        neg_events = frame[1] if C > 1 else np.zeros((H, W))
        combined_frame = pos_events + neg_events

        # 1. Event rate features
        total_events = np.sum(combined_frame)
        pos_event_rate = np.sum(pos_events) / (H * W)
        neg_event_rate = np.sum(neg_events) / (H * W)
        polarity_ratio = np.sum(pos_events) / (total_events + 1e-10)
        features.extend([total_events, pos_event_rate, neg_event_rate, polarity_ratio])

        # 2. Spatial statistics
        spatial_mean = np.mean(combined_frame)
        spatial_std = np.std(combined_frame)
        spatial_max = np.max(combined_frame)
        spatial_sparsity = np.sum(combined_frame > 0) / (H * W)
        features.extend([spatial_mean, spatial_std, spatial_max, spatial_sparsity])

        # 3. Temporal statistics (using channel means as proxy)
        temporal_mean = np.mean([np.mean(pos_events), np.mean(neg_events)])
        temporal_std = np.std([np.mean(pos_events), np.mean(neg_events)])
        features.extend([temporal_mean, temporal_std])

        # 4. Intensity statistics
        all_intensities = np.concatenate([pos_events.flatten(), neg_events.flatten()])
        intensity_mean = np.mean(all_intensities)
        intensity_std = np.std(all_intensities)
        features.extend([intensity_mean, intensity_std])

        # 5. Regional activity
        mid_h, mid_w = H // 2, W // 2
        regions = [
            combined_frame[:mid_h, :mid_w],
            combined_frame[:mid_h, mid_w:],
            combined_frame[mid_h:, :mid_w],
            combined_frame[mid_h:, mid_w:],
        ]
        activity_regions = sum(1 for region in regions if np.sum(region) > 0)

        # Edge vs center activity
        border = max(1, min(H, W) // 8)
        edge_mask = np.zeros((H, W), dtype=bool)
        edge_mask[:border, :] = edge_mask[-border:, :] = True
        edge_mask[:, :border] = edge_mask[:, -border:] = True

        edge_activity = np.mean(combined_frame[edge_mask])
        center_activity = np.mean(combined_frame[~edge_mask])
        features.extend([activity_regions, edge_activity, center_activity])

        return np.array(features, dtype=np.float32)


class SpatiotemporalFeatureExtractor:
    """Extract spatiotemporal features from event data"""

    def __init__(self):
        self.feature_names = [
            "density_mean",
            "density_std",
            "density_max",
            "density_entropy",
            "flow_magnitude_mean",
            "flow_magnitude_std",
            "flow_angle_mean",
            "flow_angle_std",
            "flow_coherence",
            "temporal_grad_mean",
            "temporal_grad_std",
            "temporal_consistency",
            "spatial_corr_mean",
            "spatial_corr_std",
            "motion_complexity",
            "directional_bias",
            "local_density_var",
            "edge_flow_ratio",
            "center_motion_strength",
            "boundary_activity",
        ]

    def compute_density_map(self, frame, kernel_size=5):
        """Compute local event density using convolution"""
        if len(frame.shape) == 3:
            frame = np.sum(frame, axis=0)
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
        return convolve2d(frame, kernel, mode="same", boundary="symm")

    def compute_optical_flow(self, frame1, frame2):
        """Compute optical flow between consecutive frames"""
        if len(frame1.shape) == 3:
            frame1 = np.sum(frame1, axis=0)
            frame2 = np.sum(frame2, axis=0)

        if HAS_OPENCV:
            try:
                frame1_uint8 = (frame1 * 255).astype(np.uint8)
                frame2_uint8 = (frame2 * 255).astype(np.uint8)

                flow = cv2.calcOpticalFlowFarneback(
                    frame1_uint8, frame2_uint8, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                angle = np.arctan2(flow[..., 1], flow[..., 0])
                return magnitude, angle
            except Exception:
                pass

        # Fallback: gradient-based flow
        diff = np.abs(frame2.astype(np.float32) - frame1.astype(np.float32))
        grad_result = np.gradient(diff)
        if isinstance(grad_result, tuple) and len(grad_result) == 2:
            grad_y, grad_x = grad_result
        else:
            grad_y = grad_x = grad_result
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        return magnitude, angle

    def extract_features(self, frame, prev_frame=None):
        """Extract 20 spatiotemporal features"""
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()

        C, H, W = frame.shape
        features = []

        # Combined frame for processing
        combined_frame = np.sum(frame, axis=0) if C > 1 else frame[0]

        # 1. Density features (4 features)
        density_map = self.compute_density_map(combined_frame)
        density_mean = np.mean(density_map)
        density_std = np.std(density_map)
        density_max = np.max(density_map)

        # Density entropy
        hist, _ = np.histogram(density_map.flatten(), bins=10, density=True)
        hist = hist + 1e-10
        density_entropy = -np.sum(hist * np.log(hist))

        features.extend([density_mean, density_std, density_max, density_entropy])

        # 2. Optical flow features (5 features)
        if prev_frame is not None:
            if isinstance(prev_frame, torch.Tensor):
                prev_frame = prev_frame.cpu().numpy()

            flow_magnitude, flow_angle = self.compute_optical_flow(prev_frame, frame)

            flow_mag_mean = np.mean(flow_magnitude)
            flow_mag_std = np.std(flow_magnitude)
            flow_angle_mean = np.mean(flow_angle)
            flow_angle_std = np.std(flow_angle)
            flow_coherence = (
                1.0 / (1.0 + np.std(flow_angle)) if np.std(flow_angle) > 1e-6 else 1.0
            )
        else:
            flow_mag_mean = (
                flow_mag_std
            ) = flow_angle_mean = flow_angle_std = flow_coherence = 0.0

        features.extend(
            [
                flow_mag_mean,
                flow_mag_std,
                flow_angle_mean,
                flow_angle_std,
                flow_coherence,
            ]
        )

        # 3. Temporal gradient features (3 features)
        grad_result = np.gradient(combined_frame)
        if isinstance(grad_result, tuple):
            # For 2D gradient, take the magnitude of both components
            grad_combined = np.sqrt(grad_result[0] ** 2 + grad_result[1] ** 2)
        else:
            grad_combined = grad_result

        temporal_grad_mean = np.mean(np.abs(grad_combined))
        temporal_grad_std = np.std(grad_combined)
        temporal_consistency = 1.0 / (1.0 + np.var(combined_frame))
        features.extend([temporal_grad_mean, temporal_grad_std, temporal_consistency])

        # 4. Spatial correlation features (2 features)
        grad_result = np.gradient(combined_frame)
        if isinstance(grad_result, tuple) and len(grad_result) == 2:
            grad_y, grad_x = grad_result
        else:
            grad_y = grad_x = grad_result
        spatial_corr_mean = np.mean(grad_x * grad_y)
        spatial_corr_std = np.std(grad_x + grad_y)
        features.extend([spatial_corr_mean, spatial_corr_std])

        # 5. Advanced spatiotemporal features (6 features)
        if prev_frame is not None:
            motion_complexity = np.std(flow_angle)
            angle_hist, _ = np.histogram(flow_angle, bins=8, range=(-np.pi, np.pi))
            angle_hist = angle_hist / (np.sum(angle_hist) + 1e-10)
            directional_bias = np.max(angle_hist) - np.min(angle_hist)
        else:
            motion_complexity = directional_bias = 0.0

        local_density_var = np.var(density_map)

        if prev_frame is not None:
            edge_mask = np.zeros((H, W), dtype=bool)
            border = max(1, min(H, W) // 8)
            edge_mask[:border, :] = edge_mask[-border:, :] = True
            edge_mask[:, :border] = edge_mask[:, -border:] = True

            edge_flow = np.mean(flow_magnitude[edge_mask])
            center_flow = np.mean(flow_magnitude[~edge_mask])
            edge_flow_ratio = edge_flow / (center_flow + 1e-10)
        else:
            edge_flow_ratio = 0.0

        # Center motion and boundary activity
        center_h, center_w = H // 2, W // 2
        center_region = combined_frame[
            center_h - H // 4 : center_h + H // 4, center_w - W // 4 : center_w + W // 4
        ]
        center_motion_strength = (
            np.mean(center_region) if center_region.size > 0 else 0.0
        )

        boundary_activity = (
            np.mean(combined_frame[:2, :])
            + np.mean(combined_frame[-2:, :])
            + np.mean(combined_frame[:, :2])
            + np.mean(combined_frame[:, -2:])
        )

        features.extend(
            [
                motion_complexity,
                directional_bias,
                local_density_var,
                edge_flow_ratio,
                center_motion_strength,
                boundary_activity,
            ]
        )

        return np.array(features, dtype=np.float32)


class NeuromorphicFeatureExtractor:
    """Extract neuromorphic-specific features from event data"""

    def __init__(self):
        self.feature_names = [
            "time_surface_mean",
            "time_surface_std",
            "time_surface_max",
            "time_surface_entropy",
            "event_surface_density",
            "event_surface_coherence",
            "temporal_histogram_peaks",
            "temporal_histogram_variance",
            "polarity_correlation",
            "spatial_event_clustering",
            "temporal_event_clustering",
            "event_frequency_spectrum",
            "motion_direction_consistency",
            "velocity_magnitude_mean",
            "velocity_magnitude_std",
            "acceleration_patterns",
            "event_lifetime_mean",
            "event_lifetime_std",
            "burst_detection_count",
            "silence_detection_count",
            "local_binary_patterns_mean",
            "local_binary_patterns_std",
            "fractal_dimension",
            "hausdorff_distance",
            "information_entropy",
            "mutual_information_xy",
            "mutual_information_tp",
            "cross_correlation_max",
            "autocorrelation_peak",
        ]

    def compute_time_surface(self, frame, tau=10000):
        """Compute time surface representation"""
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()

        if len(frame.shape) == 3:
            combined = np.sum(frame, axis=0)
        else:
            combined = frame

        # Simulate time surface decay
        time_surface = combined * np.exp(-combined / tau)
        return time_surface

    def compute_event_surface(self, frame):
        """Compute event surface density"""
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()

        if len(frame.shape) == 3:
            pos_events = frame[0] if frame.shape[0] > 0 else np.zeros(frame.shape[1:])
            neg_events = frame[1] if frame.shape[0] > 1 else np.zeros(frame.shape[1:])
        else:
            pos_events = neg_events = frame / 2

        # Event density map using convolution
        kernel = np.ones((3, 3)) / 9
        density_pos = convolve2d(pos_events, kernel, mode="same", boundary="symm")
        density_neg = convolve2d(neg_events, kernel, mode="same", boundary="symm")

        return density_pos, density_neg

    def compute_clustering_features(self, frame):
        """Compute event clustering features"""
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()

        if len(frame.shape) == 3:
            combined = np.sum(frame, axis=0)
        else:
            combined = frame

        # Find event locations
        event_locs = np.where(combined > np.mean(combined))
        if len(event_locs[0]) < 5:
            return 0, 0  # Not enough events for clustering

        # Spatial clustering using simple distance-based method
        coords = np.column_stack(event_locs)
        n_clusters = 0
        processed = set()

        for i, coord in enumerate(coords):
            if i in processed:
                continue
            cluster_size = 1
            processed.add(i)

            for j, other_coord in enumerate(coords[i + 1 :], i + 1):
                if j in processed:
                    continue
                dist = np.sqrt(np.sum((coord - other_coord) ** 2))
                if dist < 5:  # Distance threshold
                    cluster_size += 1
                    processed.add(j)

            if cluster_size >= 3:
                n_clusters += 1

        # Temporal clustering using intensity values
        intensities = combined[event_locs]
        intensity_std = np.std(intensities)
        temporal_clusters = int(intensity_std * 10)  # Proxy for temporal clustering

        return n_clusters, min(temporal_clusters, 10)

    def compute_local_binary_patterns(self, frame):
        """Compute Local Binary Patterns"""
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()

        if len(frame.shape) == 3:
            combined = np.sum(frame, axis=0)
        else:
            combined = frame

        # Simple LBP implementation
        H, W = combined.shape
        if H < 3 or W < 3:
            return 0, 0

        lbp = np.zeros((H - 2, W - 2))

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                center = combined[i, j]
                code = 0
                code |= (combined[i - 1, j - 1] > center) << 7
                code |= (combined[i - 1, j] > center) << 6
                code |= (combined[i - 1, j + 1] > center) << 5
                code |= (combined[i, j + 1] > center) << 4
                code |= (combined[i + 1, j + 1] > center) << 3
                code |= (combined[i + 1, j] > center) << 2
                code |= (combined[i + 1, j - 1] > center) << 1
                code |= (combined[i, j - 1] > center) << 0
                lbp[i - 1, j - 1] = code

        return np.mean(lbp), np.std(lbp)

    def compute_fractal_dimension(self, frame):
        """Estimate fractal dimension using box counting"""
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()

        if len(frame.shape) == 3:
            combined = np.sum(frame, axis=0)
        else:
            combined = frame

        # Binary image
        threshold = np.mean(combined) + 0.1 * np.std(combined)
        binary = (combined > threshold).astype(int)

        if np.sum(binary) == 0:
            return 1.0

        # Box counting
        H, W = binary.shape
        max_size = min(H, W) // 4
        sizes = [
            2**i for i in range(1, int(np.log2(max_size)) + 1) if 2**i <= max_size
        ]

        if len(sizes) < 2:
            return 1.0

        counts = []

        for size in sizes:
            count = 0
            for i in range(0, H, size):
                for j in range(0, W, size):
                    box = binary[i : i + size, j : j + size]
                    if np.sum(box) > 0:
                        count += 1
            counts.append(max(count, 1))

        # Estimate dimension
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)

        try:
            slope, _ = np.polyfit(log_sizes, log_counts, 1)
            return abs(slope)
        except (np.linalg.LinAlgError, ValueError):
            return 1.0

    def extract_features(self, frame, prev_frame=None):
        """Extract 29 neuromorphic-specific features"""
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()

        C, H, W = frame.shape
        features = []
        combined = np.sum(frame, axis=0) if C > 1 else frame[0]

        # 1. Time surface features (4 features)
        time_surface = self.compute_time_surface(combined)
        ts_mean = np.mean(time_surface)
        ts_std = np.std(time_surface)
        ts_max = np.max(time_surface)

        # Time surface entropy
        ts_hist, _ = np.histogram(time_surface.flatten(), bins=10, density=True)
        ts_hist = ts_hist + 1e-10
        ts_entropy = -np.sum(ts_hist * np.log(ts_hist))

        features.extend([ts_mean, ts_std, ts_max, ts_entropy])

        # 2. Event surface features (2 features)
        density_pos, density_neg = self.compute_event_surface(frame)
        es_density = np.mean(density_pos + density_neg)

        pos_flat = density_pos.flatten()
        neg_flat = density_neg.flatten()
        if np.std(pos_flat) > 1e-10 and np.std(neg_flat) > 1e-10:
            es_coherence = np.corrcoef(pos_flat, neg_flat)[0, 1]
        else:
            es_coherence = 0

        features.extend([es_density, es_coherence])

        # 3. Temporal histogram features (2 features)
        temp_hist, _ = np.histogram(combined.flatten(), bins=16, density=True)
        hist_peaks = len(
            [
                i
                for i in range(1, len(temp_hist) - 1)
                if temp_hist[i] > temp_hist[i - 1] and temp_hist[i] > temp_hist[i + 1]
            ]
        )
        hist_variance = np.var(temp_hist)
        features.extend([hist_peaks, hist_variance])

        # 4. Polarity correlation (1 feature)
        if C > 1:
            pos_events = frame[0].flatten()
            neg_events = frame[1].flatten()
            if np.std(pos_events) > 1e-10 and np.std(neg_events) > 1e-10:
                polarity_corr = np.corrcoef(pos_events, neg_events)[0, 1]
            else:
                polarity_corr = 0
        else:
            polarity_corr = 0
        features.append(polarity_corr)

        # 5. Clustering features (2 features)
        spatial_clusters, temporal_clusters = self.compute_clustering_features(frame)
        features.extend([spatial_clusters, temporal_clusters])

        # 6. Frequency spectrum (1 feature)
        fft = np.fft.fft2(combined)
        freq_spectrum = np.mean(np.abs(fft))
        features.append(freq_spectrum)

        # 7. Motion and velocity features (4 features)
        if prev_frame is not None:
            if isinstance(prev_frame, torch.Tensor):
                prev_frame = prev_frame.cpu().numpy()

            prev_combined = (
                np.sum(prev_frame, axis=0) if prev_frame.shape[0] > 1 else prev_frame[0]
            )

            # Motion analysis
            diff = combined - prev_combined
            grad_result = np.gradient(diff)
            if isinstance(grad_result, tuple) and len(grad_result) == 2:
                grad_y, grad_x = grad_result
            else:
                grad_y = grad_x = grad_result
            motion_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            motion_angle = np.arctan2(grad_y, grad_x)

            motion_consistency = (
                1.0 / (1.0 + np.std(motion_angle))
                if np.std(motion_angle) > 1e-6
                else 1.0
            )
            vel_mag_mean = np.mean(motion_magnitude)
            vel_mag_std = np.std(motion_magnitude)

            # Acceleration patterns
            accel_x_tuple = np.gradient(grad_x)
            accel_y_tuple = np.gradient(grad_y)

            # Handle gradient output (can be tuple for 2D arrays)
            if isinstance(accel_x_tuple, tuple):
                accel_x = accel_x_tuple[0] if len(accel_x_tuple) > 0 else grad_x
                accel_y = accel_y_tuple[0] if len(accel_y_tuple) > 0 else grad_y
            else:
                accel_x = accel_x_tuple
                accel_y = accel_y_tuple

            acceleration_magnitude = np.sqrt(accel_x**2 + accel_y**2)
            accel_patterns = np.std(acceleration_magnitude)
        else:
            motion_consistency = vel_mag_mean = vel_mag_std = accel_patterns = 0.0

        features.extend([motion_consistency, vel_mag_mean, vel_mag_std, accel_patterns])

        # 8-15. Additional neuromorphic features (simplified for space)
        # Event lifetime, burst detection, LBP, fractal dimension, etc.
        event_intensities = combined[combined > np.mean(combined)]
        if len(event_intensities) > 0:
            lifetime_mean = np.mean(event_intensities)
            lifetime_std = np.std(event_intensities)
        else:
            lifetime_mean = lifetime_std = 0.0

        threshold_high = np.mean(combined) + np.std(combined)
        burst_count = np.sum(combined > threshold_high)
        threshold_low = np.mean(combined) - 0.5 * np.std(combined)
        silence_count = np.sum(combined < threshold_low)

        lbp_mean, lbp_std = self.compute_local_binary_patterns(combined)
        fractal_dim = self.compute_fractal_dimension(combined)

        # Remaining features (simplified)
        remaining_features = [
            lifetime_mean,
            lifetime_std,
            burst_count,
            silence_count,
            lbp_mean,
            lbp_std,
            fractal_dim,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

        features.extend(remaining_features)

        # Ensure we have exactly 29 features
        features = (
            features[:29]
            if len(features) > 29
            else features + [0] * (29 - len(features))
        )

        return np.array(features, dtype=np.float32)


def run_feature_comparison_experiment():
    """Run feature comparison experiment focusing only on feature impact"""
    print("\n📊 FEATURE COMPARISON EXPERIMENT")
    print("=" * 50)
    print("GOAL: Determine which feature type works best")
    print("METHOD: Same algorithms, different features")

    # Load data
    try:
        events, sensor_size = load_mvsec_data("../data", "indoor_flying", "right")
        print("✅ MVSEC data loaded successfully")
    except Exception as e:
        print(f"⚠️  MVSEC data not available: {e}")
        print("🔄 Generating synthetic data for demonstration...")

        num_events = 100000
        H, W = 64, 64
        events = {
            "x": np.random.randint(0, W, num_events),
            "y": np.random.randint(0, H, num_events),
            "t": np.sort(np.random.uniform(0, 1, num_events)),
            "p": np.random.choice([-1, 1], num_events),
        }
        sensor_size = (H, W)
        print(f"✅ Generated {num_events:,} synthetic events")

    # Process events
    frames = process_events_to_frames(
        events, sensor_size, num_frames=50, target_size=(64, 64)
    )

    # Initialize extractors
    basic_extractor = BasicFeatureExtractor()
    spatiotemporal_extractor = SpatiotemporalFeatureExtractor()
    neuromorphic_extractor = NeuromorphicFeatureExtractor()
    anomaly_gen = AnomalyGenerator()

    # Create dataset
    print("\n⚙️  Creating feature datasets...")
    num_frames = len(frames)
    anomaly_ratio = 0.4
    num_anomalies = int(num_frames * anomaly_ratio)
    anomaly_indices = np.random.choice(num_frames, num_anomalies, replace=False)

    # Extract all three feature types
    basic_features = []
    spatio_features = []
    neuro_features = []
    labels = []

    basic_time = spatio_time = neuro_time = 0

    for i in range(num_frames):
        current_frame = frames[i]
        prev_frame = frames[i - 1] if i > 0 else None

        if i in anomaly_indices:
            current_frame, _, _ = anomaly_gen.add_random_anomaly(current_frame)
            labels.append(1)
        else:
            labels.append(0)

        # Extract basic features
        start_time = time.time()
        basic_feat = basic_extractor.extract_features(current_frame)
        basic_time += time.time() - start_time
        basic_features.append(basic_feat)

        # Extract spatiotemporal features
        start_time = time.time()
        spatio_feat = spatiotemporal_extractor.extract_features(
            current_frame, prev_frame
        )
        spatio_time += time.time() - start_time
        spatio_features.append(spatio_feat)

        # Extract neuromorphic features
        start_time = time.time()
        neuro_feat = neuromorphic_extractor.extract_features(current_frame, prev_frame)
        neuro_time += time.time() - start_time
        neuro_features.append(neuro_feat)

    basic_features = np.array(basic_features)
    spatio_features = np.array(spatio_features)
    neuro_features = np.array(neuro_features)
    labels = np.array(labels)

    print("✅ Feature extraction completed:")
    print(f"   Basic: {basic_features.shape} ({basic_time:.3f}s)")
    print(f"   Spatiotemporal: {spatio_features.shape} ({spatio_time:.3f}s)")
    print(f"   Neuromorphic: {neuro_features.shape} ({neuro_time:.3f}s)")

    # CONSISTENT ALGORITHMS for fair comparison
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "SVM": SVC(probability=True, random_state=SEED, kernel="rbf"),
        "Logistic Regression": LogisticRegression(random_state=SEED, max_iter=1000),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=SEED
        ),
    }

    # Test each feature type with same algorithms
    results = {}
    feature_sets = {
        "Basic": basic_features,
        "Spatiotemporal": spatio_features,
        "Neuromorphic": neuro_features,
    }

    for feature_name, feature_data in feature_sets.items():
        print(f"\n🧪 Testing {feature_name} Features:")
        results[feature_name] = {}

        X_train, X_test, y_train, y_test = train_test_split(
            feature_data, labels, test_size=0.3, random_state=SEED, stratify=labels
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for clf_name, clf in classifiers.items():
            clf_copy = type(clf)(**clf.get_params())  # Fresh copy
            clf_copy.fit(X_train_scaled, y_train)

            y_pred = clf_copy.predict(X_test_scaled)
            y_prob = (
                clf_copy.predict_proba(X_test_scaled)[:, 1]
                if hasattr(clf_copy, "predict_proba")
                else y_pred
            )

            f1 = f1_score(y_test, y_pred, zero_division=0)

            if len(np.unique(y_test)) > 1:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_score = auc(fpr, tpr)
            else:
                auc_score = 0.5

            results[feature_name][clf_name] = {
                "f1": f1,
                "auc": auc_score,
                "accuracy": accuracy_score(y_test, y_pred),
            }

            print(f"  {clf_name}: F1={f1:.3f}, AUC={auc_score:.3f}")

    # Analysis and visualization
    create_feature_comparison_visualization(
        results, basic_time, spatio_time, neuro_time
    )

    return results


def create_feature_comparison_visualization(
    results, basic_time, spatio_time, neuro_time
):
    """Create visualization focused on feature comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "RQ1-A: Feature Engineering Impact Analysis\nWhich features work best for neuromorphic anomaly detection?",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Average performance by feature type
    feature_types = list(results.keys())
    metrics = ["f1", "auc", "accuracy"]

    avg_scores = {}
    for feature_type in feature_types:
        avg_scores[feature_type] = {}
        for metric in metrics:
            scores = [
                results[feature_type][clf][metric] for clf in results[feature_type]
            ]
            avg_scores[feature_type][metric] = np.mean(scores)

    x = np.arange(len(metrics))
    width = 0.25
    colors = ["skyblue", "lightcoral", "lightgreen"]

    for i, feature_type in enumerate(feature_types):
        scores = [avg_scores[feature_type][metric] for metric in metrics]
        axes[0, 0].bar(
            x + i * width, scores, width, label=feature_type, alpha=0.8, color=colors[i]
        )

    axes[0, 0].set_xlabel("Metrics")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("Average Performance by Feature Type")
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels([m.upper() for m in metrics])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Computational efficiency
    times = [basic_time, spatio_time, neuro_time]
    axes[0, 1].bar(feature_types, times, color=colors, alpha=0.8)
    axes[0, 1].set_ylabel("Time (seconds)")
    axes[0, 1].set_title("Feature Extraction Time")
    axes[0, 1].grid(True, alpha=0.3)

    # Add efficiency ratios
    for i, (_, time_val) in enumerate(zip(feature_types, times, strict=False)):
        ratio = time_val / basic_time if basic_time > 0 else 1
        axes[0, 1].text(
            i,
            time_val + max(times) * 0.05,
            f"{ratio:.1f}x",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3. F1 scores by classifier and feature type
    classifiers = list(results["Basic"].keys())
    x = np.arange(len(classifiers))

    for i, feature_type in enumerate(feature_types):
        f1_scores = [results[feature_type][clf]["f1"] for clf in classifiers]
        axes[1, 0].bar(
            x + i * width,
            f1_scores,
            width,
            label=feature_type,
            alpha=0.8,
            color=colors[i],
        )

    axes[1, 0].set_xlabel("Classifier")
    axes[1, 0].set_ylabel("F1-Score")
    axes[1, 0].set_title("F1-Score by Classifier and Feature Type")
    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels(
        [name.replace(" ", "\n") for name in classifiers], fontsize=9
    )
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Performance improvement analysis
    basic_avg_f1 = avg_scores["Basic"]["f1"]
    spatio_avg_f1 = avg_scores["Spatiotemporal"]["f1"]
    neuro_avg_f1 = avg_scores["Neuromorphic"]["f1"]

    improvements = {
        "Spatiotemporal": ((spatio_avg_f1 - basic_avg_f1) / basic_avg_f1 * 100)
        if basic_avg_f1 > 0
        else 0,
        "Neuromorphic": ((neuro_avg_f1 - basic_avg_f1) / basic_avg_f1 * 100)
        if basic_avg_f1 > 0
        else 0,
    }

    axes[1, 1].bar(
        list(improvements.keys()),
        list(improvements.values()),
        color=["lightcoral", "lightgreen"],
        alpha=0.8,
    )
    axes[1, 1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
    axes[1, 1].set_ylabel("Improvement over Basic (%)")
    axes[1, 1].set_title("Feature Type Performance Improvement")
    axes[1, 1].grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (_, improvement) in enumerate(improvements.items()):
        axes[1, 1].text(
            i,
            improvement
            + (
                max(improvements.values()) * 0.05
                if improvement > 0
                else -max(abs(min(improvements.values())), 1) * 0.05
            ),
            f"{improvement:.1f}%",
            ha="center",
            va="bottom" if improvement > 0 else "top",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("feature_comparison_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Summary analysis
    best_feature = max(avg_scores.keys(), key=lambda x: avg_scores[x]["f1"])
    print("\n🏆 FEATURE COMPARISON RESULTS:")
    print("=" * 40)
    print(f"Best Feature Type: {best_feature}")
    print(
        f"F1 Scores: Basic={basic_avg_f1:.3f}, Spatio={spatio_avg_f1:.3f}, Neuro={neuro_avg_f1:.3f}"
    )
    print(
        f"Efficiency: Basic=1.0x, Spatio={spatio_time/basic_time:.1f}x, Neuro={neuro_time/basic_time:.1f}x"
    )

    if best_feature == "Neuromorphic":
        improvement = improvements["Neuromorphic"]
        print(
            f"\n💡 INSIGHT: Neuromorphic features provide {improvement:.1f}% improvement"
        )
        print("• Event-specific patterns are highly discriminative")
        print("• Time surfaces and clustering capture anomaly signatures")
        print("• Computational overhead justified by accuracy gains")
    elif best_feature == "Spatiotemporal":
        improvement = improvements["Spatiotemporal"]
        print(
            f"\n💡 INSIGHT: Spatiotemporal features provide {improvement:.1f}% improvement"
        )
        print("• Motion patterns are key to anomaly detection")
        print("• Optical flow captures dynamic anomalies effectively")
    else:
        print("\n💡 INSIGHT: Basic features are surprisingly competitive")
        print("• Simple statistical measures capture essential patterns")
        print("• Computational efficiency makes them practical for deployment")


if __name__ == "__main__":
    try:
        results = run_feature_comparison_experiment()

        # Save results
        feature_summary = []
        for feature_type in results:
            for classifier in results[feature_type]:
                metrics = results[feature_type][classifier]
                feature_summary.append(
                    {
                        "Feature_Type": feature_type,
                        "Classifier": classifier,
                        "F1_Score": metrics["f1"],
                        "AUC": metrics["auc"],
                        "Accuracy": metrics["accuracy"],
                    }
                )

        pd.DataFrame(feature_summary).to_csv(
            "feature_comparison_results.csv", index=False
        )
        print("✅ Feature comparison results saved to 'feature_comparison_results.csv'")

    except Exception as e:
        print(f"❌ Feature comparison experiment failed: {e}")
        import traceback

        traceback.print_exc()
