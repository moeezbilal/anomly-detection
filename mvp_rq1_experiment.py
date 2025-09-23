#!/usr/bin/env python3
"""
MVP: Basic vs Spatiotemporal Features for Neuromorphic Anomaly Detection

Minimal implementation comparing two feature extraction approaches:
- Basic Features: 9 statistical measures (event density, polarity ratio, etc.)
- Spatiotemporal Features: 12 motion/density features (optical flow, temporal gradients, etc.)

Uses Random Forest classifier with MVSEC dataset.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration constants
SEED = 42
MIN_ANOMALY_SIZE = 6
DEFAULT_FLIP_THRESHOLD = 0.1
DEFAULT_FLIP_INTENSITY = 0.6
MIN_FRAMES_FOR_FIXED_INTERVAL = 20
VISUALIZATION_SAMPLE_LIMIT = 5
EPSILON = 1e-8

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Set random seed for reproducibility - using single generator
rng = np.random.RandomState(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Check OpenCV availability
try:
    import cv2

    HAS_OPENCV = True
    logger.info("✅ OpenCV available for optical flow")
except ImportError:
    HAS_OPENCV = False
    logger.warning("⚠️  OpenCV not available, using fallback optical flow")


@dataclass
class ExperimentConfig:
    """Configuration for the MVP experiment"""

    data_path: str = "./data"
    sequence: str = "outdoor_day"
    camera: str = "left"
    max_events: int = 6400000
    sensor_size: tuple[int, int] = (64, 64)
    fixed_time_interval: float = 4.0
    anomaly_ratio: float = 0.20
    num_frames_fallback: int = 50
    test_size: float = 0.3
    n_estimators: int = 50

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.anomaly_ratio <= 0 or self.anomaly_ratio >= 1:
            raise ValueError("anomaly_ratio must be between 0 and 1")
        if self.max_events <= 0:
            raise ValueError("max_events must be positive")
        if len(self.sensor_size) != 2 or any(s <= 0 for s in self.sensor_size):
            raise ValueError("sensor_size must be tuple of two positive integers")


def load_mvsec_data(
    data_path: str = "./data", sequence: str = "outdoor_day", camera: str = "left"
) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    """Load MVSEC dataset from HDF5 files with improved error handling"""
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    try:
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
            available_files = [f for f in all_files if f.endswith(".hdf5")]
            raise FileNotFoundError(
                f"No MVSEC data files found for sequence '{sequence}' in {data_path}. "
                f"Available HDF5 files: {available_files}"
            )

        data_file = os.path.join(data_path, data_files[0])
        logger.info(f"Loading MVSEC data from: {data_file}")

        with h5py.File(data_file, "r") as f:
            # Validate file structure
            if "davis" not in f:
                raise ValueError("Invalid MVSEC file: 'davis' group not found")
            if camera not in f["davis"]:
                available_cameras = list(f["davis"].keys())
                raise ValueError(
                    f"Camera '{camera}' not found. Available cameras: {available_cameras}"
                )
            if "events" not in f["davis"][camera]:
                available_data = list(f["davis"][camera].keys())
                raise ValueError(
                    f"No events found for camera '{camera}'. Available data: {available_data}"
                )

            # Load and validate events data
            events_data = f["davis"][camera]["events"][:]
            if len(events_data) == 0:
                raise ValueError("No events found in the data file")

            logger.info(f"Loaded {len(events_data)} events from {camera} camera")

            # Extract and validate event components: [x, y, timestamp, polarity]
            events = {
                "x": events_data[:, 0].astype(np.int32),
                "y": events_data[:, 1].astype(np.int32),
                "t": events_data[:, 2].astype(np.float64),  # timestamps in seconds
                "p": events_data[:, 3].astype(np.int8),  # polarity (-1 or 1)
            }

            # Validate event data ranges
            if np.any((events["p"] != 1) & (events["p"] != -1)):
                logger.warning("Found polarity values other than -1 and 1")

            # Calculate sensor size with bounds checking
            max_x, max_y = np.max(events["x"]), np.max(events["y"])
            min_x, min_y = np.min(events["x"]), np.min(events["y"])

            if min_x < 0 or min_y < 0:
                logger.warning(
                    f"Found negative coordinates: min_x={min_x}, min_y={min_y}"
                )

            sensor_size = (max_y + 1, max_x + 1)  # (height, width)
            logger.info(f"Detected sensor resolution: {sensor_size}")

            return events, sensor_size

    except Exception as e:
        logger.error(f"Error loading MVSEC data: {e}")
        raise


def process_events_to_frames(
    events: dict[str, np.ndarray],
    sensor_size: tuple[int, int],
    num_frames: int,
    max_events: int,
    target_size: tuple[int, int],
    fixed_time_interval: Optional[float] = None,
) -> torch.Tensor:
    """Convert events to frame representation with vectorized processing"""
    # Validate inputs
    if not events or len(events["x"]) == 0:
        raise ValueError("Empty events data")

    # Sample events if too many (vectorized sampling)
    if len(events["x"]) > max_events:
        indices = rng.choice(len(events["x"]), max_events, replace=False)
        events = {key: arr[indices] for key, arr in events.items()}
        logger.info(f"Sampled {max_events} events from {len(events['x'])} total")

    x, y, t, p = events["x"], events["y"], events["t"], events["p"]

    # Normalize timestamps
    t_min = np.min(t)
    t = t - t_min
    total_duration = np.max(t)

    # Determine frame generation strategy with validation
    min_frames = MIN_FRAMES_FOR_FIXED_INTERVAL
    if fixed_time_interval and total_duration / fixed_time_interval >= min_frames:
        actual_num_frames = int(total_duration / fixed_time_interval) + 1
        time_bins = np.arange(
            0, total_duration + fixed_time_interval, fixed_time_interval
        )
    else:
        actual_num_frames = num_frames
        time_bins = np.linspace(0, total_duration, num_frames + 1)

    if actual_num_frames <= 0:
        raise ValueError(f"Invalid number of frames: {actual_num_frames}")

    # Initialize frames
    H, W = target_size
    orig_H, orig_W = sensor_size

    # Vectorized coordinate scaling with bounds checking
    x_scaled = np.clip(np.round(x * W / orig_W).astype(np.int32), 0, W - 1)
    y_scaled = np.clip(np.round(y * H / orig_H).astype(np.int32), 0, H - 1)

    # Vectorized temporal binning
    bin_indices = np.searchsorted(time_bins[1:], t)
    bin_indices = np.clip(bin_indices, 0, actual_num_frames - 1)

    # Convert polarities to channel indices (1 -> 0, -1 -> 1)
    channel_indices = (p == -1).astype(np.int32)

    # Vectorized frame accumulation using advanced indexing
    frames = torch.zeros((actual_num_frames, 2, H, W), dtype=torch.float32)

    # Create flat indices for efficient accumulation
    flat_indices = (
        bin_indices * 2 * H * W + channel_indices * H * W + y_scaled * W + x_scaled
    )

    # Use numpy's add.at for efficient accumulation
    flat_frames = np.zeros(actual_num_frames * 2 * H * W, dtype=np.float32)
    np.add.at(flat_frames, flat_indices, 1.0)

    # Reshape back to tensor format
    frames = torch.from_numpy(flat_frames.reshape(actual_num_frames, 2, H, W))

    # Vectorized normalization
    max_vals = torch.amax(frames, dim=(2, 3), keepdim=True)
    frames = torch.where(max_vals > 0, frames / max_vals, frames)

    logger.info(f"Generated {actual_num_frames} frames of size {target_size}")
    return frames


class AnomalyGenerator:
    """Generate controlled anomalies for supervised learning with random shapes"""

    def __init__(self, generator: Optional[np.random.RandomState] = None):
        self.rng = generator if generator is not None else rng
        # Storage for visualization examples (limited size)
        self.original_frames: list[torch.Tensor] = []
        self.anomalous_frames: list[torch.Tensor] = []
        self.anomaly_masks: list[torch.Tensor] = []
        self.anomaly_types: list[str] = []

    def _safe_position_selection(self, H: int, W: int, margin: int) -> tuple[int, int]:
        """Safely select a position ensuring anomaly fits within bounds"""
        # Ensure we have valid bounds
        if H <= 2 * margin or W <= 2 * margin:
            # Fallback to center if margins are too large
            return H // 2, W // 2

        cy = self.rng.randint(margin, H - margin)
        cx = self.rng.randint(margin, W - margin)
        return cy, cx

    def _get_safe_size_params(
        self, H: int, W: int, shape_type: Optional[str] = None
    ) -> dict[str, Any]:
        """Generate safe size parameters that fit within frame bounds"""
        if shape_type is None:
            shape_type = self.rng.choice(
                ["rectangle", "circle", "ellipse", "irregular"]
            )

        min_size = max(MIN_ANOMALY_SIZE, min(H, W) // 10)
        max_size = min(min(H, W) // 4, 50)  # Cap maximum size

        if min_size >= max_size:
            min_size = max_size // 2

        if shape_type == "rectangle":
            return {
                "height": self.rng.randint(min_size, max_size + 1),
                "width": self.rng.randint(min_size, max_size + 1),
            }
        elif shape_type == "circle":
            return {"radius": self.rng.randint(min_size // 2, max_size // 2 + 1)}
        elif shape_type == "ellipse":
            semi_major = self.rng.randint(min_size, max_size + 1)
            semi_minor = self.rng.randint(min_size // 2, semi_major + 1)
            return {
                "semi_major": semi_major,
                "semi_minor": semi_minor,
                "angle": self.rng.uniform(0, 2 * np.pi),
            }
        elif shape_type == "irregular":
            return {
                "base_radius": self.rng.randint(min_size // 2, max_size // 2 + 1),
                "num_blobs": self.rng.randint(3, 7),
            }

        return {}

    def _create_rectangle_mask(
        self, H: int, W: int, cy: int, cx: int, size_params: dict[str, Any]
    ) -> torch.Tensor:
        """Create rectangular mask"""
        rh = min(size_params.get("height", 20), H - 1)
        rw = min(size_params.get("width", 20), W - 1)

        y1 = max(0, cy - rh // 2)
        y2 = min(H, cy + rh // 2)
        x1 = max(0, cx - rw // 2)
        x2 = min(W, cx + rw // 2)

        mask = torch.zeros((H, W), dtype=torch.bool)
        mask[y1:y2, x1:x2] = True
        return mask

    def _create_circle_mask(
        self, H: int, W: int, cy: int, cx: int, size_params: dict[str, Any]
    ) -> torch.Tensor:
        """Create circular mask"""
        radius = min(size_params.get("radius", 15), min(H, W) // 3)

        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )

        distances = torch.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
        return distances <= radius

    def _create_ellipse_mask(
        self, H: int, W: int, cy: int, cx: int, size_params: dict[str, Any]
    ) -> torch.Tensor:
        """Create elliptical mask"""
        semi_major = min(size_params.get("semi_major", 20), min(H, W) // 2)
        semi_minor = min(size_params.get("semi_minor", 12), min(H, W) // 3)
        angle = size_params.get("angle", 0.0)

        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )

        # Translate to center and rotate
        y_rel = y_coords - cy
        x_rel = x_coords - cx

        cos_angle = torch.cos(torch.tensor(angle, dtype=torch.float32))
        sin_angle = torch.sin(torch.tensor(angle, dtype=torch.float32))

        y_rot = y_rel * cos_angle - x_rel * sin_angle
        x_rot = y_rel * sin_angle + x_rel * cos_angle

        # Ellipse equation
        ellipse_eq = (x_rot / semi_major) ** 2 + (y_rot / semi_minor) ** 2
        return ellipse_eq <= 1.0

    def _create_irregular_mask(
        self, H: int, W: int, cy: int, cx: int, size_params: dict[str, Any]
    ) -> torch.Tensor:
        """Create irregular blob-like mask"""
        base_radius = min(size_params.get("base_radius", 12), min(H, W) // 4)
        num_blobs = size_params.get("num_blobs", 5)

        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )

        mask = torch.zeros((H, W), dtype=torch.bool)

        for _ in range(num_blobs):
            # Random offset from center
            offset_y = self.rng.normal(0, base_radius * 0.4)
            offset_x = self.rng.normal(0, base_radius * 0.4)

            blob_cy = cy + offset_y
            blob_cx = cx + offset_x
            blob_radius = base_radius * self.rng.uniform(0.6, 1.4)

            distances = torch.sqrt(
                (y_coords - blob_cy) ** 2 + (x_coords - blob_cx) ** 2
            )
            blob_mask = distances <= blob_radius
            mask = mask | blob_mask

        return mask

    def generate_random_shape_mask(
        self,
        H: int,
        W: int,
        center_pos: tuple[int, int],
        size_params: dict[str, Any],
        shape_type: Optional[str] = None,
    ) -> tuple[torch.Tensor, str]:
        """Generate different geometric shape masks with improved error handling"""
        if shape_type is None:
            shape_type = self.rng.choice(
                ["rectangle", "circle", "ellipse", "irregular"]
            )

        cy, cx = center_pos

        try:
            if shape_type == "rectangle":
                mask = self._create_rectangle_mask(H, W, cy, cx, size_params)
            elif shape_type == "circle":
                mask = self._create_circle_mask(H, W, cy, cx, size_params)
            elif shape_type == "ellipse":
                mask = self._create_ellipse_mask(H, W, cy, cx, size_params)
            elif shape_type == "irregular":
                mask = self._create_irregular_mask(H, W, cy, cx, size_params)
            else:
                # Fallback to rectangle
                mask = self._create_rectangle_mask(H, W, cy, cx, size_params)
                shape_type = "rectangle"

            return mask, shape_type

        except Exception as e:
            logger.warning(
                f"Error creating {shape_type} mask: {e}. Using rectangle fallback."
            )
            mask = self._create_rectangle_mask(
                H, W, cy, cx, {"height": 10, "width": 10}
            )
            return mask, "rectangle"

    def _apply_blackout_anomaly(
        self, frame: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply blackout anomaly effect"""
        intensity = self.rng.uniform(0.7, 1.0)
        frame_with_anomaly = frame.clone()

        for c in range(frame.shape[0]):
            frame_with_anomaly[c][mask] *= 1 - intensity

        return frame_with_anomaly

    def _apply_random_noise_anomaly(
        self, frame: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply random noise anomaly effect"""
        intensity = self.rng.uniform(0.3, 0.7)
        frame_with_anomaly = frame.clone()

        H, W = frame.shape[1], frame.shape[2]

        for c in range(frame.shape[0]):
            # Create noise using consistent random generator
            noise = torch.from_numpy(
                self.rng.normal(0, intensity, (H, W)).astype(np.float32)
            )
            frame_with_anomaly[c][mask] += noise[mask]

        return torch.clamp(frame_with_anomaly, 0, 1)

    def _apply_polarity_flip_anomaly(
        self, frame: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply polarity flip anomaly with proper on/off logic"""
        if frame.shape[0] != 2:
            return self._apply_random_noise_anomaly(frame, mask)

        flip_prob = self.rng.uniform(0.6, 0.9)
        frame_with_anomaly = frame.clone()
        H, W = frame.shape[1], frame.shape[2]

        # Create random flip mask using consistent generator
        flip_mask_full = torch.from_numpy(
            (self.rng.random((H, W)) < flip_prob).astype(bool)
        )
        actual_flip_mask = mask & flip_mask_full

        if torch.sum(actual_flip_mask) == 0:
            return frame_with_anomaly

        # Get current values in flip region
        pos_values = frame_with_anomaly[0][actual_flip_mask]
        neg_values = frame_with_anomaly[1][actual_flip_mask]

        # Apply threshold-based flipping
        threshold = DEFAULT_FLIP_THRESHOLD
        flip_intensity = DEFAULT_FLIP_INTENSITY

        # Turn OFF the ON pixels, turn ON the OFF pixels
        new_pos_values = torch.where(
            pos_values > threshold,
            torch.zeros_like(pos_values),
            torch.full_like(pos_values, flip_intensity),
        )
        new_neg_values = torch.where(
            neg_values > threshold,
            torch.zeros_like(neg_values),
            torch.full_like(neg_values, flip_intensity),
        )

        frame_with_anomaly[0][actual_flip_mask] = new_pos_values
        frame_with_anomaly[1][actual_flip_mask] = new_neg_values

        return frame_with_anomaly

    def add_random_anomaly(
        self,
        frame: torch.Tensor,
        anomaly_type: Optional[str] = None,
        shape_type: Optional[str] = None,
        store_example: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Add a random anomaly to the frame with improved error handling"""
        if anomaly_type is None:
            anomaly_type = self.rng.choice(["blackout", "random_noise", "flip"])

        C, H, W = frame.shape

        try:
            # Generate shape parameters and position safely
            size_params = self._get_safe_size_params(H, W, shape_type)
            margin = max(size_params.values()) if size_params else 15
            cy, cx = self._safe_position_selection(H, W, margin)

            # Generate mask with random shape
            mask, actual_shape = self.generate_random_shape_mask(
                H, W, (cy, cx), size_params, shape_type
            )

            # Apply anomaly effect based on type
            if anomaly_type == "blackout":
                frame_with_anomaly = self._apply_blackout_anomaly(frame, mask)
            elif anomaly_type == "random_noise":
                frame_with_anomaly = self._apply_random_noise_anomaly(frame, mask)
            elif anomaly_type == "flip":
                frame_with_anomaly = self._apply_polarity_flip_anomaly(frame, mask)
            else:
                logger.warning(
                    f"Unknown anomaly type: {anomaly_type}. Using random_noise."
                )
                frame_with_anomaly = self._apply_random_noise_anomaly(frame, mask)
                anomaly_type = "random_noise"

            # Store example for visualization (increased limit for better selection)
            if store_example and len(self.original_frames) < 10:  # Store more examples
                self._store_visualization_example(
                    frame, frame_with_anomaly, mask, anomaly_type
                )

            return frame_with_anomaly, mask, anomaly_type

        except Exception as e:
            logger.error(f"Error creating anomaly: {e}")
            # Return original frame with empty mask on error
            empty_mask = torch.zeros((H, W), dtype=torch.bool)
            return frame.clone(), empty_mask, "error"

    def _store_visualization_example(
        self,
        original_frame: torch.Tensor,
        anomalous_frame: torch.Tensor,
        mask: torch.Tensor,
        anomaly_type: str,
    ) -> None:
        """Store example for visualization with debug info"""
        self.original_frames.append(original_frame.clone())
        self.anomalous_frames.append(anomalous_frame.clone())
        self.anomaly_masks.append(mask.clone())
        self.anomaly_types.append(anomaly_type)

        # Debug verification
        original_sum = torch.sum(original_frame).item()
        anomaly_sum = torch.sum(anomalous_frame).item()
        logger.debug(
            f"Created {anomaly_type} anomaly - Original sum: {original_sum:.3f}, "
            f"Anomaly sum: {anomaly_sum:.3f}"
        )

    def clear_visualization_storage(self) -> None:
        """Clear stored visualization examples to free memory"""
        self.original_frames.clear()
        self.anomalous_frames.clear()
        self.anomaly_masks.clear()
        self.anomaly_types.clear()

    def visualize_before_after_frames(self, num_samples=5):
        """Visualize original frames next to their anomalous versions (before/after pairs)"""
        if len(self.original_frames) == 0:
            print("No anomaly examples stored for visualization")
            return

        print(
            f"\n📸 Before/After Comparison: {num_samples} Original → Anomalous Frame Pairs"
        )
        print("=" * 70)

        # Select samples for visualization ensuring each anomaly type is represented
        num_available = min(num_samples, len(self.original_frames))
        if num_available == 0:
            print("No anomalous frames available for visualization")
            return

        # Get unique anomaly types and their indices
        anomaly_types_available = list(set(self.anomaly_types))
        indices = []

        # First, ensure we have at least one example of each anomaly type
        for anomaly_type in anomaly_types_available:
            type_indices = [
                i for i, t in enumerate(self.anomaly_types) if t == anomaly_type
            ]
            if type_indices:
                selected_idx = rng.choice(type_indices)
                indices.append(selected_idx)

        # If we still need more samples, randomly select from remaining
        remaining_slots = num_available - len(indices)
        if remaining_slots > 0:
            all_indices = set(range(len(self.original_frames)))
            remaining_indices = list(all_indices - set(indices))

            if remaining_indices:
                additional_indices = rng.choice(
                    remaining_indices,
                    min(remaining_slots, len(remaining_indices)),
                    replace=False,
                )
                indices.extend(additional_indices)

        # Convert to numpy array and limit to requested number
        indices = np.array(indices[:num_available])

        print(f"Selected anomaly types: {[self.anomaly_types[i] for i in indices]}")

        # Always use 3-row layout to show change map for all anomaly types
        fig, axes = plt.subplots(3, num_available, figsize=(4 * num_available, 12))
        if num_available == 1:
            axes = axes.reshape(3, 1)

        fig.suptitle(
            "Before/After: Original Frames → Synthetic Anomalies (Random Shapes)",
            fontsize=16,
            fontweight="bold",
        )

        for i, idx in enumerate(indices):
            # Get frames and anomaly info
            original_frame = self.original_frames[idx]
            anomalous_frame = self.anomalous_frames[idx]
            mask = self.anomaly_masks[idx]
            anomaly_type = self.anomaly_types[idx]

            # Row 1: Original combined view
            original_display = original_frame[0] + original_frame[1]
            im1 = axes[0, i].imshow(
                original_display.cpu().numpy(), cmap="viridis", vmin=0, vmax=1
            )
            axes[0, i].set_title(f"Original Frame #{idx}", fontsize=12)
            axes[0, i].axis("off")

            # Row 2: Anomalous combined view
            anomalous_display = anomalous_frame[0] + anomalous_frame[1]
            im2 = axes[1, i].imshow(
                anomalous_display.cpu().numpy(), cmap="viridis", vmin=0, vmax=1
            )
            axes[1, i].set_title(f"+ {anomaly_type.title()} Anomaly", fontsize=12)
            axes[1, i].axis("off")

            # Row 3: Change map (different for each anomaly type)
            if anomaly_type == "flip":
                # For polarity flip: show the absolute change in each channel
                pos_change = torch.abs(anomalous_frame[0] - original_frame[0])
                neg_change = torch.abs(anomalous_frame[1] - original_frame[1])

                # Combine changes to show overall polarity flip activity
                diff_display = pos_change + neg_change

                colormap = "Reds"  # Use red to show flip activity
                vmin, vmax = 0, (
                    torch.max(diff_display) if torch.max(diff_display) > 0 else 1
                )
                change_title = "Polarity Flip Activity Map"
            else:
                # For blackout/random_noise: show intensity difference
                diff_display = anomalous_display - original_display
                colormap = "RdYlBu_r"  # Red for increases, blue for decreases
                # Calculate appropriate range based on actual differences
                diff_max = torch.max(torch.abs(diff_display))
                if diff_max > 0:
                    vmin, vmax = -diff_max, diff_max
                else:
                    vmin, vmax = -0.1, 0.1  # Default small range

                if anomaly_type == "blackout":
                    change_title = "Intensity Reduction Map"
                elif anomaly_type == "random_noise":
                    change_title = "Random Noise Addition Map"
                else:
                    change_title = "Change Map"

            im3 = axes[2, i].imshow(
                diff_display.cpu().numpy(), cmap=colormap, vmin=vmin, vmax=vmax
            )
            axes[2, i].set_title(change_title, fontsize=12)
            axes[2, i].axis("off")

            # Enhanced mask overlay for different shapes - use contour instead of rectangle
            if mask is not None and torch.sum(mask) > 0:
                mask_np = mask.cpu().numpy().astype(float)

                # Use contour visualization for all shape types (works for any shape)
                for row_idx in range(3):
                    contours = axes[row_idx, i].contour(
                        mask_np, levels=[0.5], colors="red", linewidths=2, alpha=0.8
                    )
                    # Also add a semi-transparent filled contour to highlight the region better
                    axes[row_idx, i].contourf(
                        mask_np, levels=[0.5, 1.0], colors=["red"], alpha=0.2
                    )

        plt.tight_layout()
        plt.show()

        # Print detailed comparison statistics with shape information
        print("\n📊 Before/After Comparison Statistics:")
        print(f"   • Total anomalous frames created: {len(self.anomalous_frames)}")
        print("   • Anomaly types distribution:")

        anomaly_counts = {}
        for atype in self.anomaly_types:
            anomaly_counts[atype] = anomaly_counts.get(atype, 0) + 1

        for atype, count in anomaly_counts.items():
            # Format display name properly
            display_name = atype.replace("_", " ").title()
            print(f"     - {display_name}: {count} frames")


class BasicFeatureExtractor:
    """Extract basic statistical features"""

    def __init__(self):
        self.feature_names = [
            "total_event_density",
            "polarity_ratio",
            "spatial_sparsity",
            "event_concentration",
            "center_edge_ratio",
            "spatial_entropy",
            "max_activity_ratio",
            "polarity_imbalance",
            "activity_regularity",
        ]

    def extract_features(self, frame):
        """Extract 9 basic features"""
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()

        if len(frame.shape) == 2:
            frame = frame[np.newaxis, :]

        C, H, W = frame.shape
        features = []

        pos_events = frame[0] if C > 0 else np.zeros((H, W))
        neg_events = frame[1] if C > 1 else np.zeros((H, W))
        combined_frame = pos_events + neg_events

        total_pixels = H * W
        total_events = np.sum(combined_frame)

        # Core statistics
        features.append(total_events / total_pixels)  # event density
        features.append(np.sum(pos_events) / (total_events + 1e-8))  # polarity ratio
        features.append(np.sum(combined_frame > 0) / total_pixels)  # spatial sparsity
        features.append(
            np.std(combined_frame) / (np.mean(combined_frame) + 1e-8)
        )  # concentration

        # Spatial distribution
        border_width = max(1, min(H, W) // 8)
        edge_mask = np.zeros((H, W), dtype=bool)
        edge_mask[:border_width, :] = edge_mask[-border_width:, :] = True
        edge_mask[:, :border_width] = edge_mask[:, -border_width:] = True

        edge_activity = np.mean(combined_frame[edge_mask])
        center_activity = np.mean(combined_frame[~edge_mask])
        features.append(center_activity / (edge_activity + 1e-8))  # center/edge ratio

        # Spatial entropy
        if total_events > 0:
            prob_dist = combined_frame.flatten() / total_events
            prob_dist = prob_dist[prob_dist > 0]
            spatial_entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
        else:
            spatial_entropy = 0.0
        features.append(spatial_entropy)

        features.append(
            np.max(combined_frame) / (np.mean(combined_frame) + 1e-8)
        )  # max/mean ratio

        # Channel relationships
        pos_sum, neg_sum = np.sum(pos_events), np.sum(neg_events)
        features.append(
            abs(pos_sum - neg_sum) / (pos_sum + neg_sum + 1e-8)
        )  # polarity imbalance
        features.append(1.0 / (1.0 + np.var(combined_frame)))  # activity regularity

        return np.array(features, dtype=np.float32)

    def get_feature_names(self):
        return self.feature_names.copy()


class SpatiotemporalFeatureExtractor:
    """Extract spatiotemporal features"""

    def __init__(self):
        self.feature_names = [
            "flow_magnitude_mean",
            "flow_coherence",
            "motion_complexity",
            "dominant_flow_direction",
            "motion_stability",
            "density_contrast",
            "density_entropy",
            "density_clustering",
            "density_gradient_mag",
            "temporal_gradient_energy",
            "temporal_consistency",
            "event_persistence",
        ]

    def compute_optical_flow(self, frame1, frame2):
        """Compute optical flow between frames"""
        if len(frame1.shape) == 3:
            frame1 = np.sum(frame1, axis=0)
            frame2 = np.sum(frame2, axis=0)

        if HAS_OPENCV and np.max(frame1) > 0 and np.max(frame2) > 0:
            try:
                frame1_norm = ((frame1 / np.max(frame1)) * 255).astype(np.uint8)
                frame2_norm = ((frame2 / np.max(frame2)) * 255).astype(np.uint8)
                flow = cv2.calcOpticalFlowFarneback(
                    frame1_norm, frame2_norm, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                angle = np.arctan2(flow[..., 1], flow[..., 0])
                return magnitude, angle
            except:
                pass

        # Fallback: gradient-based
        diff = frame2.astype(np.float32) - frame1.astype(np.float32)
        grad_y, grad_x = np.gradient(diff)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        return magnitude, angle

    def extract_features(self, frame, prev_frame=None):
        """Extract 12 spatiotemporal features"""
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()

        C = frame.shape[0]
        features = []
        combined_frame = np.sum(frame, axis=0) if C > 1 else frame[0]

        # Motion dynamics (5 features)
        if prev_frame is not None:
            if isinstance(prev_frame, torch.Tensor):
                prev_frame = prev_frame.cpu().numpy()

            flow_magnitude, flow_angle = self.compute_optical_flow(prev_frame, frame)

            features.append(np.mean(flow_magnitude))  # flow magnitude mean
            features.append(1.0 / (1.0 + np.std(flow_angle)))  # flow coherence
            features.append(np.std(flow_angle))  # motion complexity

            # Dominant flow direction
            mean_cos = np.mean(np.cos(flow_angle))
            mean_sin = np.mean(np.sin(flow_angle))
            dominant_direction = np.arctan2(mean_sin, mean_cos)
            features.append((dominant_direction + np.pi) / (2 * np.pi))  # normalized

            features.append(1.0 / (1.0 + np.var(flow_magnitude)))  # motion stability
        else:
            features.extend([0.0, 0.0, 0.0, 0.5, 1.0])

        # Density patterns (4 features)
        density_map = gaussian_filter(combined_frame, sigma=1.0)

        features.append(
            np.std(density_map) / (np.mean(density_map) + 1e-8)
        )  # density contrast

        # Density entropy
        if np.sum(density_map) > 0:
            prob_dist = density_map.flatten() / np.sum(density_map)
            prob_dist = prob_dist[prob_dist > 1e-10]
            density_entropy = -np.sum(prob_dist * np.log(prob_dist))
        else:
            density_entropy = 0.0
        features.append(density_entropy)

        # Simple clustering measure
        features.append(
            np.mean(density_map[density_map > 0])
            if np.sum(density_map > 0) > 0
            else 0.0
        )

        # Density gradients
        grad_y, grad_x = np.gradient(density_map)
        features.append(np.mean(np.sqrt(grad_x**2 + grad_y**2)))

        # Temporal dynamics (3 features) - simplified for single frame
        features.extend([0.0, 1.0, 0.0])  # placeholder values

        return np.array(features, dtype=np.float32)

    def get_feature_names(self):
        return self.feature_names.copy()


def run_mvp_experiment(config: Optional[ExperimentConfig] = None) -> dict[str, Any]:
    """Run optimized MVP RQ1 experiment with improved components"""
    if config is None:
        config = ExperimentConfig()

    logger.info("🚀 MVP: Basic vs Spatiotemporal Features Comparison")
    logger.info("=" * 60)

    logger.info("Experiment configuration:")
    logger.info(f"  • Data path: {config.data_path}")
    logger.info(f"  • Sequence: {config.sequence}")
    logger.info(f"  • Camera: {config.camera}")
    logger.info(f"  • Max events: {config.max_events:,}")
    logger.info(f"  • Sensor size: {config.sensor_size}")
    logger.info(f"  • Fixed time interval: {config.fixed_time_interval}s")
    logger.info(f"  • Anomaly ratio: {config.anomaly_ratio:.1%}")

    try:
        # Step 1: Load data
        logger.info("\n📊 Step 1: Loading MVSEC data...")
        events, sensor_size = load_mvsec_data(
            config.data_path, config.sequence, config.camera
        )
        logger.info(f"✅ Loaded {len(events['x']):,} events")

        # Step 2: Create frames
        logger.info("\n🎬 Step 2: Converting events to frames...")
        frames = process_events_to_frames(
            events,
            sensor_size,
            config.num_frames_fallback,
            config.max_events,
            config.sensor_size,
            config.fixed_time_interval,
        )
        logger.info(
            f"✅ Generated {frames.shape[0]} frames of size {config.sensor_size}"
        )

        # Step 3: Extract features with improved anomaly generation
        logger.info("\n🔧 Step 3: Extracting features with anomaly injection...")

        # Initialize components with consistent random state
        basic_extractor = BasicFeatureExtractor()
        spatio_extractor = SpatiotemporalFeatureExtractor()
        anomaly_gen = AnomalyGenerator(generator=rng)

        basic_features: list[np.ndarray] = []
        spatio_features: list[np.ndarray] = []
        labels: list[int] = []

        # Generate anomalies with better distribution
        num_frames = len(frames)
        num_anomalies = int(num_frames * config.anomaly_ratio)
        anomaly_indices = set(rng.choice(num_frames, num_anomalies, replace=False))

        logger.info(
            f"Generating {num_anomalies} anomalies ({config.anomaly_ratio:.1%} ratio)"
        )

        for i in range(num_frames):
            if i in anomaly_indices:
                # Create anomaly (store more examples for better visualization selection)
                store_example = len(anomaly_gen.original_frames) < 10
                frame_with_anomaly, _, anomaly_type = anomaly_gen.add_random_anomaly(
                    frames[i], store_example=store_example
                )

                # Extract features from anomalous frame
                basic_feat = basic_extractor.extract_features(frame_with_anomaly)
                spatio_feat = spatio_extractor.extract_features(
                    frame_with_anomaly, frames[i - 1] if i > 0 else None
                )
                labels.append(1)  # Anomaly
            else:
                # Extract features from normal frame
                basic_feat = basic_extractor.extract_features(frames[i])
                spatio_feat = spatio_extractor.extract_features(
                    frames[i], frames[i - 1] if i > 0 else None
                )
                labels.append(0)  # Normal

            basic_features.append(basic_feat)
            spatio_features.append(spatio_feat)

        # Convert to arrays with validation
        basic_features = np.array(basic_features, dtype=np.float32)
        spatio_features = np.array(spatio_features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        # Validate feature extraction
        if np.any(np.isnan(basic_features)) or np.any(np.isnan(spatio_features)):
            logger.warning("NaN values detected in features")

        logger.info(f"✅ Basic features: {basic_features.shape}")
        logger.info(f"✅ Spatiotemporal features: {spatio_features.shape}")
        logger.info(f"✅ Actual anomaly ratio: {np.mean(labels):.1%}")

        # Visualize before/after frames
        logger.info("\n🎬 Step 4: Generating before/after visualization...")
        anomaly_gen.visualize_before_after_frames(num_samples=5)

        # Step 5: Train and evaluate models
        logger.info("\n🤖 Step 5: Training Random Forest models...")

        # Split data with stratification
        X_basic_train, X_basic_test, y_train, y_test = train_test_split(
            basic_features,
            labels,
            test_size=config.test_size,
            random_state=SEED,
            stratify=labels,
        )
        X_spatio_train, X_spatio_test, _, _ = train_test_split(
            spatio_features,
            labels,
            test_size=config.test_size,
            random_state=SEED,
            stratify=labels,
        )

        logger.info(
            f"Train/test split: {len(X_basic_train)}/{len(X_basic_test)} samples"
        )

        # Scale features
        scaler_basic = StandardScaler()
        X_basic_train_scaled = scaler_basic.fit_transform(X_basic_train)
        X_basic_test_scaled = scaler_basic.transform(X_basic_test)

        scaler_spatio = StandardScaler()
        X_spatio_train_scaled = scaler_spatio.fit_transform(X_spatio_train)
        X_spatio_test_scaled = scaler_spatio.transform(X_spatio_test)

        # Train models with improved parameters
        rf_basic = RandomForestClassifier(
            n_estimators=config.n_estimators,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,  # Use all cores
        )
        rf_spatio = RandomForestClassifier(
            n_estimators=config.n_estimators,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )

        logger.info("Training basic features model...")
        rf_basic.fit(X_basic_train_scaled, y_train)

        logger.info("Training spatiotemporal features model...")
        rf_spatio.fit(X_spatio_train_scaled, y_train)

        # Step 6: Evaluate and compare results
        logger.info("\n📊 Step 6: Evaluating model performance...")

        # Generate predictions
        y_pred_basic = rf_basic.predict(X_basic_test_scaled)
        y_pred_spatio = rf_spatio.predict(X_spatio_test_scaled)
        y_prob_basic = rf_basic.predict_proba(X_basic_test_scaled)[:, 1]
        y_prob_spatio = rf_spatio.predict_proba(X_spatio_test_scaled)[:, 1]

        # Calculate comprehensive metrics
        basic_metrics = {
            "f1": f1_score(y_test, y_pred_basic, zero_division=0),
            "accuracy": accuracy_score(y_test, y_pred_basic),
            "auc_pr": average_precision_score(y_test, y_prob_basic),
        }

        spatio_metrics = {
            "f1": f1_score(y_test, y_pred_spatio, zero_division=0),
            "accuracy": accuracy_score(y_test, y_pred_spatio),
            "auc_pr": average_precision_score(y_test, y_prob_spatio),
        }

        # Display results with improved formatting
        logger.info("\n📊 Results Comparison:")
        logger.info("=" * 50)
        logger.info(
            f"{'Feature Type':<20} {'F1-Score':<10} {'Accuracy':<10} {'AUC-PR':<10}"
        )
        logger.info(
            f"{'Basic Features':<20} {basic_metrics['f1']:<10.3f} {basic_metrics['accuracy']:<10.3f} {basic_metrics['auc_pr']:<10.3f}"
        )
        logger.info(
            f"{'Spatiotemporal':<20} {spatio_metrics['f1']:<10.3f} {spatio_metrics['accuracy']:<10.3f} {spatio_metrics['auc_pr']:<10.3f}"
        )

        # Calculate improvements safely
        improvements = {}
        for metric in ["f1", "accuracy", "auc_pr"]:
            basic_val = basic_metrics[metric]
            spatio_val = spatio_metrics[metric]

            if basic_val > EPSILON:
                improvement = ((spatio_val - basic_val) / basic_val) * 100
            else:
                improvement = 0.0
            improvements[metric] = improvement

        logger.info(
            f"{'Improvement':<20} {improvements['f1']:+.1f}%      {improvements['accuracy']:+.1f}%      {improvements['auc_pr']:+.1f}%"
        )

        # Determine winner systematically
        wins = sum(
            1
            for metric in ["f1", "accuracy", "auc_pr"]
            if spatio_metrics[metric] > basic_metrics[metric]
        )

        winner = "Spatiotemporal" if wins >= 2 else "Basic"
        logger.info(f"\n🏆 WINNER: {winner} Features ({wins}/3 metrics)")

        # Feature importance analysis
        logger.info("\n🔍 Feature Importance Analysis:")

        basic_importance = rf_basic.feature_importances_
        basic_names = basic_extractor.get_feature_names()
        basic_top = sorted(
            zip(basic_names, basic_importance, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        logger.info("Top 3 Basic Features:")
        for i, (name, imp) in enumerate(basic_top, 1):
            logger.info(f"   {i}. {name}: {imp:.3f}")

        spatio_importance = rf_spatio.feature_importances_
        spatio_names = spatio_extractor.get_feature_names()
        spatio_top = sorted(
            zip(spatio_names, spatio_importance, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        logger.info("Top 3 Spatiotemporal Features:")
        for i, (name, imp) in enumerate(spatio_top, 1):
            logger.info(f"   {i}. {name}: {imp:.3f}")

        # Clean up visualization storage
        anomaly_gen.clear_visualization_storage()

        logger.info("\n✅ MVP EXPERIMENT COMPLETED SUCCESSFULLY!")
        logger.info(f"📊 {winner} features perform better overall")

        # Return comprehensive results
        return {
            "winner": winner,
            "config": config,
            "basic_metrics": basic_metrics,
            "spatiotemporal_metrics": spatio_metrics,
            "improvements": improvements,
            "feature_importance": {
                "basic": dict(zip(basic_names, basic_importance, strict=False)),
                "spatiotemporal": dict(
                    zip(spatio_names, spatio_importance, strict=False)
                ),
            },
            "data_info": {
                "num_frames": num_frames,
                "num_anomalies": num_anomalies,
                "anomaly_ratio": np.mean(labels),
                "train_samples": len(X_basic_train),
                "test_samples": len(X_basic_test),
            },
        }

    except Exception as e:
        logger.error(f"❌ MVP experiment failed: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e), "winner": None}


if __name__ == "__main__":
    # Run the optimized MVP experiment
    results = run_mvp_experiment()

    if results and "error" not in results:
        logger.info(f"\n🎯 FINAL RESULT: {results['winner']} features win!")
        logger.info("Experiment completed successfully with all improvements applied!")
    elif results and "error" in results:
        logger.error(f"Experiment failed with error: {results['error']}")
    else:
        logger.error("Experiment failed to return results")
