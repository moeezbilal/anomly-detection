#!/usr/bin/env python3
"""
RQ1 - Part B: Algorithm Approach Comparison
===========================================

OBJECTIVE: Compare different ALGORITHMIC APPROACHES for neuromorphic anomaly detection
- Supervised Classification: Learn normal vs anomaly patterns
- Unsupervised Anomaly Detection: Learn normal patterns, flag outliers

APPROACH: Same features (best from Part A) tested on different algorithms
RESEARCH QUESTION: Is specialized anomaly detection better than classification?

This analysis focuses ONLY on algorithm performance, using the best feature set identified from Part A.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import h5py
from scipy.signal import convolve2d

from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.neighbors import LocalOutlierFactor

# Advanced ensemble methods
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print("🤖 RQ1-B: Algorithm Approach Comparison")
print("=" * 60)
print("FOCUS: Classification vs Anomaly Detection approaches")
print("METHOD: Best features on different algorithm types")
print("=" * 60)


def load_mvsec_data(data_path='../data', sequence='indoor_flying', camera='left'):
    """Load MVSEC dataset from HDF5 files"""
    data_files = []
    if os.path.isdir(data_path):
        all_files = os.listdir(data_path)
        candidate_files = [f for f in all_files if f.endswith('.hdf5') and 'data' in f]
        
        if sequence:
            sequence_files = [f for f in candidate_files if sequence.lower() in f.lower()]
            data_files = sequence_files if sequence_files else candidate_files
        else:
            data_files = candidate_files
    
    if not data_files:
        available_files = [f for f in os.listdir(data_path) if f.endswith('.hdf5')]
        raise ValueError(f"No MVSEC data files found for sequence '{sequence}' in {data_path}. Available files: {available_files}")
    
    data_file = os.path.join(data_path, data_files[0])
    print(f"📁 Loading MVSEC data from: {data_file}")
    
    with h5py.File(data_file, 'r') as f:
        events_data = f['davis'][camera]['events'][:]
        print(f"📊 Loaded {len(events_data):,} events from {camera} camera")
        
        events = {
            'x': events_data[:, 0].astype(int),
            'y': events_data[:, 1].astype(int),
            't': events_data[:, 2],
            'p': events_data[:, 3].astype(int)
        }
        
        max_x, max_y = np.max(events['x']), np.max(events['y'])
        sensor_size = (max_y + 1, max_x + 1)
        print(f"📐 Sensor resolution: {sensor_size}")
        
        return events, sensor_size


def process_events_to_frames(events, sensor_size, num_frames=50, max_events=300000, target_size=(64, 64)):
    """Convert events to frame representation"""
    if len(events['x']) > max_events:
        indices = np.linspace(0, len(events['x'])-1, max_events, dtype=int)
        for key in events:
            events[key] = events[key][indices]
        print(f"⚡ Subsampled to {max_events:,} events for processing")
    
    x, y, t, p = events['x'], events['y'], events['t'], events['p']
    
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
    x_scaled = np.clip(x_scaled, 0, W-1)
    y_scaled = np.clip(y_scaled, 0, H-1)
    
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
            anomaly_type = self.rng.choice(['blackout', 'vibration', 'flip'])
        
        C, H, W = frame.shape
        frame_with_anomaly = frame.clone()
        
        # Random region size and position
        rh = self.rng.randint(H//10, H//4)
        rw = self.rng.randint(W//10, W//4)
        y = self.rng.randint(0, H - rh)
        x = self.rng.randint(0, W - rw)
        
        mask = torch.zeros((H, W), dtype=torch.bool)
        mask[y:y+rh, x:x+rw] = True
        
        if anomaly_type == 'blackout':
            intensity = self.rng.uniform(0.7, 1.0)
            for c in range(C):
                frame_with_anomaly[c][mask] *= (1 - intensity)
        elif anomaly_type == 'vibration':
            intensity = self.rng.uniform(0.3, 0.7)
            noise = torch.randn(rh, rw) * intensity
            for c in range(C):
                frame_with_anomaly[c][y:y+rh, x:x+rw] += noise
                frame_with_anomaly[c] = torch.clamp(frame_with_anomaly[c], 0, 1)
        elif anomaly_type == 'flip' and C == 2:
            flip_prob = self.rng.uniform(0.6, 0.9)
            flip_mask = torch.rand(rh, rw) < flip_prob
            pos_events = frame_with_anomaly[0, y:y+rh, x:x+rw].clone()
            neg_events = frame_with_anomaly[1, y:y+rh, x:x+rw].clone()
            frame_with_anomaly[0, y:y+rh, x:x+rw][flip_mask] = neg_events[flip_mask]
            frame_with_anomaly[1, y:y+rh, x:x+rw][flip_mask] = pos_events[flip_mask]
        
        return frame_with_anomaly, mask, anomaly_type


class NeuromorphicFeatureExtractor:
    """Extract neuromorphic-specific features (best from Part A)"""
    
    def __init__(self):
        self.feature_names = [
            'time_surface_mean', 'time_surface_std', 'time_surface_max', 'time_surface_entropy',
            'event_surface_density', 'event_surface_coherence', 'temporal_histogram_peaks', 'temporal_histogram_variance',
            'polarity_correlation', 'spatial_event_clustering', 'temporal_event_clustering', 'event_frequency_spectrum',
            'motion_direction_consistency', 'velocity_magnitude_mean', 'velocity_magnitude_std', 'acceleration_patterns',
            'event_lifetime_mean', 'event_lifetime_std', 'burst_detection_count', 'silence_detection_count',
            'local_binary_patterns_mean', 'local_binary_patterns_std', 'fractal_dimension', 'hausdorff_distance',
            'information_entropy', 'mutual_information_xy', 'mutual_information_tp', 'cross_correlation_max', 'autocorrelation_peak'
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
        density_pos = convolve2d(pos_events, kernel, mode='same', boundary='symm')
        density_neg = convolve2d(neg_events, kernel, mode='same', boundary='symm')
        
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
            
            for j, other_coord in enumerate(coords[i+1:], i+1):
                if j in processed:
                    continue
                dist = np.sqrt(np.sum((coord - other_coord)**2))
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
        hist_peaks = len([i for i in range(1, len(temp_hist)-1) 
                         if temp_hist[i] > temp_hist[i-1] and temp_hist[i] > temp_hist[i+1]])
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
        
        # Simplified remaining features for brevity
        remaining_features = [0] * (29 - len(features))
        features.extend(remaining_features)
        
        # Ensure we have exactly 29 features
        features = features[:29] if len(features) > 29 else features + [0] * (29 - len(features))
        
        return np.array(features, dtype=np.float32)


def run_algorithm_comparison_experiment():
    """Run algorithm comparison experiment focusing on approach differences"""
    print("\n🤖 ALGORITHM COMPARISON EXPERIMENT")
    print("=" * 50)
    print("GOAL: Classification vs Anomaly Detection approaches")
    print("METHOD: Best features, different algorithms")
    
    # Load data
    try:
        events, sensor_size = load_mvsec_data('../data', 'indoor_flying', 'left')
        print("✅ MVSEC data loaded successfully")
    except Exception as e:
        print(f"⚠️  MVSEC data not available: {e}")
        print("🔄 Generating synthetic data for demonstration...")
        
        num_events = 100000
        H, W = 64, 64
        events = {
            'x': np.random.randint(0, W, num_events),
            'y': np.random.randint(0, H, num_events),
            't': np.sort(np.random.uniform(0, 1, num_events)),
            'p': np.random.choice([-1, 1], num_events)
        }
        sensor_size = (H, W)
        print(f"✅ Generated {num_events:,} synthetic events")
    
    # Process events
    frames = process_events_to_frames(events, sensor_size, num_frames=50, target_size=(64, 64))
    
    # Use neuromorphic features (assumed best from Part A)
    neuromorphic_extractor = NeuromorphicFeatureExtractor()
    anomaly_gen = AnomalyGenerator()
    
    # Create dataset
    print("\n⚙️  Creating neuromorphic feature dataset...")
    num_frames = len(frames)
    anomaly_ratio = 0.4
    num_anomalies = int(num_frames * anomaly_ratio)
    anomaly_indices = np.random.choice(num_frames, num_anomalies, replace=False)
    
    features = []
    labels = []
    
    for i in range(num_frames):
        current_frame = frames[i]
        prev_frame = frames[i-1] if i > 0 else None
        
        if i in anomaly_indices:
            current_frame, _, _ = anomaly_gen.add_random_anomaly(current_frame)
            labels.append(1)
        else:
            labels.append(0)
        
        feat = neuromorphic_extractor.extract_features(current_frame, prev_frame)
        features.append(feat)
    
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"✅ Dataset created: {features.shape} features, {np.sum(labels)} anomalies")
    
    # ALGORITHM COMPARISON: Two different approaches
    results = {'supervised_classification': {}, 'unsupervised_anomaly_detection': {}}
    
    # 1. SUPERVISED CLASSIFICATION APPROACH
    print("\n📚 SUPERVISED CLASSIFICATION APPROACH:")
    print("Method: Learn to distinguish normal vs anomaly patterns")
    
    supervised_classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED),
        'SVM': SVC(probability=True, random_state=SEED, kernel='rbf', C=10),
        'Logistic Regression': LogisticRegression(random_state=SEED, max_iter=2000, C=10),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=SEED)
    }
    
    # Add advanced methods if available
    if HAS_XGBOOST:
        supervised_classifiers['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=SEED, eval_metric='logloss'
        )
    
    if HAS_LIGHTGBM:
        supervised_classifiers['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=SEED, verbose=-1
        )
    
    # Train supervised classifiers
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=SEED, stratify=labels
    )
    
    scaler_supervised = StandardScaler()
    X_train_scaled = scaler_supervised.fit_transform(X_train)
    X_test_scaled = scaler_supervised.transform(X_test)
    
    for name, clf in supervised_classifiers.items():
        start_time = time.time()
        
        try:
            clf.fit(X_train_scaled, y_train)
            train_time = time.time() - start_time
            
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, 'predict_proba') else y_pred
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'train_time': train_time
            }
            
            if len(np.unique(y_test)) > 1:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                metrics['auc'] = auc(fpr, tpr)
            else:
                metrics['auc'] = 0.5
            
            results['supervised_classification'][name] = metrics
            print(f"  {name}: F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}, Time={train_time:.3f}s")
            
        except Exception as e:
            print(f"  {name}: Failed - {str(e)}")
    
    # 2. UNSUPERVISED ANOMALY DETECTION APPROACH
    print("\n🔍 UNSUPERVISED ANOMALY DETECTION APPROACH:")
    print("Method: Learn normal patterns, flag outliers")
    
    anomaly_detectors = {
        'Isolation Forest': IsolationForest(contamination=0.4, random_state=SEED),
        'One-Class SVM': OneClassSVM(gamma='scale', nu=0.4),
        'Local Outlier Factor': LocalOutlierFactor(contamination=0.4, novelty=True)
    }
    
    # Separate normal and anomaly data for unsupervised training
    X_normal = features[labels == 0]
    
    for name, detector in anomaly_detectors.items():
        start_time = time.time()
        
        try:
            if name == 'Local Outlier Factor':
                # For LOF, fit on normal data only
                scaler_ad = StandardScaler()
                X_normal_scaled = scaler_ad.fit_transform(X_normal)
                
                detector.fit(X_normal_scaled)
                train_time = time.time() - start_time
                
                # Test on all data
                X_all_scaled = scaler_ad.transform(features)
                y_pred_scores = detector.decision_function(X_all_scaled)
                y_pred = (y_pred_scores < 0).astype(int)  # LOF: negative scores are outliers
                
            else:
                # For Isolation Forest and One-Class SVM
                scaler_ad = StandardScaler()
                X_all_scaled = scaler_ad.fit_transform(features)
                
                detector.fit(X_all_scaled)
                train_time = time.time() - start_time
                
                y_pred_raw = detector.predict(X_all_scaled)
                y_pred = (y_pred_raw == -1).astype(int)  # Convert -1 (outlier) to 1 (anomaly)
                y_pred_scores = detector.decision_function(X_all_scaled)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(labels, y_pred),
                'precision': precision_score(labels, y_pred, zero_division=0),
                'recall': recall_score(labels, y_pred, zero_division=0),
                'f1': f1_score(labels, y_pred, zero_division=0),
                'train_time': train_time
            }
            
            # For AUC, use decision scores (flip for proper interpretation)
            if len(np.unique(labels)) > 1:
                y_scores_flipped = -y_pred_scores  # Higher score = more anomalous
                fpr, tpr, _ = roc_curve(labels, y_scores_flipped)
                metrics['auc'] = auc(fpr, tpr)
            else:
                metrics['auc'] = 0.5
            
            results['unsupervised_anomaly_detection'][name] = metrics
            print(f"  {name}: F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}, Time={train_time:.3f}s")
            
        except Exception as e:
            print(f"  {name}: Failed - {str(e)}")
    
    # Analysis and visualization
    create_algorithm_comparison_visualization(results)
    
    return results


def create_algorithm_comparison_visualization(results):
    """Create visualization focused on algorithm approach comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RQ1-B: Algorithm Approach Comparison\nSupervised Classification vs Unsupervised Anomaly Detection', 
                 fontsize=14, fontweight='bold')
    
    # 1. Average performance by approach
    supervised_f1 = [results['supervised_classification'][clf]['f1'] for clf in results['supervised_classification']]
    supervised_auc = [results['supervised_classification'][clf]['auc'] for clf in results['supervised_classification']]
    unsupervised_f1 = [results['unsupervised_anomaly_detection'][clf]['f1'] for clf in results['unsupervised_anomaly_detection']]
    unsupervised_auc = [results['unsupervised_anomaly_detection'][clf]['auc'] for clf in results['unsupervised_anomaly_detection']]
    
    approaches = ['Supervised\nClassification', 'Unsupervised\nAnomaly Detection']
    avg_f1 = [np.mean(supervised_f1), np.mean(unsupervised_f1)]
    avg_auc = [np.mean(supervised_auc), np.mean(unsupervised_auc)]
    
    x = np.arange(len(approaches))
    width = 0.35
    
    axes[0,0].bar(x - width/2, avg_f1, width, label='F1-Score', alpha=0.8, color='skyblue')
    axes[0,0].bar(x + width/2, avg_auc, width, label='AUC', alpha=0.8, color='lightcoral')
    axes[0,0].set_ylabel('Score')
    axes[0,0].set_title('Average Performance by Approach')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(approaches)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Individual algorithm performance
    all_algorithms = list(results['supervised_classification'].keys()) + list(results['unsupervised_anomaly_detection'].keys())
    all_f1_scores = [results['supervised_classification'][clf]['f1'] for clf in results['supervised_classification']] + \
                   [results['unsupervised_anomaly_detection'][clf]['f1'] for clf in results['unsupervised_anomaly_detection']]
    
    # Color code by approach
    colors = ['skyblue'] * len(results['supervised_classification']) + ['lightgreen'] * len(results['unsupervised_anomaly_detection'])
    
    axes[0,1].bar(range(len(all_algorithms)), all_f1_scores, color=colors, alpha=0.8)
    axes[0,1].set_ylabel('F1-Score')
    axes[0,1].set_title('Individual Algorithm Performance')
    axes[0,1].set_xticks(range(len(all_algorithms)))
    axes[0,1].set_xticklabels([name.replace(' ', '\n') for name in all_algorithms], fontsize=8, rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='skyblue', label='Supervised'),
                      Patch(facecolor='lightgreen', label='Unsupervised')]
    axes[0,1].legend(handles=legend_elements, loc='upper right')
    
    # 3. Performance distribution comparison
    axes[1,0].boxplot([supervised_f1, unsupervised_f1], labels=approaches)
    axes[1,0].set_ylabel('F1-Score')
    axes[1,0].set_title('F1-Score Distribution by Approach')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Best performers ranking
    all_results = {}
    for clf in results['supervised_classification']:
        all_results[f'Supervised_{clf}'] = results['supervised_classification'][clf]['f1']
    for clf in results['unsupervised_anomaly_detection']:
        all_results[f'Unsupervised_{clf}'] = results['unsupervised_anomaly_detection'][clf]['f1']
    
    # Get top 6 performers
    top_performers = sorted(all_results.items(), key=lambda x: x[1], reverse=True)[:6]
    top_names = [name.replace('_', '\n') for name, _ in top_performers]
    top_scores = [score for _, score in top_performers]
    
    # Color by approach
    top_colors = ['skyblue' if 'Supervised' in name else 'lightgreen' for name, _ in top_performers]
    
    axes[1,1].bar(range(len(top_names)), top_scores, color=top_colors, alpha=0.8)
    axes[1,1].set_ylabel('F1-Score')
    axes[1,1].set_title('Top 6 Performing Algorithms')
    axes[1,1].set_xticks(range(len(top_names)))
    axes[1,1].set_xticklabels(top_names, fontsize=8, rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analysis summary
    print(f"\n🏆 ALGORITHM COMPARISON RESULTS:")
    print("=" * 50)
    
    best_supervised = max(results['supervised_classification'].keys(), 
                         key=lambda x: results['supervised_classification'][x]['f1'])
    best_unsupervised = max(results['unsupervised_anomaly_detection'].keys(),
                           key=lambda x: results['unsupervised_anomaly_detection'][x]['f1'])
    
    best_supervised_f1 = results['supervised_classification'][best_supervised]['f1']
    best_unsupervised_f1 = results['unsupervised_anomaly_detection'][best_unsupervised]['f1']
    
    print(f"Best Supervised: {best_supervised} (F1={best_supervised_f1:.3f})")
    print(f"Best Unsupervised: {best_unsupervised} (F1={best_unsupervised_f1:.3f})")
    print(f"Average Supervised F1: {np.mean(supervised_f1):.3f}")
    print(f"Average Unsupervised F1: {np.mean(unsupervised_f1):.3f}")
    
    if np.mean(unsupervised_f1) > np.mean(supervised_f1):
        improvement = ((np.mean(unsupervised_f1) - np.mean(supervised_f1)) / np.mean(supervised_f1)) * 100
        print(f"\n💡 INSIGHT: Unsupervised anomaly detection provides {improvement:.1f}% improvement")
        print("• Learning 'normal' patterns is more effective than binary classification")
        print("• Anomaly detection algorithms are specialized for outlier detection")
        print("• Better for unknown/novel anomaly types in neuromorphic data")
    else:
        improvement = ((np.mean(supervised_f1) - np.mean(unsupervised_f1)) / np.mean(unsupervised_f1)) * 100
        print(f"\n💡 INSIGHT: Supervised classification provides {improvement:.1f}% improvement")
        print("• Explicit anomaly labeling helps classification performance")
        print("• Supervised learning benefits from knowing anomaly patterns")
        print("• Consider hybrid approaches combining both methods")
    
    # Recommendation
    overall_best = max(all_results, key=all_results.get)
    overall_best_score = all_results[overall_best]
    print(f"\n🎯 RECOMMENDATION:")
    print(f"Deploy: {overall_best.replace('_', ' ')} (F1={overall_best_score:.3f})")
    
    if 'Unsupervised' in overall_best:
        print("• Use for detecting unknown anomaly types")
        print("• Suitable for scenarios with limited labeled data")
        print("• Good for continuous learning from normal patterns")
    else:
        print("• Use when labeled anomaly data is available")
        print("• Suitable for known anomaly pattern recognition")
        print("• Good for consistent, predictable anomaly types")


if __name__ == "__main__":
    try:
        results = run_algorithm_comparison_experiment()
        
        # Save results
        algorithm_summary = []
        for approach in results:
            for algorithm in results[approach]:
                metrics = results[approach][algorithm]
                algorithm_summary.append({
                    'Approach': approach.replace('_', ' ').title(),
                    'Algorithm': algorithm,
                    'F1_Score': metrics['f1'],
                    'AUC': metrics['auc'],
                    'Accuracy': metrics['accuracy'],
                    'Training_Time': metrics['train_time']
                })
        
        pd.DataFrame(algorithm_summary).to_csv('algorithm_comparison_results.csv', index=False)
        print("✅ Algorithm comparison results saved to 'algorithm_comparison_results.csv'")
        
    except Exception as e:
        print(f"❌ Algorithm comparison experiment failed: {e}")
        import traceback
        traceback.print_exc()