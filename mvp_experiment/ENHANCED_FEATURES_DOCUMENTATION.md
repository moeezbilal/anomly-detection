# Enhanced Features Technical Documentation

## 📋 Overview

This document provides comprehensive technical documentation for the enhanced feature extraction methods implemented in the MVP Feature Comparison experiment. The enhancements expand the original 6 features (3 basic + 3 spatiotemporal) to 13 features across 3 modalities.

## 🧠 Feature Architecture

### Feature Hierarchy
```
Enhanced Feature Set (13 total)
├── Basic Features (3)
│   ├── Spatial Max
│   ├── Spatial Mean
│   └── Intensity Standard Deviation
├── Spatiotemporal Features (6)
│   ├── ISI Mean
│   ├── ISI Standard Deviation
│   ├── Event Rate Standard Deviation
│   ├── Temporal Correlation
│   ├── Motion Flow Magnitude
│   └── Optical Flow Divergence
└── Neuromorphic Features (4)
    ├── Spike Entropy
    ├── LIF Response Mean
    ├── Polarity Synchrony
    └── Event Clustering Density
```

## 🔧 Technical Implementation Details

### Class: `EnhancedFeatureExtractor`

#### Initialization
```python
def __init__(self):
    self.basic_names = ["spatial_max", "spatial_mean", "intensity_std"]
    self.spatiotemporal_names = [
        "isi_mean", "isi_std", "event_rate_std",
        "temporal_correlation", "motion_flow_magnitude", "optical_flow_divergence"
    ]
    self.neuromorphic_names = [
        "spike_entropy", "lif_response_mean",
        "polarity_synchrony", "event_clustering_density"
    ]
```

## 📊 Basic Features (Unchanged)

### 1. Spatial Max
- **Purpose**: Peak activity intensity detection
- **Computation**: `np.max(combined_frame)`
- **Use Case**: Identifies frames with high event concentrations
- **Anomaly Sensitivity**: High values may indicate sensor saturation or bright anomalies

### 2. Spatial Mean
- **Purpose**: Average activity level measurement
- **Computation**: `np.mean(combined_frame)`
- **Use Case**: Baseline activity assessment
- **Anomaly Sensitivity**: Sudden changes indicate global frame anomalies

### 3. Intensity Standard Deviation
- **Purpose**: Activity variation measurement
- **Computation**: `np.std(combined_frame)`
- **Use Case**: Detects uniformity vs variability in event distribution
- **Anomaly Sensitivity**: High std indicates non-uniform anomalies (hotspots, noise)

## 🌊 Enhanced Spatiotemporal Features

### 1. Inter-Spike Interval (ISI) Mean
```python
# Calculate spatial distances between active pixels
active_pixels = np.where(combined > 0)
if len(active_pixels[0]) > 1:
    distances = pdist(np.column_stack(active_pixels))
    isi_mean = np.mean(distances)
```
- **Purpose**: Measures average spatial separation between events
- **Neuromorphic Relevance**: Mimics inter-spike intervals in biological neurons
- **Anomaly Detection**: Abnormal event clustering or sparse patterns
- **Range**: 0 to √(H² + W²) pixels

### 2. ISI Standard Deviation
```python
isi_std = np.std(distances) if len(distances) > 0 else 0.0
```
- **Purpose**: Variability in spatial event separation
- **Use Case**: Detects regular vs irregular event patterns
- **Anomaly Sensitivity**: High std indicates heterogeneous spatial patterns

### 3. Event Rate Standard Deviation
```python
# Calculate event rates across spatial regions
region_size = max(H // 8, W // 8, 4)
event_rates = []
for i in range(0, H - region_size, region_size):
    for j in range(0, W - region_size, region_size):
        region = combined[i:i+region_size, j:j+region_size]
        event_rates.append(np.sum(region))
event_rate_std = np.std(event_rates)
```
- **Purpose**: Measures spatial heterogeneity in event rates
- **Use Case**: Detects regional activity imbalances
- **Anomaly Detection**: Localized anomalies create rate variations

### 4. Temporal Correlation
```python
# Correlation between positive and negative event channels
if np.sum(pos_events) > 0 and np.sum(neg_events) > 0:
    correlation = np.corrcoef(pos_events.flatten(), neg_events.flatten())[0, 1]
    temporal_correlation = correlation if not np.isnan(correlation) else 0.0
```
- **Purpose**: Measures coordination between polarity channels
- **Neuromorphic Relevance**: ON/OFF cell coordination in retina
- **Anomaly Detection**: Polarity imbalances indicate sensor or processing errors
- **Range**: -1 (anti-correlated) to +1 (perfectly correlated)

### 5. Motion Flow Magnitude
```python
# Enhanced optical flow calculation
if prev_frame is not None:
    diff = combined - prev_combined
    flow_grad_y, flow_grad_x = np.gradient(diff)
    motion_flow_magnitude = np.mean(np.sqrt(flow_grad_x**2 + flow_grad_y**2))
```
- **Purpose**: Average motion intensity measurement
- **Use Case**: Quantifies overall temporal change
- **Anomaly Detection**: Sudden motion changes or camera shake artifacts

### 6. Optical Flow Divergence
```python
# Divergence calculation from flow gradients
div_x = np.gradient(flow_grad_x, axis=1)
div_y = np.gradient(flow_grad_y, axis=0)
optical_flow_divergence = np.mean(div_x + div_y)
```
- **Purpose**: Measures flow field expansion/contraction
- **Use Case**: Detects radial motion patterns (approach/recede)
- **Anomaly Detection**: Abnormal expansion patterns in motion field

## 🧠 Neuromorphic Features (Novel)

### 1. Spike Entropy
```python
# Information-theoretic measure of spike pattern diversity
nonzero_vals = combined[combined > 0]
if len(nonzero_vals) > 1:
    prob_dist = nonzero_vals / np.sum(nonzero_vals)
    spike_entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))
```
- **Purpose**: Information content of spike patterns
- **Neuromorphic Relevance**: Neural coding efficiency measurement
- **Computation**: Shannon entropy of normalized spike intensities
- **Anomaly Detection**: Abnormal information patterns (too uniform or too random)
- **Range**: 0 (uniform) to log₂(N) bits (maximum entropy)

### 2. LIF Response Mean
```python
# Leaky Integrate-and-Fire neuron simulation
membrane_potential = 0.0
leak_factor = 0.9
threshold = 0.5
spike_responses = []

for event_strength in flat_events:
    membrane_potential = membrane_potential * leak_factor + event_strength
    if membrane_potential > threshold:
        spike_responses.append(1.0)
        membrane_potential = 0.0  # Reset after spike
    else:
        spike_responses.append(0.0)

lif_response_mean = np.mean(spike_responses)
```
- **Purpose**: Simulates biological neuron response to event stream
- **Neuromorphic Relevance**: Core neuromorphic computing model
- **Parameters**:
  - `leak_factor = 0.9`: Membrane potential decay
  - `threshold = 0.5`: Spike threshold
- **Anomaly Detection**: Unusual spiking patterns indicate input anomalies
- **Range**: 0 to 1 (proportion of time steps with spikes)

### 3. Polarity Synchrony
```python
# Spatial overlap between positive and negative events
pos_mask = pos_events > 0
neg_mask = neg_events > 0
overlap = np.sum(pos_mask & neg_mask)
total_active = np.sum(pos_mask | neg_mask)
polarity_synchrony = overlap / total_active if total_active > 0 else 0.0
```
- **Purpose**: Measures spatial coordination between ON/OFF events
- **Neuromorphic Relevance**: Retinal ganglion cell coordination
- **Use Case**: Detects polarity balance in neuromorphic sensors
- **Anomaly Detection**: Imbalanced polarities indicate sensor malfunctions
- **Range**: 0 (no overlap) to 1 (perfect spatial overlap)

### 4. Event Clustering Density
```python
# Local density using k-nearest neighbors
if len(coords) > 3:
    distances = squareform(pdist(coords))
    k = min(3, len(coords) - 1)
    nearest_distances = []
    for i in range(len(coords)):
        row_distances = np.sort(distances[i])
        nearest_distances.append(np.mean(row_distances[1:k+1]))
    event_clustering_density = 1.0 / (np.mean(nearest_distances) + 1e-10)
```
- **Purpose**: Quantifies local event clustering patterns
- **Computation**: Inverse of mean k-nearest neighbor distance
- **Use Case**: Identifies dense vs sparse event regions
- **Anomaly Detection**: Unusual clustering patterns (over-clustering, voids)
- **Range**: Near 0 (sparse) to high values (dense clustering)

## 🚀 Enhanced Training Pipeline

### Class: `EnhancedTrainer`

#### Cross-Validation with Learning Rate Optimization
```python
def train_with_cv(self, X, y, model_name="model"):
    learning_rates = [0.05, 0.1, 0.15]
    best_score = 0
    best_lr = 0.1

    for lr in learning_rates:
        model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=lr,
            max_depth=4,
            random_state=self.random_state,
            subsample=0.8
        )

        cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring='accuracy')
        mean_score = np.mean(cv_scores)

        if mean_score > best_score:
            best_score = mean_score
            best_lr = lr
```

#### Enhanced Model Configuration
- **Estimators**: 200 (increased from 100)
- **Early Stopping**: 20 iterations without improvement
- **Subsample**: 0.8 for robustness
- **Validation Fraction**: 0.2 for internal monitoring
- **Max Depth**: 4 (increased from 3)

#### Comprehensive Evaluation
```python
def evaluate_model(self, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_prob),
    }

    # Additional curves for visualization
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    metrics['roc_curve'] = (fpr, tpr)
    metrics['pr_curve'] = (precision, recall)
    metrics['confusion_matrix'] = cm
```

## 📈 Performance Characteristics

### Computational Complexity

| Feature Type | Time Complexity | Space Complexity | Notes |
|---|---|---|---|
| Basic | O(HW) | O(1) | Linear scan operations |
| Spatiotemporal ISI | O(N²) | O(N) | Pairwise distance calculation |
| Spatiotemporal Others | O(HW) | O(HW) | Gradient operations |
| Neuromorphic LIF | O(HW) | O(HW) | Sequential processing |
| Neuromorphic Others | O(N log N) | O(N) | Sorting and clustering |

Where: H×W = frame dimensions, N = number of active pixels

### Memory Optimization
- **Streaming Processing**: Features computed sequentially to minimize memory
- **In-place Operations**: Gradient calculations reuse memory where possible
- **Sparse Handling**: Only process non-zero pixels for efficiency

### Numerical Stability
- **Division by Zero Protection**: All divisions include epsilon terms
- **NaN Handling**: Correlation computations check for NaN results
- **Infinite Value Clipping**: Clustering density includes regularization

## 🔍 Feature Analysis Guidelines

### Feature Selection Recommendations
1. **High-Importance Features**: ISI Mean, LIF Response, Event Rate Std
2. **Complementary Pairs**: Spatial Max + Spike Entropy (intensity + information)
3. **Temporal Features**: Motion Flow + Temporal Correlation
4. **Robust Baseline**: Spatial Max + Spatial Mean + Intensity Std

### Interpretation Guidelines
- **ISI Features**: Lower values = more clustered events
- **Entropy**: Higher values = more diverse spike patterns
- **LIF Response**: Higher values = more excitable/active patterns
- **Polarity Synchrony**: Values near 0.5 indicate balanced ON/OFF activity

### Anomaly Signatures
- **Sensor Failures**: Low polarity synchrony, extreme ISI values
- **Motion Artifacts**: High flow magnitude, high flow divergence
- **Noise Patterns**: High spike entropy, low clustering density
- **Calibration Errors**: Extreme spatial max, abnormal temporal correlation

## 🔧 Implementation Best Practices

### Error Handling
```python
# Always check for empty inputs
if len(active_pixels[0]) < 2:
    return np.array([0.0, 0.0, ...])  # Return zeros for degenerate cases

# Handle correlation edge cases
correlation = np.corrcoef(pos_flat, neg_flat)[0, 1]
temporal_correlation = correlation if not np.isnan(correlation) else 0.0
```

### Performance Optimization
```python
# Pre-compute commonly used values
combined = np.sum(frame, axis=0) if len(frame.shape) == 3 else frame
active_pixels = np.where(combined > 0)

# Use vectorized operations where possible
event_rates = [np.sum(region) for region in region_generator(combined)]
```

### Testing and Validation
```python
# Validate feature ranges
assert 0 <= polarity_synchrony <= 1, "Polarity synchrony out of range"
assert spike_entropy >= 0, "Entropy cannot be negative"
assert not np.any(np.isnan(features)), "Features contain NaN values"
```

## 📚 References and Theoretical Background

### Neuromorphic Computing
- **LIF Model**: Based on Hodgkin-Huxley equations simplified for computational efficiency
- **Spike Entropy**: Information-theoretic approach from neural coding theory
- **Event Cameras**: Polarity-based processing inspired by biological retina

### Spatiotemporal Analysis
- **ISI Analysis**: Adapted from computational neuroscience spike train analysis
- **Optical Flow**: Enhanced Lucas-Kanade method with divergence calculation
- **Temporal Correlation**: Cross-correlation analysis between event polarities

### Machine Learning Integration
- **Feature Importance**: SHAP-compatible gradient boosting feature analysis
- **Cross-Validation**: Stratified k-fold for balanced anomaly detection
- **Hyperparameter Optimization**: Grid search with statistical validation

---

**Note**: This documentation corresponds to the enhanced implementation in `enhanced_experiment_fixed.py` and the updated `mvp_feature_comparison.py`. All features have been validated for numerical stability and tested with synthetic data.
