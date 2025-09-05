# MVSEC Anomaly Detection System Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Model Architectures](#model-architectures)
5. [Anomaly Generation Strategy](#anomaly-generation-strategy)
6. [Training and Evaluation](#training-and-evaluation)
7. [Code Structure](#code-structure)
8. [Usage Guide](#usage-guide)
9. [Performance Analysis](#performance-analysis)
10. [Future Improvements](#future-improvements)

## Overview

### Objective
Develop an anomaly detection system for neuromorphic event-based data using Spiking Neural Networks (SNNs) and compare with traditional deep learning approaches (RNN, TCN).

### Key Features
- Real MVSEC dataset integration (Multi-Vehicle Stereo Event Camera)
- Artificial anomaly injection for supervised learning
- Multiple neural network architectures comparison
- Event-based data preprocessing and temporal binning
- Comprehensive evaluation metrics and visualization

### Primary Strategy
The system employs a **supervised anomaly detection approach** where:
1. Normal event data is loaded from MVSEC dataset
2. Artificial anomalies are systematically injected
3. Models learn to distinguish between normal and anomalous patterns
4. Performance is evaluated using standard classification metrics

## System Architecture

### High-Level Architecture
```
MVSEC HDF5 Files → Event Loading → Temporal Binning → Anomaly Injection → Model Training → Evaluation
      ↓                ↓               ↓                 ↓                 ↓              ↓
  Raw Events    Event Dictionary   Frame Tensors    Labeled Dataset    Trained Models   Metrics
```

### Component Overview
1. **Data Layer**: MVSEC HDF5 file handling and event extraction
2. **Preprocessing Layer**: Event-to-frame conversion and normalization
3. **Anomaly Layer**: Systematic injection of different anomaly types
4. **Model Layer**: SNN, RNN, and TCN implementations
5. **Training Layer**: Supervised learning with cross-entropy loss
6. **Evaluation Layer**: Comprehensive metrics and visualization

## Data Pipeline

### 1. MVSEC Data Loading
```python
def load_mvsec_data(data_path, sequence, camera):
    """
    Load event data from MVSEC HDF5 files
    Format: [x, y, timestamp, polarity]
    """
```

**Key Steps:**
- Locate HDF5 files matching sequence pattern
- Navigate HDF5 structure: `davis/{camera}/events`
- Extract event arrays with 4D format: `[x, y, t, p]`
- Handle coordinate bounds and polarity mapping

### 2. Event Preprocessing
```python
def preprocess_events(events, num_frames, sensor_size):
    """
    Convert event stream to temporal frames
    Output: (num_frames, 2, H, W) tensor
    """
```

**Temporal Binning Strategy:**
- Divide time range into uniform bins
- Map events to spatial-temporal frames
- Separate positive/negative polarity channels
- Normalize frame intensities to [0, 1]

**Frame Structure:**
- Channel 0: Positive events (polarity = +1)
- Channel 1: Negative events (polarity = -1)
- Spatial resolution: Configurable (default 64×64)

### 3. Memory Management
- **Event Sampling**: Limit to 500K events for manageable processing
- **Batch Processing**: Small batch sizes (8-16) for memory efficiency
- **Frame Resizing**: Downscale from 260×346 to 64×64 for faster training

## Model Architectures

### 1. Spiking Neural Network (SNN)

#### Core Concept
- **Bio-inspired**: Mimics neural spike-based communication
- **Temporal Dynamics**: Membrane potential accumulation and decay
- **Surrogate Gradients**: Enable backpropagation through discrete spikes

#### Architecture Details
```python
class SpikingAnomalyDetector:
    - SpikingConv2d(2 → 16, kernel=3, stride=2)
    - SpikingConv2d(16 → 32, kernel=3, stride=2)
    - SpikingConv2d(32 → 64, kernel=3, stride=2)
    - GlobalAvgPool2d + Linear(64 → 2)
```

#### Spiking Neuron Model
```
Membrane Potential: V(t) = β·V(t-1) + I(t)
Spike Generation: S(t) = H(V(t) - θ)
Reset: V(t) ← V(t) - S(t)·θ
```
Where: β=decay factor, I=input current, θ=threshold, H=Heaviside

#### Surrogate Gradient
```python
def surrogate_gradient(x, α=10):
    return α * exp(-α|x|) / (1 + exp(-αx))²
```

### 2. Recurrent Neural Network (RNN)

#### Architecture
- **Convolutional Feature Extraction**: 2×Conv2d layers
- **Temporal Processing**: GRU layer for sequence modeling
- **Classification Head**: Linear layer for binary output

```python
class RNNAnomalyDetector:
    - Conv2d(2 → 16, kernel=3, stride=2)
    - Conv2d(16 → 32, kernel=3, stride=2)
    - GRU(features → 64)
    - Linear(64 → 2)
```

### 3. Temporal Convolutional Network (TCN)

#### Architecture
- **Dilated Convolutions**: Capture long-range temporal dependencies
- **Residual Connections**: Enable deep architectures
- **Causal Convolutions**: Maintain temporal ordering

```python
class TCNAnomalyDetector:
    - TemporalBlock(2 → 16, dilation=1)
    - TemporalBlock(16 → 32, dilation=2)
    - TemporalBlock(32 → 64, dilation=4)
    - GlobalAvgPool2d + Linear(64 → 2)
```

## Anomaly Generation Strategy

### Systematic Anomaly Injection
Instead of relying on natural anomalies (which are rare), we artificially inject known anomaly patterns:

### 1. Blackout Regions
**Purpose**: Simulate sensor failures or occlusions
```python
def add_blackout_region(frame, region_size, intensity):
    """Reduce pixel intensities in specified region"""
    frame[y:y+h, x:x+w] *= (1 - intensity)
```
**Parameters:**
- Region size: 10-40 pixels
- Intensity: 0.7-1.0 (70-100% reduction)
- Position: Random placement

### 2. Vibration Noise
**Purpose**: Simulate camera shake or mechanical vibrations
```python
def add_vibration_noise(frame, region_size, intensity):
    """Add Gaussian noise to simulate vibrations"""
    noise = torch.randn(region_size) * intensity
    frame[region] += noise
```
**Parameters:**
- Noise intensity: 0.3-0.7
- Spatial extent: 20-60 pixels
- Distribution: Gaussian noise

### 3. Polarity Flipping
**Purpose**: Simulate hardware errors in event generation
```python
def flip_polarities(frame, region_size, flip_prob):
    """Swap positive/negative event channels"""
    pos_events ↔ neg_events (with probability flip_prob)
```
**Parameters:**
- Flip probability: 0.6-0.9
- Affected region: 15-45 pixels
- Channel swapping: pos ↔ neg

### Anomaly Distribution
- **50% Normal frames**: Original MVSEC data
- **50% Anomalous frames**: Equal distribution of 3 anomaly types
- **Random placement**: Ensures diverse anomaly patterns
- **Intensity variation**: Prevents overfitting to specific anomaly strengths

## Training and Evaluation

### Training Strategy
```python
def train_model(model, train_loader, val_loader, epochs):
    criterion = CrossEntropyLoss()
    optimizer = Adam(lr=0.001)
    scheduler = ReduceLROnPlateau()
```

**Key Aspects:**
- **Loss Function**: Cross-entropy for binary classification
- **Optimizer**: Adam with learning rate 0.001
- **Regularization**: Learning rate scheduling
- **Validation**: Monitor overfitting with validation loss

### Evaluation Metrics
1. **Accuracy**: Overall classification correctness
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **F1-Score**: Harmonic mean of precision and recall
5. **ROC-AUC**: Area under receiver operating characteristic curve
6. **Confusion Matrix**: Detailed classification breakdown

### Performance Comparison Framework
```python
def compare_models(snn_metrics, rnn_metrics, tcn_metrics):
    """Generate comparative analysis and ROC curves"""
```

## Code Structure

### File Organization
```
anomaly_detection.ipynb
├── Environment Setup
├── Data Loading Functions
├── Preprocessing Classes
├── Anomaly Generation
├── Model Implementations
├── Training/Evaluation
├── Visualization Tools
└── Main Pipeline
```

### Key Classes and Functions

#### Data Handling
- `load_mvsec_data()`: HDF5 file parsing
- `MVSECDataHandler`: Complete data pipeline management
- `EventAnomalyDataset`: PyTorch dataset with anomaly injection

#### Models
- `SpikingAnomalyDetector`: SNN implementation
- `RNNAnomalyDetector`: RNN-based detector
- `TCNAnomalyDetector`: Temporal CNN detector

#### Training
- `train_model()`: Unified training loop
- `evaluate_model()`: Validation and testing
- `test_model()`: Comprehensive metric calculation

#### Visualization
- `visualize_event_frame()`: Event data visualization
- `plot_metrics()`: Training curve plotting
- `plot_confusion_matrix()`: Classification analysis
- `plot_roc_curves()`: ROC comparison

### Memory Management Strategy
1. **Event Sampling**: Limit to 500K events per sequence
2. **Frame Caching**: Process in batches to avoid memory overflow
3. **Model Checkpointing**: Save intermediate results
4. **Garbage Collection**: Explicit memory cleanup

## Usage Guide

### Quick Start
```python
# 1. Test data loading
events, sensor_size = load_mvsec_data('./data', 'indoor_flying', 'left')

# 2. Run pipeline
results = run_mvsec_anomaly_detection_pipeline(
    data_path='./data',
    sequence='indoor_flying',
    num_epochs=5
)

# 3. Analyze results
print(f"Best model: {results['best_model']}")
```

### Configuration Options
```python
CONFIG = {
    'sequence': 'indoor_flying',  # or 'outdoor_day', 'outdoor_night'
    'camera': 'left',             # or 'right'
    'sensor_size': (64, 64),      # target frame resolution
    'num_frames': 50,             # temporal sequence length
    'max_events': 500000,         # memory management
    'anomaly_ratio': 0.5,         # percentage of anomalous data
    'batch_size': 8,              # training batch size
    'num_epochs': 3               # training iterations
}
```

### Custom Anomaly Types
```python
# Add new anomaly type
def add_custom_anomaly(frame, **params):
    """Implement custom anomaly pattern"""
    return modified_frame, anomaly_mask

# Register in AnomalyGenerator
anomaly_gen.add_anomaly_type('custom', add_custom_anomaly)
```

## Performance Analysis

### Expected Results
Based on the architecture and data characteristics:

#### SNN Performance
- **Advantages**:
  - Natural fit for event-based data
  - Energy-efficient spike-based computation
  - Temporal dynamics modeling
- **Challenges**:
  - Training complexity with surrogate gradients
  - Limited optimization compared to standard networks
- **Expected Accuracy**: 75-85%

#### RNN Performance
- **Advantages**:
  - Proven temporal sequence modeling
  - Well-established training procedures
  - Good baseline performance
- **Challenges**:
  - Vanishing gradient problems
  - Sequential processing limitations
- **Expected Accuracy**: 80-90%

#### TCN Performance
- **Advantages**:
  - Parallel processing capabilities
  - Long-range temporal dependencies
  - Stable training dynamics
- **Challenges**:
  - Higher computational requirements
  - Memory intensive for long sequences
- **Expected Accuracy**: 85-95%

### Benchmarking Strategy
1. **Cross-validation**: 5-fold validation for robust metrics
2. **Statistical significance**: Multiple runs with different seeds
3. **Computational efficiency**: Training time and memory usage
4. **Scalability testing**: Performance with larger datasets

## Future Improvements

### Short-term Enhancements
1. **Data Augmentation**:
   - Temporal jittering
   - Spatial transformations
   - Event dropout

2. **Advanced Anomalies**:
   - Motion-based anomalies
   - Object-specific anomalies
   - Contextual anomalies

3. **Model Improvements**:
   - Attention mechanisms
   - Multi-scale processing
   - Ensemble methods

### Long-term Research Directions
1. **Unsupervised Learning**:
   - Autoencoder-based anomaly detection
   - Generative adversarial networks
   - Self-supervised learning

2. **Real-time Processing**:
   - Online learning capabilities
   - Streaming data processing
   - Edge deployment optimization

3. **Multi-modal Integration**:
   - Fusion with traditional cameras
   - IMU data integration
   - LiDAR correlation

4. **Advanced SNN Features**:
   - Adaptive thresholds
   - Plasticity mechanisms
   - Neuromorphic hardware deployment

### Scalability Considerations
1. **Distributed Training**: Multi-GPU support for larger datasets
2. **Cloud Integration**: AWS/Azure deployment strategies
3. **Production Pipeline**: MLOps integration and monitoring
4. **Real-world Validation**: Testing with actual anomalous scenarios

## Conclusion

This anomaly detection system provides a comprehensive framework for evaluating different neural architectures on event-based data. The systematic approach to anomaly generation, combined with proper MVSEC data handling, creates a solid foundation for neuromorphic computer vision research.

The comparison between SNN, RNN, and TCN architectures offers insights into the strengths and limitations of each approach for temporal anomaly detection in event-based data streams.
