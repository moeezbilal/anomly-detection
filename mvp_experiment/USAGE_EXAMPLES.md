# Enhanced MVP Experiment - Usage Examples

## 🚀 Getting Started

### Quick Start Commands
```bash
# Navigate to experiment directory
cd mvp_experiment

# Run complete enhanced experiment (recommended)
python enhanced_experiment_fixed.py

# Test individual components first
python test_enhanced_features.py

# Run original experiment for comparison
python mvp_feature_comparison.py
```

## 📊 Basic Usage Examples

### 1. Run Complete Enhanced Experiment
```python
from enhanced_experiment_fixed import run_fixed_enhanced_experiment

# Run with default parameters
results = run_fixed_enhanced_experiment()

# Access results
print(f"Best overall model: {results['best_model']}")
print(f"Best individual features: {results['best_individual']}")
print(f"Feature importance winner: {results['importance_winner']}")
print(f"Total experiment time: {results['total_time']:.2f}s")

# Access detailed metrics
basic_acc = results['basic_metrics']['accuracy']
neuro_acc = results['neuro_metrics']['accuracy']
combined_acc = results['combined_metrics']['accuracy']

print(f"Accuracy comparison:")
print(f"  Basic: {basic_acc:.1%}")
print(f"  Neuromorphic: {neuro_acc:.1%}")
print(f"  Combined: {combined_acc:.1%}")
```

### 2. Custom Experiment Configuration
```python
# Run with custom parameters
results = run_fixed_enhanced_experiment(
    data_path="./my_data",           # Custom data directory
    sequence="outdoor_day",          # Different MVSEC sequence
    num_frames=100                   # More frames for larger dataset
)
```

## 🔧 Component-Level Usage

### 1. Enhanced Feature Extraction
```python
from mvp_feature_comparison import EnhancedFeatureExtractor
import torch
import numpy as np

# Initialize extractor
extractor = EnhancedFeatureExtractor()

# Create sample neuromorphic frames (2 channels: pos/neg events)
frame_current = torch.rand(2, 64, 64) * 0.1
frame_previous = torch.rand(2, 64, 64) * 0.1

# Extract all enhanced features
all_features = extractor.extract_all_features(frame_current, frame_previous)
print(f"Extracted {len(all_features)} features: {all_features}")

# Extract specific feature groups
basic_features = extractor.extract_basic_features(frame_current)
spatio_features = extractor.extract_spatiotemporal_features(frame_current, frame_previous)
neuro_features = extractor.extract_neuromorphic_features(frame_current)

print("Feature breakdown:")
print(f"  Basic: {basic_features}")
print(f"  Spatiotemporal: {spatio_features}")
print(f"  Neuromorphic: {neuro_features}")
```

### 2. Enhanced Training Pipeline
```python
from mvp_feature_comparison import EnhancedTrainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.rand(200, 13)  # 200 samples, 13 features
y = np.random.randint(0, 2, 200)  # Binary labels

# Initialize enhanced trainer
trainer = EnhancedTrainer(
    n_estimators=100,      # Number of boosting stages
    max_epochs=150,        # Maximum training epochs
    cv_folds=5,            # Cross-validation folds
    random_state=42        # Reproducibility
)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train with cross-validation and learning rate optimization
model, cv_score = trainer.train_with_cv(X_train_scaled, y_train, "my_model")
print(f"Best CV score: {cv_score:.4f}")
print(f"Optimal parameters: {trainer.best_params['my_model']}")

# Fit final model and evaluate
model.fit(X_train_scaled, y_train)
metrics = trainer.evaluate_model(model, X_test_scaled, y_test)

print("Test results:")
for metric, value in metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {metric}: {value:.4f}")
```

### 3. Individual Feature Analysis
```python
# Analyze specific features
extractor = EnhancedFeatureExtractor()

# Get feature information
print("Feature groups and names:")
print(f"Basic ({len(extractor.basic_names)}): {extractor.basic_names}")
print(f"Spatiotemporal ({len(extractor.spatiotemporal_names)}): {extractor.spatiotemporal_names}")
print(f"Neuromorphic ({len(extractor.neuromorphic_names)}): {extractor.neuromorphic_names}")

# Test individual feature methods
frame = torch.rand(2, 32, 32) * 0.2

# Test basic features
basic_feats = extractor.extract_basic_features(frame)
print(f"\nBasic features: {dict(zip(extractor.basic_names, basic_feats))}")

# Test neuromorphic features
neuro_feats = extractor.extract_neuromorphic_features(frame)
print(f"Neuromorphic features: {dict(zip(extractor.neuromorphic_names, neuro_feats))}")
```

## 📈 Advanced Usage Patterns

### 1. Batch Processing Multiple Sequences
```python
def process_multiple_sequences():
    """Process multiple MVSEC sequences and compare results"""
    sequences = ["indoor_flying", "outdoor_day", "outdoor_night"]
    results_comparison = {}

    for seq in sequences:
        try:
            print(f"Processing {seq}...")
            results = run_fixed_enhanced_experiment(
                sequence=seq,
                num_frames=30
            )
            results_comparison[seq] = results
        except Exception as e:
            print(f"Failed to process {seq}: {e}")
            continue

    # Compare results across sequences
    for seq, results in results_comparison.items():
        best_acc = max(
            results['basic_metrics']['accuracy'],
            results['spatio_metrics']['accuracy'],
            results['neuro_metrics']['accuracy']
        )
        print(f"{seq}: Best accuracy = {best_acc:.1%}")

    return results_comparison

# Run batch processing
batch_results = process_multiple_sequences()
```

### 2. Feature Importance Analysis
```python
def analyze_feature_importance(results):
    """Detailed feature importance analysis"""
    from mvp_feature_comparison import EnhancedFeatureExtractor

    extractor = EnhancedFeatureExtractor()
    feature_names = extractor.get_feature_names()

    # Get feature importance from combined model results
    # (This would need to be extracted from the actual trained model)
    print("Feature Importance Analysis:")
    print("=" * 40)

    # Example analysis structure
    groups = {
        'Basic': extractor.basic_names,
        'Spatiotemporal': extractor.spatiotemporal_names,
        'Neuromorphic': extractor.neuromorphic_names
    }

    for group_name, features in groups.items():
        print(f"\n{group_name} Features:")
        for i, feature in enumerate(features):
            print(f"  {i+1}. {feature.replace('_', ' ').title()}")

# Example usage
# analyze_feature_importance(results)
```

### 3. Custom Anomaly Generation
```python
from mvp_feature_comparison import SimpleAnomalyGenerator
import torch

# Initialize anomaly generator
anomaly_gen = SimpleAnomalyGenerator(seed=42)

# Create sample frame
original_frame = torch.rand(2, 64, 64) * 0.1

# Generate different types of anomalies
anomaly_types = []

# Blackout anomaly
blackout_frame, blackout_mask, _ = anomaly_gen.add_blackout(original_frame, intensity=0.8)
anomaly_types.append(("Blackout", blackout_frame, blackout_mask))

# Vibration anomaly
vibration_frame, vibration_mask, _ = anomaly_gen.add_vibration(original_frame, intensity=0.5)
anomaly_types.append(("Vibration", vibration_frame, vibration_mask))

# Polarity flip anomaly
flip_frame, flip_mask, _ = anomaly_gen.flip_polarities(original_frame, flip_prob=0.7)
anomaly_types.append(("Flip", flip_frame, flip_mask))

# Extract features for each anomaly type
extractor = EnhancedFeatureExtractor()
for anomaly_name, anomaly_frame, mask in anomaly_types:
    features = extractor.extract_all_features(anomaly_frame)
    print(f"{anomaly_name} anomaly features: {features[:5]}...")  # First 5 features
```

## 🔍 Debugging and Validation

### 1. Feature Validation
```python
def validate_features():
    """Comprehensive feature validation"""
    from test_enhanced_features import test_enhanced_features, test_trainer

    print("🧪 Running feature validation...")

    # Test feature extraction
    if test_enhanced_features():
        print("✅ Feature extraction validated")
    else:
        print("❌ Feature extraction failed")
        return False

    # Test training pipeline
    if test_trainer():
        print("✅ Training pipeline validated")
    else:
        print("❌ Training pipeline failed")
        return False

    print("🎉 All validations passed!")
    return True

# Run validation
validate_features()
```

### 2. Performance Profiling
```python
import time
import numpy as np

def profile_feature_extraction():
    """Profile feature extraction performance"""
    extractor = EnhancedFeatureExtractor()

    # Generate test frames
    frames = [torch.rand(2, 64, 64) * 0.1 for _ in range(100)]

    # Profile basic features
    start_time = time.time()
    for frame in frames:
        basic_features = extractor.extract_basic_features(frame)
    basic_time = time.time() - start_time

    # Profile neuromorphic features
    start_time = time.time()
    for frame in frames:
        neuro_features = extractor.extract_neuromorphic_features(frame)
    neuro_time = time.time() - start_time

    # Profile all features
    start_time = time.time()
    for i, frame in enumerate(frames):
        prev_frame = frames[i-1] if i > 0 else None
        all_features = extractor.extract_all_features(frame, prev_frame)
    all_time = time.time() - start_time

    print("Performance Profile:")
    print(f"  Basic features: {basic_time:.3f}s ({100/basic_time:.1f} FPS)")
    print(f"  Neuromorphic features: {neuro_time:.3f}s ({100/neuro_time:.1f} FPS)")
    print(f"  All features: {all_time:.3f}s ({100/all_time:.1f} FPS)")

# Run profiling
profile_feature_extraction()
```

## 🎯 Experiment Customization

### 1. Custom Feature Weights
```python
def weighted_feature_experiment():
    """Example of using feature weights for analysis"""
    # This is a conceptual example - actual implementation would require
    # modifying the enhanced experiment to accept feature weights

    feature_weights = {
        # Emphasize neuromorphic features
        'spike_entropy': 2.0,
        'lif_response_mean': 2.0,
        'polarity_synchrony': 1.5,
        'event_clustering_density': 1.5,
        # Standard weights for others
        'spatial_max': 1.0,
        'isi_mean': 1.0,
        # ... etc
    }

    print("Feature weights configuration:")
    for feature, weight in feature_weights.items():
        print(f"  {feature}: {weight}")

    # In actual implementation, these weights would be applied during training
    return feature_weights
```

### 2. Parameter Sensitivity Analysis
```python
def parameter_sensitivity_analysis():
    """Analyze sensitivity to key parameters"""

    # Test different numbers of frames
    frame_counts = [20, 30, 50, 75, 100]
    results_by_frames = {}

    for num_frames in frame_counts:
        print(f"Testing with {num_frames} frames...")
        try:
            results = run_fixed_enhanced_experiment(num_frames=num_frames)
            best_acc = max(
                results['basic_metrics']['accuracy'],
                results['spatio_metrics']['accuracy'],
                results['neuro_metrics']['accuracy']
            )
            results_by_frames[num_frames] = best_acc
        except Exception as e:
            print(f"Failed with {num_frames} frames: {e}")

    print("\\nAccuracy vs Frame Count:")
    for frames, accuracy in results_by_frames.items():
        print(f"  {frames} frames: {accuracy:.1%}")

    return results_by_frames
```

## 📝 Output Interpretation

### Understanding Results Output
```python
def interpret_results(results):
    """Helper function to interpret experiment results"""

    print("🎯 EXPERIMENT RESULTS INTERPRETATION")
    print("=" * 50)

    # Best model identification
    best_model = results['best_model']
    print(f"Best Overall Model: {best_model}")

    # Performance comparison
    models = ['basic', 'spatio', 'neuro', 'combined']
    print("\\nPerformance Comparison:")
    for model in models:
        if f'{model}_metrics' in results:
            metrics = results[f'{model}_metrics']
            acc = metrics['accuracy']
            cv = metrics.get('cv_score', 'N/A')
            print(f"  {model.title():12s}: {acc:.1%} accuracy (CV: {cv})")

    # Feature importance insights
    importance_winner = results['importance_winner']
    print(f"\\nFeature Importance Winner: {importance_winner}")

    # Training efficiency
    total_time = results['total_time']
    print(f"\\nTotal Experiment Time: {total_time:.1f} seconds")

    # Practical recommendations
    print("\\n💡 PRACTICAL INSIGHTS:")
    if results['best_individual'] == 'Neuromorphic':
        print("   → Neuromorphic features show superior performance")
        print("   → Consider brain-inspired processing for similar applications")
    elif results['best_individual'] == 'Spatiotemporal':
        print("   → Temporal dynamics are crucial for this data")
        print("   → Focus on motion and flow analysis")
    else:
        print("   → Basic features provide competitive performance")
        print("   → Simple statistics may be sufficient")

# Example usage with actual results
# interpret_results(results)
```

---

## 🚀 Next Steps

After running the experiments, consider:

1. **Scale to Real Data**: Test with actual MVSEC sequences
2. **Hyperparameter Tuning**: Optimize classifier parameters further
3. **Feature Engineering**: Develop domain-specific neuromorphic features
4. **Real-time Implementation**: Optimize for streaming data processing
5. **Hardware Deployment**: Test on neuromorphic computing platforms

For more technical details, see `ENHANCED_FEATURES_DOCUMENTATION.md`.
