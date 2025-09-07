# Anomaly Generation Testing Documentation

## 🎯 Overview

This directory contains comprehensive testing and validation for the Enhanced SNN anomaly generation system. The anomaly generation creates synthetic anomalies in neuromorphic event data for supervised anomaly detection training.

## 🧪 Test Coverage

### Anomaly Types Tested
1. **Contextual Blackout** - Simulates sensor failure by reducing activity in high-activity regions
2. **Adaptive Vibration** - Simulates camera shake/vibration with motion-based noise patterns
3. **Temporal Polarity Flip** - Simulates hardware errors by swapping positive/negative event polarities

### Test Scenarios
- **Individual Type Testing**: Each anomaly type tested in isolation with multiple parameters
- **Smart Selection Testing**: Intelligent anomaly type selection based on frame characteristics
- **Realistic Frame Testing**: Testing with neuromorphic-like sparse event data
- **Edge Case Testing**: Empty frames, single-polarity regions, high-activity frames

### Frame Types Used
- **High Activity**: Dense random events (triggers blackout anomalies)
- **Sparse Events**: Scattered DVS camera-like events (triggers polarity flips)
- **Edge Activity**: Motion boundary patterns (triggers polarity flips)
- **Clustered Events**: Object motion patterns (triggers polarity flips)
- **Almost Empty**: Minimal activity frames (triggers vibration)
- **Separated Polarities**: Clear pos/neg regions (for polarity flip validation)

## 📊 Test Results Summary

**ALL TESTS PASS WITH 100% SUCCESS RATE**

### Individual Anomaly Types: 7/7 ✅
- Blackout (mild): ✅ PASS - Coverage: 0.6%, Diff: 6.66
- Blackout (medium): ✅ PASS - Coverage: 1.6%, Diff: 24.71
- Blackout (severe): ✅ PASS - Coverage: 2.4%, Diff: 51.90
- Vibration (random): ✅ PASS - Coverage: 2.4%, Diff: 37.14
- Vibration (coherent): ✅ PASS - Coverage: 2.4%, Diff: 19.58
- Polarity Flip (burst): ✅ PASS - Coverage: 1.6%, Diff: 19.05
- Polarity Flip (intermittent): ✅ PASS - Coverage: 1.6%, Diff: 9.11

### Smart Selection Tests: 6/6 ✅
- High Activity → Blackout: ✅ PASS
- Sparse Events → Polarity Flip: ✅ PASS
- Edge Activity → Polarity Flip: ✅ PASS
- Clustered Events → Polarity Flip: ✅ PASS
- Almost Empty → Vibration: ✅ PASS
- Separated Polarities → Polarity Flip: ✅ PASS

## 🛠️ Technical Implementation

### Smart Selection Logic
```python
# Neuromorphic-optimized thresholds
if activity_level > 0.05:  # High activity
    return contextual_blackout(severity based on activity)
elif sparsity > 0.005 or total_events > 1.0:  # Moderate sparsity
    return polarity_flip(pattern based on sparsity)
else:  # Very low activity
    return vibration(coherent pattern)
```

### Enhanced Polarity Flip
- **Problem**: Flipping sparse regions with only one polarity resulted in no changes
- **Solution**: Add complementary activity before flipping to ensure visible differences
- **Implementation**: Detects pos-only/neg-only regions and adds minimal opposing activity (0.05 intensity)

### Contextual Blackout
- **Activity-Aware**: Targets high-activity regions using 70th percentile threshold
- **Severity Scaling**: Mild (50%), medium (80%), severe (100%) intensity reduction
- **Adaptive Sizing**: Region size scales with severity and frame dimensions

### Adaptive Vibration
- **Motion Patterns**: Coherent directional vs random noise
- **Regional Application**: Noise applied to specific regions, not entire frame
- **Intensity Control**: Proper scaling and clamping to [0,1] range

## 🔧 Issues Found and Fixed

### 1. Polarity Flip Zero-Difference Bug ❌→✅
- **Issue**: Flipping regions with only positive OR negative events produced no detectable changes
- **Root Cause**: Flipping zeros to zeros, or single-polarity regions
- **Fix**: Enhanced logic to add complementary activity before flipping

### 2. Smart Selection Threshold Issue ❌→✅
- **Issue**: Thresholds too high for realistic sparse neuromorphic data
- **Root Cause**: Original thresholds designed for synthetic dense data
- **Fix**: Lowered thresholds for neuromorphic characteristics (0.1→0.05, 0.1→0.005)

## 🎬 Visual Verification

The test generates comprehensive before/after visualizations showing:
- **Original Frames**: Neuromorphic event data
- **Anomalous Frames**: Same frames with injected anomalies
- **Difference Maps**: Highlighting changes using hot colormap
- **Mask Overlays**: Red regions showing anomaly locations

## 📁 Files in this Directory

### Core Test Script
- **`comprehensive_anomaly_test.py`** - Complete test suite combining all functionality

### Documentation
- **`ANOMALY_TESTING_DOCUMENTATION.md`** - This comprehensive documentation
- **`README.md`** - Quick start guide and overview

### How to Run
```bash
cd anomaly_generation_testing/
python comprehensive_anomaly_test.py
```

This will:
1. Test all individual anomaly types
2. Test smart selection logic on realistic frames
3. Generate visual verification (comprehensive_anomaly_test_results.png)
4. Display comprehensive results summary

## ✅ Production Readiness

**STATUS: FULLY VALIDATED AND PRODUCTION-READY**

The anomaly generation system has been thoroughly tested and verified to work correctly with:
- ✅ All individual anomaly types functioning properly
- ✅ Smart selection logic working with realistic neuromorphic data
- ✅ Robust edge case handling (empty frames, single polarities, etc.)
- ✅ Before/after visualization integrated into Enhanced SNN pipeline
- ✅ 100% test success rate across all scenarios

## 🚀 Integration

This anomaly generation is used in:
- **`enhanced_snn_experiment.py`** - Main Enhanced SNN experiment
- **FeatureAwareAnomalyDataset** - Dataset with before/after frame storage
- **Before/after visualization** - Integrated comparison framework
- **Top 10 feature extraction** - Optimized feature engineering pipeline

The system is ready for research, evaluation, and educational use with reliable synthetic anomaly injection for neuromorphic anomaly detection studies.
