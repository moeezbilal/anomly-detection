# Comprehensive Neuromorphic Anomaly Detection Analysis

## 🎯 Overview

This directory contains analysis scripts that complement the main notebook implementations. The analysis aligns with the **two distinct research approaches** implemented in the project:

### 📊 **Architecture-Based Analysis** (Complements `anomaly_detection.ipynb`)
- **Focus**: Neural network architecture performance comparison
- **Approach**: SNN vs RNN vs TCN evaluation framework
- **Method**: End-to-end deep learning with comprehensive metrics
- **Alignment**: Extends the architecture comparison from the main notebook

### 🤖 **Feature-Based Analysis** (Complements `rq1_spatiotemporal_vs_basic_features.ipynb`)
- **Focus**: Feature engineering effectiveness analysis
- **Approach**: Basic vs Spatiotemporal feature comparison
- **Method**: Traditional ML with engineered features (35 total features)
- **Alignment**: Extends the feature comparison from the dedicated notebook

## 📋 **Connection to Main Notebooks**

This analysis directory provides **automated scripts** that replicate and extend the manual analysis performed in:
- `../notebooks/anomaly_detection.ipynb` - Manual architecture comparison
- `../notebooks/rq1_spatiotemporal_vs_basic_features.ipynb` - Manual feature analysis
- `../notebooks/Notebook_Comparison_Analysis.tex` - Comparative analysis documentation

## 🔄 **Analysis Structure**

**Two-Phase Approach**:
- **Phase 1**: Feature engineering impact analysis (like `rq1_spatiotemporal_vs_basic_features.ipynb`)
- **Phase 2**: Architecture approach comparison (like `anomaly_detection.ipynb`)
- **Integration**: Combined insights from both approaches

**Benefits**:
- **Automated Execution**: Scripts run the analysis automatically
- **Reproducible Results**: Consistent evaluation across approaches
- **Extended Analysis**: Additional algorithms and feature combinations

## 📁 File Structure

```
comprehensive_analysis/
├── README.md                           # This file
├── run_comprehensive_analysis.py       # Main orchestrator script
├── rq1_feature_comparison.py          # Part A: Feature analysis
├── rq1_algorithm_comparison.py        # Part B: Algorithm analysis
└── run_rq1_comprehensive.py          # Original combined script (moved here)
```

## 🚀 How to Run

### Option 1: Run Complete Analysis
```bash
cd comprehensive_analysis
python run_comprehensive_analysis.py
```

### Option 2: Run Individual Parts
```bash
# Part A only - Feature comparison
python rq1_feature_comparison.py

# Part B only - Algorithm comparison
python rq1_algorithm_comparison.py
```

## 📊 Expected Outputs

### Part A Outputs:
- `feature_comparison_results.csv` - Detailed feature performance data
- `feature_comparison_results.png` - Feature comparison visualizations

### Part B Outputs:
- `algorithm_comparison_results.csv` - Detailed algorithm performance data
- `algorithm_comparison_results.png` - Algorithm comparison visualizations

### Combined Outputs:
- `comprehensive_analysis_results.csv` - All results combined
- `comprehensive_analysis_summary.png` - Summary visualization with recommendations

## 🧠 Analysis Logic

### Part A: Feature Engineering
```
Same Algorithms + Different Features = Feature Impact
├── Basic Features (15) → [RF, SVM, LogReg, GradBoost]
├── Spatiotemporal Features (20) → [RF, SVM, LogReg, GradBoost]
└── Neuromorphic Features (29) → [RF, SVM, LogReg, GradBoost]
```

### Part B: Algorithm Approach
```
Best Features + Different Approaches = Algorithm Impact
├── Supervised Classification → [RF, SVM, LogReg, GradBoost, XGBoost, LightGBM]
└── Unsupervised Anomaly Detection → [Isolation Forest, One-Class SVM, LOF]
```

## 💡 Key Insights Expected

### Feature Analysis (Part A):
- **Neuromorphic features** likely to outperform due to domain specificity
- Time surfaces, event clustering capture event-camera patterns
- Computational overhead vs accuracy trade-off analysis

### Algorithm Analysis (Part B):
- **Anomaly detection** may outperform classification for novelty detection
- Unsupervised methods good for unknown anomaly types
- Supervised methods good for known patterns

### Combined Recommendation:
- **Best Feature Type** + **Best Algorithm Approach** = Optimal deployment solution

## 🔧 Technical Details

### Dependencies:
- torch, numpy, matplotlib, pandas, scipy
- scikit-learn (core algorithms)
- h5py (MVSEC data loading)
- xgboost, lightgbm (optional, advanced methods)
- opencv-python (optional, for optical flow)

### Data Requirements:
- MVSEC dataset in `../data/` directory (HDF5 format)
- Falls back to synthetic data if MVSEC unavailable

### Computational Requirements:
- ~2-5 minutes total runtime on modern hardware
- Memory: ~1-2GB for feature extraction
- Storage: ~50MB for all outputs

## 📈 Performance Expectations

Based on neuromorphic vision research:

### Feature Performance:
- **Basic**: 0.6-0.7 F1-Score (baseline)
- **Spatiotemporal**: 0.7-0.8 F1-Score (+15-20% improvement)
- **Neuromorphic**: 0.8-0.9 F1-Score (+25-35% improvement)

### Algorithm Performance:
- **Supervised Classification**: Good for known anomaly types
- **Unsupervised Anomaly Detection**: Better for novel/unknown anomalies
- **Expected**: 10-20% additional improvement from optimal algorithm choice

## 🎯 Use Cases

### Research:
- Feature engineering impact quantification
- Algorithm comparison for neuromorphic data
- Baseline establishment for neuromorphic anomaly detection

### Deployment:
- Feature selection for production systems
- Algorithm choice for specific applications
- Performance vs computational cost analysis

## 🔄 Extensibility

### Adding New Features:
1. Extend feature extractor classes
2. Update feature_names lists
3. Ensure consistent extraction interface

### Adding New Algorithms:
1. Add to appropriate category (supervised/unsupervised)
2. Follow existing parameter optimization patterns
3. Ensure consistent evaluation metrics

### Adding New Data:
1. Implement data loader following MVSEC pattern
2. Ensure consistent event format
3. Update path configurations

---

This restructured analysis provides **clear separation of concerns** and **actionable insights** for both research and deployment scenarios in neuromorphic anomaly detection.
