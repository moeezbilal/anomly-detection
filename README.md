# MVSEC Anomaly Detection with Neural Networks

This project implements a comprehensive anomaly detection system for neuromorphic event-based data using multiple neural network architectures, with a focus on comparing Spiking Neural Networks (SNNs) with traditional approaches.

## 🎯 **Project Overview**

**Two Complementary Research Approaches:**

### **Architecture Comparison** (`anomaly_detection.ipynb`)
- **Focus**: Neural network architecture comparison (SNN vs RNN vs TCN)
- **Method**: End-to-end deep learning with minimal feature engineering
- **Output**: Performance ranking of architectures with comprehensive metrics

### **Feature Engineering Analysis** (`rq1_spatiotemporal_vs_basic_features.ipynb`)  
- **Focus**: Feature comparison (basic vs spatiotemporal features)
- **Method**: Traditional ML with extensive feature engineering (35 features)
- **Output**: Feature effectiveness analysis with computational efficiency metrics

**Key Features:**
- **Real MVSEC Data Integration**: Native HDF5 processing and event handling
- **Systematic Anomaly Generation**: Three distinct anomaly types with balanced datasets
- **Comprehensive Analysis**: Statistical comparison with detailed visualizations
- **Complete Documentation**: 80+ pages of technical guides and analysis

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.12+
- Poetry (for dependency management)
- MVSEC dataset files in `data/` directory

### **Installation**
```bash
# Install dependencies
poetry install

# Activate environment
poetry shell
```

### **Usage**
```bash
# Architecture comparison approach
jupyter notebook notebooks/anomaly_detection.ipynb

# Feature engineering analysis approach  
jupyter notebook notebooks/rq1_spatiotemporal_vs_basic_features.ipynb

# View notebook comparison analysis
open notebooks/Notebook_Comparison_Analysis.tex
```

### **LaTeX Documents**
```bash
# Professional LaTeX documents ready for Overleaf
cd overleaf/

# Thesis project plan (official academic document)
# Upload 2025_MA_Moeez_Thesis_Project_Plan.tex to Overleaf

# Complete technical code guide (80+ pages)
# Upload MVSEC_Anomaly_Detection_Code_Guide.tex to Overleaf
```

## 📁 **Complete Project Structure**

The project follows a clean, organized structure with clear separation between implementation, documentation, and analysis:

```
kth/all_llms/
├── 📋 README.md                          # This file (main project overview)
├── ⚙️ pyproject.toml                     # Poetry configuration
├── 🔒 poetry.lock                        # Dependency lock file
│
├── 📓 notebooks/                         # Jupyter notebooks  
│   ├── anomaly_detection.ipynb          # Architecture comparison approach (SNN/RNN/TCN)
│   ├── rq1_spatiotemporal_vs_basic_features.ipynb  # Feature analysis approach
│   └── Notebook_Comparison_Analysis.tex # Detailed comparison of both approaches
│
├── 📄 overleaf/                          # LaTeX documents for Overleaf
│   ├── 2025_MA_Moeez_Thesis_Project_Plan.tex     # Official thesis project plan
│   ├── MVSEC_Anomaly_Detection_Code_Guide.tex    # Technical code guide (80+ pages)
│   └── README.md                         # LaTeX compilation instructions
│
├── 📊 data/                              # MVSEC dataset files
│   ├── indoor_flying2_data-002.hdf5     # Event data files
│   ├── indoor_flying3_data-003.hdf5
│   ├── outdoor_day1_data-008.hdf5
│   ├── outdoor_night1_data-009.hdf5
│   └── [additional MVSEC files...]
│
├── 📚 docs/                              # Complete documentation package
│   ├── 📑 INDEX.md                       # Master documentation index (navigation hub)
│   ├── 📄 MVSEC_Anomaly_Detection_Documentation.md  # Technical specs (45+ pages)
│   ├── 📋 Project_Summary.md             # Executive summary (15+ pages)
│   │
│   ├── 🏗️ code_structure/                # Code architecture documentation
│   │   └── 📝 Code_Structure_Documentation.md  # Implementation guide (20+ pages)
│   │
│   ├── 🎨 diagrams/                      # Visual documentation
│   │   ├── 🐍 system_architecture_diagram.py    # Diagram generator script
│   │   ├── 🖼️ system_architecture.png          # High-level system overview
│   │   ├── 🖼️ data_flow_diagram.png            # Data processing pipeline
│   │   ├── 🖼️ model_architectures.png          # Neural network comparison
│   │   ├── 🖼️ anomaly_strategy.png             # Anomaly generation visualization
│   │   └── 🖼️ evaluation_framework.png         # Performance evaluation methodology
│   │
│   └── 📊 analysis/                      # Analysis results (generated at runtime)
│       └── [Runtime-generated analysis files]
│
├── 💻 src/                               # Source code (if extracted)
│   └── all_llms/
│       └── __init__.py
│
├── 🧪 tests/                             # Unit tests
│   └── __init__.py
│
└── 🔬 comprehensive_analysis/           # Automated analysis scripts
    ├── README.md                        # Analysis directory guide  
    ├── run_comprehensive_analysis.py   # Main orchestrator script
    ├── rq1_feature_comparison.py      # Feature analysis automation
    └── rq1_algorithm_comparison.py    # Algorithm comparison automation
```

### **🎯 Organization Benefits**
- **🏗️ Clear Hierarchy**: Main directory → Implementation → Documentation → Analysis
- **📚 Streamlined Navigation**: Direct path from README → docs/INDEX.md → Specific documentation
- **🔬 Dual Research Support**: Both architecture and feature analysis approaches clearly organized
- **📋 Professional Structure**: Industry-standard layout with comprehensive documentation

## 📚 **Documentation**

**→ [Master Documentation Index](./docs/INDEX.md)** ← **Start Here for All Documentation**

### **Essential Documents**
- **[Notebook Comparison Analysis](./notebooks/Notebook_Comparison_Analysis.tex)**: Detailed comparison of the two research approaches
- **[Technical Documentation](./docs/MVSEC_Anomaly_Detection_Documentation.md)**: Complete system specifications (45+ pages)
- **[Project Summary](./docs/Project_Summary.md)**: Executive overview and achievements
- **[LaTeX Documents](./overleaf/)**: Professional academic documents ready for Overleaf

### **Quick Access by Audience**
- **🆕 New Users**: Start with this README → [docs/INDEX.md](./docs/INDEX.md) → [Project Summary](./docs/Project_Summary.md)
- **🔬 Researchers**: [Notebook Comparison](./notebooks/Notebook_Comparison_Analysis.tex) → [Technical Docs](./docs/MVSEC_Anomaly_Detection_Documentation.md)
- **💻 Developers**: [Code Structure Guide](./docs/code_structure/Code_Structure_Documentation.md) → [Implementation Notebooks](./notebooks/)

## 🏆 **Key Features**

### **Model Architectures**
- ✅ **Spiking Neural Network (SNN)**: Bio-inspired processing with surrogate gradients
- ✅ **Recurrent Neural Network (RNN)**: GRU-based sequential temporal modeling  
- ✅ **Temporal Convolutional Network (TCN)**: Dilated convolutions for parallel processing

### **Anomaly Types**
- **Blackout Regions**: Sensor failure simulation
- **Vibration Noise**: Camera shake modeling
- **Polarity Flipping**: Hardware error representation

### **Evaluation Framework**
- **Performance Comparison Table**: All metrics side-by-side
- **ROC Curve Analysis**: Visual performance comparison
- **Training History**: Convergence pattern analysis
- **Statistical Significance**: Best model identification

## 📊 **Results Preview**

The system provides comprehensive comparison across all architectures:

| Architecture | Key Strengths | Expected Use Case |
|-------------|---------------|-------------------|
| **SNN** | Bio-inspired, energy-efficient | Edge devices, neuromorphic chips |
| **RNN** | Proven temporal modeling | Baseline performance, sequential data |
| **TCN** | Parallel processing | High-performance applications |

## 🔧 **Configuration**

Key parameters can be adjusted in the notebook:

```python
# Main configuration
sequence = "indoor_flying"    # MVSEC sequence
camera = "left"              # Camera selection  
num_epochs = 10              # Training epochs
batch_size = 16              # Batch size
anomaly_ratio = 0.5          # 50% anomalous data
```

## 🏃‍♂️ **Google Colab Setup**

For easy cloud execution:

1. Upload `anomaly_detection.ipynb` to Google Colab
2. Run the first cell which auto-installs dependencies
3. Execute all cells for complete model comparison

## 📋 **Requirements**

All dependencies managed through Poetry:
- PyTorch (neural network implementation)
- Tonic (neuromorphic data handling)  
- NumPy, Matplotlib, Pandas (data processing)
- Jupyter (notebook interface)
- Scikit-learn (evaluation metrics)

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Update documentation in `docs/` if needed
4. Submit a pull request

## 📝 **License**

This project is available for research and educational purposes.

## 📞 **Contact**

For questions, issues, or collaboration opportunities, please open an issue or contact the maintainers.

---

**Status**: ✅ Complete implementation with comprehensive documentation  
**Models**: SNN, RNN, TCN all implemented and compared  
**Data**: Real MVSEC neuromorphic data integration  
**Documentation**: 80+ pages of comprehensive guides and analysis