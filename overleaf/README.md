# LaTeX Documents for Overleaf

This directory contains professional LaTeX documents ready for compilation in Overleaf or local LaTeX environments.

## 📄 **Professional LaTeX Documents**

### **1. Thesis Project Plan** 
**File:** `2025_MA_Moeez_Thesis_Project_Plan.tex`
- **Purpose:** Official academic thesis project plan
- **Content:** Research questions, methodology, timeline, MVSEC dataset specifications
- **Status:** Updated to reflect current dual-approach implementation
- **Length:** ~15 pages with comprehensive research planning

### **2. Technical Code Guide**
**File:** `MVSEC_Anomaly_Detection_Code_Guide.tex`
- **Purpose:** Complete technical documentation for software engineers and researchers
- **Content:** Architecture explanations, code walkthroughs, both approaches covered
- **Status:** Comprehensive guide with detailed implementation analysis
- **Length:** ~80 pages with extensive code examples and technical details

### **3. Notebook Comparison Analysis**
**File:** `../notebooks/Notebook_Comparison_Analysis.tex` (Reference)
- **Purpose:** Detailed comparison of architecture vs feature-based approaches
- **Content:** Scope, output, and technical differences analysis  
- **Status:** Comprehensive analysis document
- **Length:** ~30 pages of systematic approach comparison

## 🚀 **How to Use**

### **Option 1: Overleaf (Recommended)**
1. Upload `.tex` files to [Overleaf](https://www.overleaf.com)
2. Compile automatically with built-in LaTeX environment
3. Share and collaborate online

### **Option 2: Local LaTeX**
```bash
# Compile thesis project plan
pdflatex 2025_MA_Moeez_Thesis_Project_Plan.tex

# Compile technical guide  
pdflatex MVSEC_Anomaly_Detection_Code_Guide.tex
```

### **Option 3: VS Code with LaTeX Workshop**
1. Install LaTeX Workshop extension
2. Open `.tex` files
3. Use Ctrl+Alt+B to build

## 📋 **Document Features**

### **Professional Formatting:**
- ✅ Academic paper layout with proper sections
- ✅ Code syntax highlighting for Python snippets
- ✅ Professional diagrams and flowcharts
- ✅ Comprehensive table of contents
- ✅ Cross-references and hyperlinks

### **Technical Content:**
- ✅ Complete system architecture explanations
- ✅ Step-by-step implementation details
- ✅ Performance analysis and results interpretation
- ✅ Troubleshooting guides and best practices

## 🔧 **Compilation Requirements**

Both documents require these LaTeX packages:
- `times` (font)
- `amsmath`, `amsfonts`, `amssymb` (mathematics)  
- `tikz`, `pgfplots` (diagrams)
- `listings` (code highlighting)
- `hyperref` (links and references)
- `booktabs` (professional tables)

All packages are standard and available in full LaTeX distributions like TeX Live or MiKTeX.

## 📊 **Output PDFs**

After compilation, you'll get:
- `2025_MA_Moeez_Thesis_Project_Plan.pdf` - Official project plan
- `MVSEC_Anomaly_Detection_Code_Guide.pdf` - Technical implementation guide

Both PDFs are publication-ready with professional formatting suitable for academic submission or technical documentation.