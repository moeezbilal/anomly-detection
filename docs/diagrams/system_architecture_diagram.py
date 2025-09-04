#!/usr/bin/env python3
"""
System Architecture Diagram Generator for MVSEC Anomaly Detection
Creates visual representations of the system components and data flow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_system_overview_diagram():
    """Create high-level system architecture diagram"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    colors = {
        'data': '#E3F2FD',      # Light blue
        'process': '#FFF3E0',    # Light orange  
        'model': '#E8F5E8',     # Light green
        'output': '#FCE4EC'     # Light pink
    }
    
    # Title
    ax.text(8, 9.5, 'MVSEC Anomaly Detection System Architecture', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Data Layer
    data_box = FancyBboxPatch((0.5, 7.5), 3, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['data'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(data_box)
    ax.text(2, 8.25, 'MVSEC Dataset\n(HDF5 Files)', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # Preprocessing Layer
    preprocess_box = FancyBboxPatch((4.5, 7.5), 3, 1.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['process'],
                                   edgecolor='black', linewidth=2)
    ax.add_patch(preprocess_box)
    ax.text(6, 8.25, 'Event Processing\n& Temporal Binning', ha='center', va='center',
            fontsize=12, fontweight='bold')
    
    # Anomaly Generation
    anomaly_box = FancyBboxPatch((8.5, 7.5), 3, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['process'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(anomaly_box)
    ax.text(10, 8.25, 'Anomaly Injection\n(Supervised Learning)', ha='center', va='center',
            fontsize=12, fontweight='bold')
    
    # Dataset Creation
    dataset_box = FancyBboxPatch((12.5, 7.5), 3, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['process'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(dataset_box)
    ax.text(14, 8.25, 'Labeled Dataset\nCreation', ha='center', va='center',
            fontsize=12, fontweight='bold')
    
    # Model Architectures
    models = ['SNN', 'RNN', 'TCN']
    model_positions = [2, 6, 10]
    
    for i, (model, pos) in enumerate(zip(models, model_positions)):
        model_box = FancyBboxPatch((pos-0.75, 5), 1.5, 1.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['model'],
                                  edgecolor='black', linewidth=2)
        ax.add_patch(model_box)
        ax.text(pos, 5.75, f'{model}\nModel', ha='center', va='center',
                fontsize=11, fontweight='bold')
    
    # Training Process
    training_box = FancyBboxPatch((12.5, 5), 3, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['process'],
                                 edgecolor='black', linewidth=2)
    ax.add_patch(training_box)
    ax.text(14, 5.75, 'Training &\nOptimization', ha='center', va='center',
            fontsize=12, fontweight='bold')
    
    # Evaluation
    eval_box = FancyBboxPatch((2, 2.5), 4, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['output'],
                             edgecolor='black', linewidth=2)
    ax.add_patch(eval_box)
    ax.text(4, 3.25, 'Performance Evaluation\n(Accuracy, F1, ROC-AUC)', ha='center', va='center',
            fontsize=12, fontweight='bold')
    
    # Model Comparison
    comparison_box = FancyBboxPatch((7, 2.5), 4, 1.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['output'],
                                   edgecolor='black', linewidth=2)
    ax.add_patch(comparison_box)
    ax.text(9, 3.25, 'Model Comparison\n& Analysis', ha='center', va='center',
            fontsize=12, fontweight='bold')
    
    # Visualization
    viz_box = FancyBboxPatch((12, 2.5), 3.5, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['output'],
                            edgecolor='black', linewidth=2)
    ax.add_patch(viz_box)
    ax.text(13.75, 3.25, 'Results Visualization\n& Reporting', ha='center', va='center',
            fontsize=12, fontweight='bold')
    
    # Add arrows showing data flow
    arrows = [
        # Horizontal flow at top
        ((3.5, 8.25), (4.5, 8.25)),
        ((7.5, 8.25), (8.5, 8.25)),
        ((11.5, 8.25), (12.5, 8.25)),
        
        # From dataset to models
        ((14, 7.5), (14, 6.5)),
        ((14, 6.5), (2, 6.5)),
        ((2, 6.5), (2, 6.5)),
        ((6, 6.5), (6, 6.5)),
        ((10, 6.5), (10, 6.5)),
        
        # From models to training
        ((10.75, 5.75), (12.5, 5.75)),
        
        # From training to evaluation
        ((14, 5), (14, 4.5)),
        ((14, 4.5), (4, 4.5)),
        ((4, 4.5), (4, 4)),
        ((9, 4.5), (9, 4)),
        ((13.75, 4.5), (13.75, 4)),
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=0, shrinkB=0,
                               mutation_scale=20, fc="black")
        ax.add_patch(arrow)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors['data'], label='Data Layer'),
        mpatches.Patch(color=colors['process'], label='Processing Layer'),
        mpatches.Patch(color=colors['model'], label='Model Layer'),
        mpatches.Patch(color=colors['output'], label='Output Layer')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    plt.tight_layout()
    plt.savefig('/Users/eahmmoe/localWorkSpace/kth/anomly-detection/system_architecture.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_data_flow_diagram():
    """Create detailed data flow diagram"""
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(9, 11.5, 'MVSEC Data Processing Pipeline', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Define stages
    stages = [
        {
            'name': 'Raw MVSEC Data',
            'pos': (2, 9.5),
            'size': (3, 1.5),
            'details': ['HDF5 Files', 'davis/left/events', '[x, y, t, p] format'],
            'color': '#E3F2FD'
        },
        {
            'name': 'Event Extraction',
            'pos': (6.5, 9.5),
            'size': (3, 1.5),
            'details': ['Load ~500K events', 'Coordinate bounds', 'Polarity mapping'],
            'color': '#FFF3E0'
        },
        {
            'name': 'Temporal Binning',
            'pos': (11, 9.5),
            'size': (3, 1.5),
            'details': ['50 time bins', '2-channel frames', 'Normalization'],
            'color': '#FFF3E0'
        },
        {
            'name': 'Frame Tensor',
            'pos': (15, 9.5),
            'size': (2.5, 1.5),
            'details': ['(50, 2, 64, 64)', 'Pos/Neg channels', '[0, 1] range'],
            'color': '#E8F5E8'
        },
        {
            'name': 'Blackout Anomaly',
            'pos': (1.5, 6.5),
            'size': (2.5, 1.5),
            'details': ['Sensor failure', '70-100% reduction', 'Random regions'],
            'color': '#FFEBEE'
        },
        {
            'name': 'Vibration Noise',
            'pos': (5, 6.5),
            'size': (2.5, 1.5),
            'details': ['Camera shake', 'Gaussian noise', '0.3-0.7 intensity'],
            'color': '#FFEBEE'
        },
        {
            'name': 'Polarity Flip',
            'pos': (8.5, 6.5),
            'size': (2.5, 1.5),
            'details': ['Hardware error', 'Channel swapping', '60-90% probability'],
            'color': '#FFEBEE'
        },
        {
            'name': 'Labeled Dataset',
            'pos': (13, 6.5),
            'size': (3, 1.5),
            'details': ['50% normal', '50% anomalous', 'Binary labels'],
            'color': '#E8F5E8'
        },
        {
            'name': 'Train/Val/Test Split',
            'pos': (4, 3.5),
            'size': (3.5, 1.5),
            'details': ['70% training', '15% validation', '15% testing'],
            'color': '#FFF3E0'
        },
        {
            'name': 'Data Loaders',
            'pos': (9, 3.5),
            'size': (3, 1.5),
            'details': ['Batch size: 8', 'Custom collate', 'Shuffled training'],
            'color': '#FFF3E0'
        },
        {
            'name': 'Model Training',
            'pos': (13.5, 3.5),
            'size': (3, 1.5),
            'details': ['Cross-entropy loss', 'Adam optimizer', 'LR scheduling'],
            'color': '#E8F5E8'
        }
    ]
    
    # Draw all stages
    for stage in stages:
        # Main box
        box = FancyBboxPatch(stage['pos'], stage['size'][0], stage['size'][1],
                            boxstyle="round,pad=0.1",
                            facecolor=stage['color'],
                            edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        
        # Title
        title_y = stage['pos'][1] + stage['size'][1] - 0.3
        ax.text(stage['pos'][0] + stage['size'][0]/2, title_y, stage['name'],
                ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Details
        for i, detail in enumerate(stage['details']):
            detail_y = title_y - 0.4 - (i * 0.25)
            ax.text(stage['pos'][0] + stage['size'][0]/2, detail_y, detail,
                    ha='center', va='center', fontsize=9)
    
    # Add data flow arrows
    flow_arrows = [
        # Main horizontal flow
        ((5, 10.25), (6.5, 10.25)),
        ((9.5, 10.25), (11, 10.25)),
        ((14, 10.25), (15, 10.25)),
        
        # From frames to anomaly generation
        ((15.5, 9.5), (15.5, 8.5)),
        ((15.5, 8.5), (2.75, 8.5)),
        ((2.75, 8.5), (2.75, 8)),
        ((2.75, 8), (6.25, 8)),
        ((6.25, 8), (6.25, 8)),
        ((6.25, 8), (9.75, 8)),
        ((9.75, 8), (9.75, 8)),
        
        # From anomalies to dataset
        ((11, 7.25), (13, 7.25)),
        
        # From dataset to split
        ((14.5, 6.5), (14.5, 5.5)),
        ((14.5, 5.5), (5.75, 5.5)),
        ((5.75, 5.5), (5.75, 5)),
        
        # Split to loaders to training
        ((7.5, 4.25), (9, 4.25)),
        ((12, 4.25), (13.5, 4.25)),
    ]
    
    for start, end in flow_arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=0, shrinkB=0,
                               mutation_scale=15, fc="black")
        ax.add_patch(arrow)
    
    # Add process annotations
    ax.text(9, 1.5, 'Key Processing Steps:', fontsize=14, fontweight='bold')
    ax.text(9, 1, '1. Load event data from MVSEC HDF5 files', fontsize=11)
    ax.text(9, 0.7, '2. Convert events to temporal frame sequences', fontsize=11)
    ax.text(9, 0.4, '3. Inject artificial anomalies for supervised learning', fontsize=11)
    ax.text(9, 0.1, '4. Train neural networks to distinguish normal vs anomalous patterns', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('/Users/eahmmoe/localWorkSpace/kth/anomly-detection/data_flow_diagram.png',
                dpi=300, bbox_inches='tight')
    plt.show()

def create_model_architecture_diagram():
    """Create detailed model architecture comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 12))
    fig.suptitle('Neural Network Architecture Comparison', fontsize=20, fontweight='bold')
    
    # SNN Architecture
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('Spiking Neural Network (SNN)', fontsize=16, fontweight='bold', pad=20)
    
    # SNN layers
    snn_layers = [
        {'name': 'Input\n(2, 64, 64)', 'pos': (2, 10.5), 'size': (6, 1), 'color': '#E3F2FD'},
        {'name': 'SpikingConv2d\n2→16, k=3, s=2', 'pos': (2, 9), 'size': (6, 1), 'color': '#E8F5E8'},
        {'name': 'Spiking Neuron\nβ=0.9, θ=1.0', 'pos': (2, 7.5), 'size': (6, 1), 'color': '#FFF3E0'},
        {'name': 'SpikingConv2d\n16→32, k=3, s=2', 'pos': (2, 6), 'size': (6, 1), 'color': '#E8F5E8'},
        {'name': 'SpikingConv2d\n32→64, k=3, s=2', 'pos': (2, 4.5), 'size': (6, 1), 'color': '#E8F5E8'},
        {'name': 'Global Avg Pool\n→ (64,)', 'pos': (2, 3), 'size': (6, 1), 'color': '#FCE4EC'},
        {'name': 'Linear\n64→2', 'pos': (2, 1.5), 'size': (6, 1), 'color': '#FCE4EC'},
    ]
    
    for layer in snn_layers:
        box = FancyBboxPatch(layer['pos'], layer['size'][0], layer['size'][1],
                            boxstyle="round,pad=0.1", facecolor=layer['color'],
                            edgecolor='black', linewidth=1)
        ax1.add_patch(box)
        ax1.text(layer['pos'][0] + layer['size'][0]/2, layer['pos'][1] + layer['size'][1]/2,
                layer['name'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add SNN-specific annotations
    ax1.text(9, 7.5, 'Membrane\nPotential\nDynamics', ha='center', va='center', 
             fontsize=9, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))
    
    # SNN arrows
    for i in range(len(snn_layers)-1):
        y_start = snn_layers[i]['pos'][1]
        y_end = snn_layers[i+1]['pos'][1] + snn_layers[i+1]['size'][1]
        arrow = ConnectionPatch((5, y_start), (5, y_end), "data", "data",
                               arrowstyle="->", shrinkA=0, shrinkB=0, mutation_scale=15)
        ax1.add_patch(arrow)
    
    # RNN Architecture
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_title('Recurrent Neural Network (RNN)', fontsize=16, fontweight='bold', pad=20)
    
    rnn_layers = [
        {'name': 'Input\n(2, 64, 64)', 'pos': (2, 10.5), 'size': (6, 1), 'color': '#E3F2FD'},
        {'name': 'Conv2d\n2→16, k=3, s=2', 'pos': (2, 9), 'size': (6, 1), 'color': '#E8F5E8'},
        {'name': 'ReLU + Conv2d\n16→32, k=3, s=2', 'pos': (2, 7.5), 'size': (6, 1), 'color': '#E8F5E8'},
        {'name': 'Flatten\n→ (32×16×16,)', 'pos': (2, 6), 'size': (6, 1), 'color': '#FFF3E0'},
        {'name': 'Reshape for RNN\n→ (1, features)', 'pos': (2, 4.5), 'size': (6, 1), 'color': '#FFF3E0'},
        {'name': 'GRU Layer\nfeatures→64', 'pos': (2, 3), 'size': (6, 1), 'color': '#FFEBEE'},
        {'name': 'Linear\n64→2', 'pos': (2, 1.5), 'size': (6, 1), 'color': '#FCE4EC'},
    ]
    
    for layer in rnn_layers:
        box = FancyBboxPatch(layer['pos'], layer['size'][0], layer['size'][1],
                            boxstyle="round,pad=0.1", facecolor=layer['color'],
                            edgecolor='black', linewidth=1)
        ax2.add_patch(box)
        ax2.text(layer['pos'][0] + layer['size'][0]/2, layer['pos'][1] + layer['size'][1]/2,
                layer['name'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # RNN arrows
    for i in range(len(rnn_layers)-1):
        y_start = rnn_layers[i]['pos'][1]
        y_end = rnn_layers[i+1]['pos'][1] + rnn_layers[i+1]['size'][1]
        arrow = ConnectionPatch((5, y_start), (5, y_end), "data", "data",
                               arrowstyle="->", shrinkA=0, shrinkB=0, mutation_scale=15)
        ax2.add_patch(arrow)
    
    # TCN Architecture
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 12)
    ax3.axis('off')
    ax3.set_title('Temporal Convolutional Network (TCN)', fontsize=16, fontweight='bold', pad=20)
    
    tcn_layers = [
        {'name': 'Input\n(2, 64, 64)', 'pos': (2, 10.5), 'size': (6, 1), 'color': '#E3F2FD'},
        {'name': 'TemporalBlock\n2→16, dilation=1', 'pos': (2, 9), 'size': (6, 1), 'color': '#E8F5E8'},
        {'name': 'TemporalBlock\n16→32, dilation=2', 'pos': (2, 7.5), 'size': (6, 1), 'color': '#E8F5E8'},
        {'name': 'TemporalBlock\n32→64, dilation=4', 'pos': (2, 6), 'size': (6, 1), 'color': '#E8F5E8'},
        {'name': 'Global Avg Pool\n→ (64,)', 'pos': (2, 4.5), 'size': (6, 1), 'color': '#FCE4EC'},
        {'name': 'Linear\n64→2', 'pos': (2, 3), 'size': (6, 1), 'color': '#FCE4EC'},
    ]
    
    for layer in tcn_layers:
        box = FancyBboxPatch(layer['pos'], layer['size'][0], layer['size'][1],
                            boxstyle="round,pad=0.1", facecolor=layer['color'],
                            edgecolor='black', linewidth=1)
        ax3.add_patch(box)
        ax3.text(layer['pos'][0] + layer['size'][0]/2, layer['pos'][1] + layer['size'][1]/2,
                layer['name'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add TCN-specific annotation
    ax3.text(9, 7.5, 'Dilated\nConvolutions\nfor Long-range\nDependencies', 
             ha='center', va='center', fontsize=9, style='italic',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.3))
    
    # TCN arrows
    for i in range(len(tcn_layers)-1):
        y_start = tcn_layers[i]['pos'][1]
        y_end = tcn_layers[i+1]['pos'][1] + tcn_layers[i+1]['size'][1]
        arrow = ConnectionPatch((5, y_start), (5, y_end), "data", "data",
                               arrowstyle="->", shrinkA=0, shrinkB=0, mutation_scale=15)
        ax3.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig('/Users/eahmmoe/localWorkSpace/kth/anomly-detection/model_architectures.png',
                dpi=300, bbox_inches='tight')
    plt.show()

def create_anomaly_strategy_diagram():
    """Create anomaly generation strategy visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Anomaly Generation Strategy', fontsize=18, fontweight='bold')
    
    # Original frame (top-left)
    ax1 = axes[0, 0]
    ax1.set_title('Original Event Frame', fontsize=14, fontweight='bold')
    
    # Create synthetic event frame
    np.random.seed(42)
    original_frame = np.random.exponential(scale=0.1, size=(64, 64))
    original_frame = np.clip(original_frame, 0, 1)
    
    im1 = ax1.imshow(original_frame, cmap='viridis', aspect='equal')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Blackout anomaly (top-right)
    ax2 = axes[0, 1]
    ax2.set_title('Blackout Anomaly', fontsize=14, fontweight='bold')
    
    blackout_frame = original_frame.copy()
    # Add blackout region
    blackout_frame[20:40, 15:35] *= 0.1  # 90% reduction
    
    im2 = ax2.imshow(blackout_frame, cmap='viridis', aspect='equal')
    ax2.add_patch(plt.Rectangle((15, 20), 20, 20, fill=False, edgecolor='red', linewidth=2))
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Vibration noise (bottom-left)
    ax3 = axes[1, 0]
    ax3.set_title('Vibration Noise Anomaly', fontsize=14, fontweight='bold')
    
    vibration_frame = original_frame.copy()
    # Add noise to a region
    noise_region = np.random.normal(0, 0.3, (25, 25))
    vibration_frame[10:35, 25:50] += noise_region
    vibration_frame = np.clip(vibration_frame, 0, 1)
    
    im3 = ax3.imshow(vibration_frame, cmap='viridis', aspect='equal')
    ax3.add_patch(plt.Rectangle((25, 10), 25, 25, fill=False, edgecolor='orange', linewidth=2))
    ax3.set_xlabel('X coordinate')
    ax3.set_ylabel('Y coordinate')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Polarity flip (bottom-right)
    ax4 = axes[1, 1]
    ax4.set_title('Polarity Flip Visualization', fontsize=14, fontweight='bold')
    
    # Create a visualization showing channel swapping concept
    pos_channel = np.random.exponential(scale=0.15, size=(32, 32))
    neg_channel = np.random.exponential(scale=0.1, size=(32, 32))
    
    # Combine channels for visualization
    combined = np.zeros((32, 32, 3))
    combined[:, :, 0] = pos_channel  # Red for positive
    combined[:, :, 2] = neg_channel  # Blue for negative
    
    # Add flip region
    flip_mask = np.zeros((32, 32), dtype=bool)
    flip_mask[8:24, 8:24] = True
    
    # Swap channels in flip region
    temp = combined[flip_mask, 0].copy()
    combined[flip_mask, 0] = combined[flip_mask, 2]
    combined[flip_mask, 2] = temp
    
    ax4.imshow(combined, aspect='equal')
    ax4.add_patch(plt.Rectangle((8, 8), 16, 16, fill=False, edgecolor='white', linewidth=2))
    ax4.set_xlabel('X coordinate')
    ax4.set_ylabel('Y coordinate')
    ax4.text(16, 30, 'Red: Positive Events\nBlue: Negative Events\nWhite Box: Flipped Region', 
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/Users/eahmmoe/localWorkSpace/kth/anomly-detection/anomaly_strategy.png',
                dpi=300, bbox_inches='tight')
    plt.show()

def create_evaluation_framework_diagram():
    """Create evaluation framework visualization"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(7, 9.5, 'Evaluation Framework & Metrics', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Training Process
    train_box = FancyBboxPatch((1, 7), 3.5, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor='#E8F5E8',
                              edgecolor='black', linewidth=2)
    ax.add_patch(train_box)
    ax.text(2.75, 7.75, 'Training Process\n• Cross-entropy loss\n• Adam optimizer\n• LR scheduling', 
            ha='center', va='center', fontsize=10)
    
    # Validation
    val_box = FancyBboxPatch((5.5, 7), 3, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor='#FFF3E0',
                            edgecolor='black', linewidth=2)
    ax.add_patch(val_box)
    ax.text(7, 7.75, 'Validation\n• Overfitting monitoring\n• Early stopping\n• Hyperparameter tuning', 
            ha='center', va='center', fontsize=10)
    
    # Testing
    test_box = FancyBboxPatch((9.5, 7), 3.5, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor='#FCE4EC',
                             edgecolor='black', linewidth=2)
    ax.add_patch(test_box)
    ax.text(11.25, 7.75, 'Testing\n• Unseen data evaluation\n• Final performance metrics\n• Statistical significance', 
            ha='center', va='center', fontsize=10)
    
    # Metrics boxes
    metrics = [
        {'name': 'Accuracy\nTP+TN/Total', 'pos': (0.5, 4.5), 'color': '#E3F2FD'},
        {'name': 'Precision\nTP/(TP+FP)', 'pos': (3, 4.5), 'color': '#E3F2FD'},
        {'name': 'Recall\nTP/(TP+FN)', 'pos': (5.5, 4.5), 'color': '#E3F2FD'},
        {'name': 'F1-Score\n2×P×R/(P+R)', 'pos': (8, 4.5), 'color': '#E3F2FD'},
        {'name': 'ROC-AUC\nArea Under Curve', 'pos': (10.5, 4.5), 'color': '#E3F2FD'},
    ]
    
    for metric in metrics:
        box = FancyBboxPatch(metric['pos'], 2, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor=metric['color'],
                            edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(metric['pos'][0] + 1, metric['pos'][1] + 0.75, metric['name'],
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Visualization outputs
    viz_outputs = [
        {'name': 'Training Curves', 'pos': (1, 2), 'color': '#FFEBEE'},
        {'name': 'Confusion Matrix', 'pos': (4, 2), 'color': '#FFEBEE'},
        {'name': 'ROC Curves', 'pos': (7, 2), 'color': '#FFEBEE'},
        {'name': 'Model Comparison', 'pos': (10, 2), 'color': '#FFEBEE'},
    ]
    
    for viz in viz_outputs:
        box = FancyBboxPatch(viz['pos'], 2.5, 1,
                            boxstyle="round,pad=0.1",
                            facecolor=viz['color'],
                            edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(viz['pos'][0] + 1.25, viz['pos'][1] + 0.5, viz['name'],
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add arrows
    arrows = [
        ((2.75, 7), (2.75, 6)),
        ((7, 7), (7, 6)),
        ((11.25, 7), (11.25, 6)),
        
        # From metrics to visualizations
        ((1.5, 4.5), (2.25, 3)),
        ((4, 4.5), (5.25, 3)),
        ((6.5, 4.5), (8.25, 3)),
        ((9, 4.5), (11.25, 3)),
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=0, shrinkB=0,
                               mutation_scale=15, fc="black")
        ax.add_patch(arrow)
    
    # Add evaluation strategy text
    ax.text(7, 0.5, 'Evaluation Strategy: Cross-validation, Statistical Testing, Comparative Analysis',
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig('/Users/eahmmoe/localWorkSpace/kth/anomly-detection/evaluation_framework.png',
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Generating system architecture diagrams...")
    
    # Generate all diagrams
    create_system_overview_diagram()
    print("✅ System overview diagram created")
    
    create_data_flow_diagram()
    print("✅ Data flow diagram created")
    
    create_model_architecture_diagram()
    print("✅ Model architecture diagram created")
    
    create_anomaly_strategy_diagram()
    print("✅ Anomaly strategy diagram created")
    
    create_evaluation_framework_diagram()
    print("✅ Evaluation framework diagram created")
    
    print("\n🎉 All diagrams generated successfully!")
    print("Files saved to:")
    print("- system_architecture.png")
    print("- data_flow_diagram.png")
    print("- model_architectures.png")
    print("- anomaly_strategy.png")
    print("- evaluation_framework.png")