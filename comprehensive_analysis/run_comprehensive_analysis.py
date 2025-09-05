#!/usr/bin/env python3
"""
RQ1: Comprehensive Neuromorphic Anomaly Detection Analysis
========================================================

This script orchestrates two separate but complementary analyses:

PART A: FEATURE ENGINEERING IMPACT (rq1_feature_comparison.py)
- Question: Which feature type works best?
- Comparison: Basic vs Spatiotemporal vs Neuromorphic features
- Method: Same algorithms on different features (apples-to-apples)
- Output: feature_comparison_results.csv + visualization

PART B: ALGORITHM APPROACH COMPARISON (rq1_algorithm_comparison.py)
- Question: Which algorithmic approach works best?
- Comparison: Supervised Classification vs Unsupervised Anomaly Detection
- Method: Best features on different algorithm types
- Output: algorithm_comparison_results.csv + visualization

COMBINED INSIGHT:
- Best Feature Type + Best Algorithm Approach = Optimal Solution
- Separates "what to extract" from "how to process"
- Provides clear recommendations for deployment
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_feature_analysis():
    """Run Part A: Feature Engineering Impact Analysis"""
    print("🔬 STARTING PART A: FEATURE ENGINEERING ANALYSIS")
    print("=" * 60)

    try:
        # Import and run feature comparison
        from rq1_feature_comparison import run_feature_comparison_experiment

        start_time = time.time()
        feature_results = run_feature_comparison_experiment()
        feature_time = time.time() - start_time

        print(f"\n✅ Part A completed in {feature_time:.1f} seconds")
        return feature_results, feature_time

    except Exception as e:
        print(f"❌ Part A failed: {e}")
        return None, 0


def run_algorithm_analysis():
    """Run Part B: Algorithm Approach Comparison"""
    print("\n\n🤖 STARTING PART B: ALGORITHM APPROACH ANALYSIS")
    print("=" * 60)

    try:
        # Import and run algorithm comparison
        from rq1_algorithm_comparison import run_algorithm_comparison_experiment

        start_time = time.time()
        algorithm_results = run_algorithm_comparison_experiment()
        algorithm_time = time.time() - start_time

        print(f"\n✅ Part B completed in {algorithm_time:.1f} seconds")
        return algorithm_results, algorithm_time

    except Exception as e:
        print(f"❌ Part B failed: {e}")
        return None, 0


def create_comprehensive_summary(feature_results, algorithm_results):
    """Create comprehensive summary combining both analyses"""
    print("\n📊 CREATING COMPREHENSIVE SUMMARY")
    print("=" * 40)

    # Summary figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "RQ1: Comprehensive Neuromorphic Anomaly Detection Analysis\nFeature Engineering + Algorithm Approach",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Feature comparison summary
    if feature_results:
        feature_types = list(feature_results.keys())
        avg_f1_by_feature = {}

        for feature_type in feature_types:
            f1_scores = [
                feature_results[feature_type][clf]["f1"]
                for clf in feature_results[feature_type]
            ]
            avg_f1_by_feature[feature_type] = np.mean(f1_scores)

        colors = ["skyblue", "lightcoral", "lightgreen"]
        axes[0].bar(
            feature_types, list(avg_f1_by_feature.values()), color=colors, alpha=0.8
        )
        axes[0].set_ylabel("Average F1-Score")
        axes[0].set_title("Part A: Feature Type Performance")
        axes[0].grid(True, alpha=0.3)

        best_feature = max(avg_f1_by_feature, key=avg_f1_by_feature.get)
        best_feature_score = avg_f1_by_feature[best_feature]
    else:
        axes[0].text(
            0.5,
            0.5,
            "Part A\nFailed",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
            fontsize=14,
        )
        best_feature = "Unknown"
        best_feature_score = 0

    # 2. Algorithm comparison summary
    if algorithm_results:
        supervised_f1 = [
            algorithm_results["supervised_classification"][clf]["f1"]
            for clf in algorithm_results["supervised_classification"]
        ]
        unsupervised_f1 = [
            algorithm_results["unsupervised_anomaly_detection"][clf]["f1"]
            for clf in algorithm_results["unsupervised_anomaly_detection"]
        ]

        approaches = ["Supervised\nClassification", "Unsupervised\nAnomaly Detection"]
        avg_f1_by_approach = [np.mean(supervised_f1), np.mean(unsupervised_f1)]

        colors_approach = ["orange", "purple"]
        axes[1].bar(approaches, avg_f1_by_approach, color=colors_approach, alpha=0.8)
        axes[1].set_ylabel("Average F1-Score")
        axes[1].set_title("Part B: Algorithm Approach Performance")
        axes[1].grid(True, alpha=0.3)

        best_approach = approaches[np.argmax(avg_f1_by_approach)]
        best_approach_score = max(avg_f1_by_approach)

        # Get best individual algorithm
        all_algorithms = {}
        for clf in algorithm_results["supervised_classification"]:
            all_algorithms[f"Supervised_{clf}"] = algorithm_results[
                "supervised_classification"
            ][clf]["f1"]
        for clf in algorithm_results["unsupervised_anomaly_detection"]:
            all_algorithms[f"Unsupervised_{clf}"] = algorithm_results[
                "unsupervised_anomaly_detection"
            ][clf]["f1"]

        best_algorithm = max(all_algorithms, key=all_algorithms.get)
        best_algorithm_score = all_algorithms[best_algorithm]
    else:
        axes[1].text(
            0.5,
            0.5,
            "Part B\nFailed",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=14,
        )
        best_approach = "Unknown"
        best_approach_score = 0
        best_algorithm = "Unknown"
        best_algorithm_score = 0

    # 3. Combined recommendations
    axes[2].axis("off")

    summary_text = f"""COMPREHENSIVE RESULTS

PART A - BEST FEATURES:
{best_feature}
F1-Score: {best_feature_score:.3f}

PART B - BEST APPROACH:
{best_approach.replace(chr(10), ' ')}
F1-Score: {best_approach_score:.3f}

BEST OVERALL ALGORITHM:
{best_algorithm.replace('_', ' ')}
F1-Score: {best_algorithm_score:.3f}

DEPLOYMENT RECOMMENDATION:
• Use {best_feature} features
• Apply {best_algorithm.replace('_', ' ')} algorithm
• Expected Performance: {best_algorithm_score:.3f} F1-Score

KEY INSIGHTS:
• Feature engineering impact: {((best_feature_score - 0.5) * 100):.0f}% above baseline
• Algorithm choice impact: {((best_approach_score - 0.5) * 100):.0f}% above baseline
• Combined approach optimizes both aspects"""

    axes[2].text(
        0.05,
        0.95,
        summary_text,
        transform=axes[2].transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig("comprehensive_analysis_summary.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Create comprehensive CSV
    comprehensive_data = []

    # Add feature results
    if feature_results:
        for feature_type in feature_results:
            for classifier in feature_results[feature_type]:
                metrics = feature_results[feature_type][classifier]
                comprehensive_data.append(
                    {
                        "Analysis_Type": "Feature_Comparison",
                        "Category": feature_type,
                        "Method": classifier,
                        "F1_Score": metrics["f1"],
                        "AUC": metrics["auc"],
                        "Accuracy": metrics["accuracy"],
                    }
                )

    # Add algorithm results
    if algorithm_results:
        for approach in algorithm_results:
            for algorithm in algorithm_results[approach]:
                metrics = algorithm_results[approach][algorithm]
                comprehensive_data.append(
                    {
                        "Analysis_Type": "Algorithm_Comparison",
                        "Category": approach.replace("_", " ").title(),
                        "Method": algorithm,
                        "F1_Score": metrics["f1"],
                        "AUC": metrics["auc"],
                        "Accuracy": metrics["accuracy"],
                    }
                )

    comprehensive_df = pd.DataFrame(comprehensive_data)
    comprehensive_df.to_csv("comprehensive_analysis_results.csv", index=False)

    return {
        "best_feature": best_feature,
        "best_feature_score": best_feature_score,
        "best_approach": best_approach,
        "best_approach_score": best_approach_score,
        "best_algorithm": best_algorithm,
        "best_algorithm_score": best_algorithm_score,
    }


def main():
    """Run comprehensive analysis"""
    print("🚀 COMPREHENSIVE NEUROMORPHIC ANOMALY DETECTION ANALYSIS")
    print("=" * 70)
    print("OBJECTIVE: Optimize both feature engineering and algorithm choice")
    print("METHOD: Two separate analyses addressing different aspects")
    print("=" * 70)

    total_start_time = time.time()

    # Run Part A: Feature Analysis
    feature_results, feature_time = run_feature_analysis()

    # Run Part B: Algorithm Analysis
    algorithm_results, algorithm_time = run_algorithm_analysis()

    # Create comprehensive summary
    if feature_results or algorithm_results:
        summary = create_comprehensive_summary(feature_results, algorithm_results)

        total_time = time.time() - total_start_time

        print("\n🎉 COMPREHENSIVE ANALYSIS COMPLETED")
        print("=" * 50)
        print(f"Total Time: {total_time:.1f} seconds")
        print(f"Part A (Features): {feature_time:.1f}s")
        print(f"Part B (Algorithms): {algorithm_time:.1f}s")

        if feature_results and algorithm_results:
            print("\n🏆 FINAL RECOMMENDATIONS:")
            print(
                f"• Best Features: {summary['best_feature']} (F1={summary['best_feature_score']:.3f})"
            )
            print(
                f"• Best Approach: {summary['best_approach'].replace(chr(10), ' ')} (F1={summary['best_approach_score']:.3f})"
            )
            print(
                f"• Best Overall: {summary['best_algorithm'].replace('_', ' ')} (F1={summary['best_algorithm_score']:.3f})"
            )

            print("\n📁 OUTPUT FILES:")
            print("• feature_comparison_results.csv - Part A detailed results")
            print("• algorithm_comparison_results.csv - Part B detailed results")
            print("• comprehensive_analysis_results.csv - Combined results")
            print("• feature_comparison_results.png - Part A visualization")
            print("• algorithm_comparison_results.png - Part B visualization")
            print("• comprehensive_analysis_summary.png - Combined summary")

        else:
            print("\n⚠️  Partial results - check individual analysis outputs")

    else:
        print("\n❌ Both analyses failed - check error messages above")


if __name__ == "__main__":
    # Change to the script directory to ensure relative imports work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    main()
