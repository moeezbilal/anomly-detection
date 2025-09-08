#!/usr/bin/env python3
"""
Test script for enhanced features to validate implementation
"""

import numpy as np
import torch
from mvp_feature_comparison import EnhancedFeatureExtractor, EnhancedTrainer


def test_enhanced_features():
    """Test the enhanced feature extraction"""
    print("🧪 Testing Enhanced Feature Extraction")

    # Create test extractor
    extractor = EnhancedFeatureExtractor()

    print("Feature groups:")
    print(f"  Basic ({len(extractor.basic_names)}): {extractor.basic_names}")
    print(
        f"  Spatiotemporal ({len(extractor.spatiotemporal_names)}): {extractor.spatiotemporal_names}"
    )
    print(
        f"  Neuromorphic ({len(extractor.neuromorphic_names)}): {extractor.neuromorphic_names}"
    )
    print(f"  Total features: {len(extractor.all_names)}")

    # Test with dummy frames
    frame1 = torch.rand(2, 32, 32) * 0.1
    frame2 = torch.rand(2, 32, 32) * 0.1

    # Test basic features
    basic_features = extractor.extract_basic_features(frame1)
    print(f"\n✅ Basic features extracted: {len(basic_features)} features")

    # Test spatiotemporal features
    spatio_features = extractor.extract_spatiotemporal_features(frame1, frame2)
    print(f"✅ Spatiotemporal features extracted: {len(spatio_features)} features")

    # Test neuromorphic features
    neuro_features = extractor.extract_neuromorphic_features(frame1)
    print(f"✅ Neuromorphic features extracted: {len(neuro_features)} features")

    # Test all features
    all_features = extractor.extract_all_features(frame1, frame2)
    print(f"✅ All features extracted: {len(all_features)} features")

    # Validate no NaN or infinite values
    if np.any(np.isnan(all_features)) or np.any(np.isinf(all_features)):
        print("❌ Found NaN or infinite values in features")
        return False

    print("✅ All features are valid (no NaN or infinite values)")
    return True


def test_trainer():
    """Test the enhanced trainer"""
    print("\n🧪 Testing Enhanced Trainer")

    # Create dummy data
    np.random.seed(42)
    X = np.random.rand(100, 13)  # 100 samples, 13 features
    y = np.random.randint(0, 2, 100)  # Binary labels

    trainer = EnhancedTrainer(n_estimators=50, cv_folds=3)  # Reduced for testing

    # Test training
    model, cv_score = trainer.train_with_cv(X, y, "test_model")
    print(f"✅ Model trained with CV score: {cv_score:.4f}")

    # Test evaluation
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model.fit(X_train, y_train)

    metrics = trainer.evaluate_model(model, X_test, y_test)
    print(f"✅ Model evaluated - Accuracy: {metrics['accuracy']:.4f}")

    return True


if __name__ == "__main__":
    print("🚀 Testing Enhanced MVP Feature Comparison Components")

    try:
        # Test feature extraction
        if test_enhanced_features():
            print("\n✅ Feature extraction tests passed!")
        else:
            print("\n❌ Feature extraction tests failed!")
            exit(1)

        # Test trainer
        if test_trainer():
            print("✅ Trainer tests passed!")
        else:
            print("❌ Trainer tests failed!")
            exit(1)

        print("\n🎉 All tests passed! Enhanced implementation is working.")

    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
