"""
Apply feature extraction to your cleaned phishing dataset
This script processes your cleaned CSV and creates a feature-rich dataset
"""

import pandas as pd
import numpy as np
from feature_extraction import URLFeatureExtractor
import os

def main():
    # ==========================================
    # STEP 1: Load your cleaned dataset
    # ==========================================
    print("="*60)
    print("PHISHING URL FEATURE EXTRACTION PIPELINE")
    print("="*60)
    
    # Update this path to your cleaned CSV file
    input_file = "phishing_dataset_clean.csv"  # Change this to your actual filename
    output_file = "phishing_features.csv"
    
    print(f"\n[1/5] Loading dataset from: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"ERROR: File '{input_file}' not found!")
        print("Please update the 'input_file' variable with your cleaned CSV filename.")
        return
    
    # Load the data
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df)} URLs")
    print(f"  Columns: {list(df.columns)}")
    
    # Rename 'text' column to 'url' if needed
    if 'text' in df.columns and 'url' not in df.columns:
        df = df.rename(columns={'text': 'url'})
        print(f"✓ Renamed 'text' column to 'url'")
    
    # Choose your dataset size
    # Options:
    #   None = Full dataset (782k URLs, ~90 mins)
    #   100000 = 100k URLs (~20 mins, excellent accuracy)
    #   50000 = 50k URLs (~12 mins, very good accuracy)
    #   10000 = 10k URLs (~3 mins, good for testing)
    
    SAMPLE_SIZE = 100000  # Perfect for MVP! Change this if needed
    
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"\n⚠️  SAMPLE MODE: Processing {SAMPLE_SIZE:,} URLs")
        print(f"   Full dataset has {len(df):,} URLs")
        df = df.head(SAMPLE_SIZE)
    else:
        print(f"\n✅ FULL DATASET MODE: Processing all {len(df):,} URLs")
        print(f"   Estimated time: ~{len(df) // 10000 * 0.5:.0f}-{len(df) // 10000 * 0.7:.0f} minutes")
        print(f"   This is a one-time process - grab a coffee! ☕")
    
    # Check if 'url' and 'label' columns exist
    if 'url' not in df.columns:
        print("\nERROR: 'url' column not found in dataset!")
        print(f"Available columns: {list(df.columns)}")
        print("Please ensure your CSV has a column named 'url'")
        return
    
    if 'label' not in df.columns:
        print("\nWARNING: 'label' column not found!")
        print("If your dataset has labels with a different name, please rename it to 'label'")
    
    # ==========================================
    # STEP 2: Display dataset statistics
    # ==========================================
    print(f"\n[2/5] Dataset Statistics:")
    print(f"  Total URLs: {len(df)}")
    
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        print(f"  Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            label_name = "Phishing" if label == 1 else "Legitimate"
            print(f"    {label_name} ({label}): {count} ({percentage:.1f}%)")
        
        # Check if dataset is balanced
        balance_ratio = min(label_counts) / max(label_counts)
        if balance_ratio < 0.8:
            print(f"\n  ⚠️  Dataset is imbalanced (ratio: {balance_ratio:.2f})")
            print(f"      Consider balancing or using stratified sampling")
        else:
            print(f"\n  ✓ Dataset is well-balanced (ratio: {balance_ratio:.2f})")
    
    # ==========================================
    # STEP 3: Extract features
    # ==========================================
    print(f"\n[3/5] Extracting features from URLs...")
    print("  This may take a few minutes depending on dataset size...")
    
    extractor = URLFeatureExtractor()
    features_df = extractor.extract_features_from_dataframe(df, url_column='url')
    
    print(f"\n✓ Feature extraction complete!")
    print(f"  Total features extracted: {len(features_df.columns) - 2}")  # Exclude 'url' and 'label'
    
    # ==========================================
    # STEP 4: Display feature summary
    # ==========================================
    print(f"\n[4/5] Feature Summary:")
    
    # Get numeric columns only (exclude url and label)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'label']
    
    print(f"\n  Sample features (first 5 rows):")
    display_cols = ['url', 'url_length', 'is_https', 'has_ip', 
                    'suspicious_keyword_count', 'entropy', 'label']
    display_cols = [col for col in display_cols if col in features_df.columns]
    print(features_df[display_cols].head())
    
    # Show feature statistics
    print(f"\n  Feature statistics:")
    stats_cols = ['url_length', 'dot_count', 'suspicious_keyword_count', 
                  'subdomain_count', 'entropy']
    stats_cols = [col for col in stats_cols if col in features_df.columns]
    
    if stats_cols:
        print(features_df[stats_cols].describe())
    
    # Check for missing values
    missing_values = features_df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\n  ⚠️  Missing values detected:")
        missing_cols = missing_values[missing_cols > 0]
        for col, count in missing_cols.items():
            print(f"    {col}: {count} missing values")
        
        print(f"\n  Filling missing values with 0...")
        features_df = features_df.fillna(0)
    else:
        print(f"\n  ✓ No missing values found")
    
    # ==========================================
    # STEP 5: Save the feature dataset
    # ==========================================
    print(f"\n[5/5] Saving feature dataset...")
    
    features_df.to_csv(output_file, index=False)
    print(f"✓ Saved to: {output_file}")
    print(f"  Shape: {features_df.shape}")
    print(f"  Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    
    # ==========================================
    # Final Summary
    # ==========================================
    print("\n" + "="*60)
    print("FEATURE EXTRACTION COMPLETE!")
    print("="*60)
    print(f"\n✓ Your feature-rich dataset is ready: {output_file}")
    print(f"✓ Total features: {len(features_df.columns) - 2}")
    print(f"✓ Total samples: {len(features_df)}")
    print(f"\nNext Steps:")
    print(f"  1. Review the features in {output_file}")
    print(f"  2. Proceed to Phase 3: Train your ML model")
    print(f"  3. Run: python train_model.py")
    print("="*60)


if __name__ == "__main__":
    main()