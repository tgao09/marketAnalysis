import pandas as pd
import numpy as np
import os

print("NEW MODEL PERFORMANCE ANALYSIS")
print("="*60)

# determine file path
script_dir = os.path.dirname(os.path.abspath(__file__))
results_file = os.path.join(script_dir, 'arimax/results', 'model_summary.csv')

# load new results
new_results = pd.read_csv(results_file)
successful = new_results[new_results['status'] == 'success']

print(f"\nModels trained: {len(successful)}/{len(new_results)}")
print(f"Success rate: {len(successful)/len(new_results)*100:.1f}%")

print("\n\nFeature Counts:")
if 'total_features' in successful.columns and len(successful) > 0:
    print(f"Total features per model: {successful['total_features'].iloc[0]}")
    print(f"Previous (leaky): 16 features")
    print(f"After data fix: 12 features")
    print(f"After improvements: {successful['total_features'].iloc[0]} features")
else:
    print("Feature count information not available in model summary")

print("\n\nDirectional Accuracy:")
if 'directional_accuracy' in successful.columns and len(successful) > 0:
    print(f"Mean: {successful['directional_accuracy'].mean():.2f}%")
    print(f"Median: {successful['directional_accuracy'].median():.2f}%")
    print(f"Std: {successful['directional_accuracy'].std():.2f}%")
    print(f"Min: {successful['directional_accuracy'].min():.2f}%")
    print(f"Max: {successful['directional_accuracy'].max():.2f}%")

    print("\n\nTop 10 performers:")
    top10 = successful.nlargest(10, 'directional_accuracy')[['ticker', 'directional_accuracy', 'test_rmse']]
    print(top10.to_string(index=False))

    print("\n\nBottom 10 performers:")
    bottom10 = successful.nsmallest(10, 'directional_accuracy')[['ticker', 'directional_accuracy', 'test_rmse']]
    print(bottom10.to_string(index=False))
else:
    print("Directional accuracy information not available")

print("\n\nRMSE Analysis:")
if 'test_rmse' in successful.columns and len(successful) > 0:
    print(f"Mean RMSE: {successful['test_rmse'].mean():.4f}")
    print(f"Median RMSE: {successful['test_rmse'].median():.4f}")
    print(f"Previous baseline: ~0.047")

    # calculate improvement in rmse if applicable
    baseline_rmse = 0.047
    new_rmse = successful['test_rmse'].mean()
    rmse_improvement = baseline_rmse - new_rmse
    if rmse_improvement > 0:
        print(f"RMSE improvement: -{rmse_improvement:.4f} (better)")
    else:
        print(f"RMSE change: +{abs(rmse_improvement):.4f} (slight increase expected with more features)")
else:
    print("RMSE information not available")

print("\n\nImprovement Summary:")
baseline_accuracy = 67.7
if 'directional_accuracy' in successful.columns and len(successful) > 0:
    new_accuracy = successful['directional_accuracy'].mean()
    improvement = new_accuracy - baseline_accuracy

    if improvement > 0:
        print(f"✅ IMPROVEMENT: +{improvement:.1f}% directional accuracy")
        print(f"   ({baseline_accuracy:.1f}% → {new_accuracy:.1f}%)")
    else:
        print(f"⚠️  No improvement yet: {improvement:.1f}%")
        print(f"   This might improve with more data or different stocks")

    print("\n\nExpected after Week 1: 73-75% accuracy")
    print(f"Current: {new_accuracy:.1f}% accuracy")

    if new_accuracy >= 73:
        print("🎉 TARGET ACHIEVED!")
    elif new_accuracy >= 70:
        print("📈 Good progress, close to target")
    else:
        print("📊 Initial results - may need more tuning")
        print("\nPossible next steps:")
        print("  - Try increasing n_lags to 5 for more historical context")
        print("  - Ensure sufficient training data (3+ years)")
        print("  - Consider adding sector-relative features (Week 2)")
else:
    print("Unable to calculate improvement - directional accuracy not available")

print("\n" + "="*60)
print("Analysis complete!")
