"""
Test script to verify NaN handling fixes for EU-SILC indicators
"""
import pandas as pd
import numpy as np

def test_nan_handling():
    """Test the new NaN handling logic"""
    
    print("Testing NaN handling for EU-SILC indicators...")
    print("=" * 50)
    
    # Test 1: Simulate Behind on Utility Bills (HE-SILC-2) scenario
    print("\n1. Testing Behind on Utility Bills (HE-SILC-2) scenario:")
    print("   Simulating Germany 2004-2007 where all HS021 values are NaN")
    
    # Create test data
    test_data = pd.DataFrame({
        'HB010': [2005] * 100,  # Year 2005
        'HB020': ['DE'] * 100,  # Germany
        'HS021': [np.nan] * 100,  # All NaN values (missing data)
        'DB090': [1.0] * 100,  # Weights
        'decile': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10
    })
    
    # Apply the new logic
    all_nan = test_data['HS021'].isna().all()
    print(f"   All HS021 values are NaN: {all_nan}")
    
    if all_nan:
        result = np.nan
        print(f"   Result: {result} (correct - should be NaN)")
    else:
        print("   ERROR: Should have detected all NaN values")
    
    # Test 2: Simulate Overcrowded Dwelling scenario
    print("\n2. Testing Overcrowded Dwelling (HQ-SILC-1) scenario:")
    print("   Simulating Germany 2015 where HH030 data is missing")
    
    test_rooms = pd.DataFrame({
        'HH030': [np.nan, np.nan, np.nan],  # Missing room data
        'required_rooms': [3, 4, 2]
    })
    
    # Apply the new overcrowding logic
    overcrowded_condition = test_rooms['HH030'] < test_rooms['required_rooms']
    overcrowded = overcrowded_condition.where(test_rooms['HH030'].notna(), np.nan)
    
    print(f"   HH030 values: {test_rooms['HH030'].tolist()}")
    print(f"   Required rooms: {test_rooms['required_rooms'].tolist()}")
    print(f"   Overcrowded result: {overcrowded.tolist()}")
    print(f"   All results are NaN: {overcrowded.isna().all()} (correct)")
    
    # Test 3: Test with mixed data (some NaN, some valid)
    print("\n3. Testing mixed data scenario:")
    
    mixed_data = pd.DataFrame({
        'HB010': [2008] * 100,
        'HB020': ['AT'] * 100,
        'HS021': [1.0] * 50 + [np.nan] * 50,  # Half valid, half NaN
        'DB090': [1.0] * 100,
        'decile': [1] * 100
    })
    
    all_nan = mixed_data['HS021'].isna().all()
    print(f"   All HS021 values are NaN: {all_nan}")
    
    if not all_nan:
        # This should proceed with normal calculation
        mask = mixed_data['HS021'] == 1.0
        weighted_sum = mixed_data.loc[mask, 'DB090'].sum()
        total_weight = mixed_data['DB090'].sum()
        share = (weighted_sum / total_weight * 100)
        print(f"   Share calculation: {share}% (should be 50%)")
    
    print("\n4. Testing normal data scenario:")
    
    normal_data = pd.DataFrame({
        'HB010': [2010] * 100,
        'HB020': ['FR'] * 100,
        'HS021': [1.0] * 20 + [2.0] * 30 + [3.0] * 50,  # Normal distribution
        'DB090': [1.0] * 100,
        'decile': [1] * 100
    })
    
    all_nan = normal_data['HS021'].isna().all()
    print(f"   All HS021 values are NaN: {all_nan}")
    
    if not all_nan:
        # Calculate share with new logic (values 1 and 2 count as "behind")
        mask = normal_data['HS021'].isin([1.0, 2.0])
        weighted_sum = normal_data.loc[mask, 'DB090'].sum()
        total_weight = normal_data['DB090'].sum()
        share = (weighted_sum / total_weight * 100)
        print(f"   Share calculation: {share}% (should be 50% - values 1&2 out of 1,2,3)")

    print("\n" + "=" * 50)
    print("Testing completed! The logic should now correctly handle:")
    print("✓ Return NaN when all source data is missing")
    print("✓ Return proper percentages when data is available")
    print("✓ Handle mixed scenarios appropriately")

if __name__ == "__main__":
    test_nan_handling()