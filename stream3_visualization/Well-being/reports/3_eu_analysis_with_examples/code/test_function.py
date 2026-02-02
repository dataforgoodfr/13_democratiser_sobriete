import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from eurostat_analysis import create_dwellings_per_household_2011_2022

print("=" * 70)
print("Testing new function: create_dwellings_per_household_2011_2022")
print("=" * 70)

create_dwellings_per_household_2011_2022()

print("\nTest complete!")
