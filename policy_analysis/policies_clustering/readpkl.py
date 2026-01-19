import numpy as np

# Path to your .npy file
file_path = 'src/policy_analysis/policies_clustering/labels.npy'

# Load the .npy file
try:
    data = np.load(file_path, allow_pickle=True)
    print("File loaded successfully.\n")
    
    # Check the type and shape
    print(f"Data type: {type(data)}")
    if isinstance(data, np.ndarray):
        print(f"Array shape: {data.shape}")
        # Show top 5 rows/elements depending on dimensions
        if data.ndim == 1:
            print("Top 5 elements:")
            print(data[:100])  # Show first 100 elements for 1D arrays
        else:
            print("Top 5 rows:")
            print(data[:100, ...])  # Show first 100 rows for multi-dimensional arrays
    else:
        print("The .npy file does not contain a NumPy array.")
        print("Content preview:", str(data)[:500])
except Exception as e:
    print(f"Error loading file: {e}")
