"""
Real-Time Neural Decoding Pipeline

Flow:
1. Acquire neural signal
2. Process signal (filtering)
3. Extract features
4. Feed into ML model
"""

from acquisition import *   # or specific functions
from processing import *
from ml_node import *

def main():
    print("Starting neural pipeline...")

    # Step 1: get signal (replace with your actual function)
    signal = get_data()  

    # Step 2: process signal
    processed = process_signal(signal)

    # Step 3: extract features
    features = extract_features(processed)

    # Step 4: model prediction
    output = run_model(features)

    print("Pipeline complete.")
    return output


if __name__ == "__main__":
    main()
