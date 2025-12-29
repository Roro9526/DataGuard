import sys
import os
import pandas as pd

# Add current directory to path so we can import src
sys.path.append(os.getcwd())

from src.data_loader import load_data, get_stats
from src.model import detect_anomalies

def test_workflow():
    print("Testing DataGuard Workflow...")
    
    # 1. Test Data Loading
    print("[1/3] Testing Data Loading...")
    if not os.path.exists("ventes.csv"):
        print("ERROR: ventes.csv not found.")
        return
    
    df = load_data("ventes.csv")
    if df.empty:
        print("ERROR: DataFrame is empty.")
        return
    
    if 'date' not in df.columns or 'ventes' not in df.columns:
        print("ERROR: Missing columns.")
        return
        
    print(f"OK: Loaded {len(df)} rows.")

    # 2. Test Stats
    print("[2/3] Testing Stats...")
    stats = get_stats(df)
    print(f"OK: Stats generated: {stats}")

    # 3. Test Anomaly Detection
    print("[3/3] Testing Anomaly Detection...")
    df_anom = detect_anomalies(df, contamination=0.05)
    if 'anomaly' not in df_anom.columns:
        print("ERROR: 'anomaly' column missing.")
        return
        
    num_anomalies = df_anom['anomaly'].sum()
    print(f"OK: Detected {num_anomalies} anomalies.")
    
    print("\nSUCCESS: All checks passed.")

if __name__ == "__main__":
    test_workflow()
