from sklearn.ensemble import IsolationForest
import pandas as pd

def detect_anomalies(df, contamination=0.05):
    """
    Detect anomalies in sales data using Isolation Forest.
    Args:
        df (pd.DataFrame): Dataframe containing 'ventes' column.
        contamination (float): The proportion of outliers in the data set.
    Returns:
        pd.DataFrame: Dataframe with an 'anomaly' column (-1 for anomaly, 1 for normal).
    """
    if df.empty:
        return df

    model = IsolationForest(contamination=contamination, random_state=42)
    # Reshape data for model (sklearn expects 2D array)
    X = df[['ventes']]
    
    # Fit and predict
    df['anomaly_score'] = model.fit_predict(X)
    
    # Map to simpler boolean or string for UI
    # -1 is anomaly, 1 is normal
    df['anomaly'] = df['anomaly_score'].apply(lambda x: True if x == -1 else False)
    
    return df
