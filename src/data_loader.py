import pandas as pd

def load_data(filepath):
    """
    Load sales data from CSV file.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataframe with parsed dates.
    """
    try:
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def get_stats(df):
    """
    Calculate basic statistics for the sales column.
    Args:
        df (pd.DataFrame): Dataframe containing 'ventes' column.
    Returns:
        dict: Dictionary with mean, median, std, min, max.
    """
    if df.empty:
        return {}
    
    stats = {
        'mean': df['ventes'].mean(),
        'median': df['ventes'].median(),
        'std': df['ventes'].std(),
        'min': df['ventes'].min(),
        'max': df['ventes'].max()
    }
    return stats
