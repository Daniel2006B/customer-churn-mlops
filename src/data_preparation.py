import pandas as pd
 
 
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
 
 
def prepare_features(df: pd.DataFrame, target_column: str = "Churn"):
    df = df.copy()
    y = df[target_column]
    X = df.drop(columns=[target_column])
 
    if y.dtype == "object":
        y = y.map({"No": 0, "Yes": 1}).fillna(y)
 
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)
    return X, y.astype(int)
