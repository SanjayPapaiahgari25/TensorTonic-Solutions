import pandas as pd

def create_dataframe(data):
    """
    Returns: dict with 'data', 'shape', 'columns'
    """
    df = pd.DataFrame(data)
    df_dict = df.to_dict("list")
    return {"data": df_dict,"shape": list(df.shape), "columns": list(df.columns)}