import math

def log_transform(values: list) -> list:
    """
    Returns the log1p-transformed values rounded to four decimals.
    """
    # Write code here
    y=[]
    for i in range(len(values)):
        y.append(math.log(1+values[i]))

    return y