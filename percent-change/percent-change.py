def percent_change(series: list) -> list:
    """
    Returns the fractional change between consecutive values.
    """
    # Write code here
    percent_change = []
    n = len(series)
    
    for i in range(1, n):
        if series[i-1] == 0:
            percent_change.append(0.0)
        else:
            p_i = (series[i] - series[i-1])/series[i-1]
            percent_change.append(p_i)

    return  percent_change