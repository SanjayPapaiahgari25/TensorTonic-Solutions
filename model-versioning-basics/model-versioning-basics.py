def promote_model(models: list) -> str:
    """
    Returns the model name as a string.
    """
    # Write code here
    max_accuracy = -float('inf')
    lowest_latency = float('inf')
    latest_timestamp='0000-00-00'
    best_model = None
    for model in models:
        if model['accuracy'] > max_accuracy:
            max_accuracy = model['accuracy']
            lowest_latency = model['latency']
            latest_timestamp = model['timestamp']
            best_model = model['name']
        elif model['accuracy'] == max_accuracy:
            if model['latency'] < lowest_latency:
                lowest_latency = model['latency']
                latest_timestamp = model['timestamp']
                best_model = model['name']
            elif model['latency'] == lowest_latency:
                if model['timestamp'] > latest_timestamp:
                    latest_timestamp = model['timestamp']
                    best_model = model['name']

    return best_model
            