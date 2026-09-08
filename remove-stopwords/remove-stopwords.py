def remove_stopwords(tokens: list, stopwords: list) -> list:
    """
    Returns a list of tokens.
    """
    # Write code here
    non_stopwords_list = []

    for token in tokens:
        if token in stopwords:
            continue
        non_stopwords_list.append(token)

    return non_stopwords_list