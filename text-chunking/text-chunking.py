def text_chunking(tokens: list, chunk_size: int, overlap: int) -> list:
    """
    Returns fixed-size token chunks with the requested overlap.
    """
    # Write code here
    step = chunk_size - overlap
    n = len(tokens)
    chunks_list = []
    for i in range(0, n, step):
        chunk_len = min(i+chunk_size, n)
        chunk = tokens[i:chunk_len]
        chunks_list.append(chunk)
        if chunk_len >= n:
            break
    return chunks_list