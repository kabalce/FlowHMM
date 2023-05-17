def filter_dataset(interactions_pdf, min_items=5, min_session_len=5, item_col='itemID', user_col='userID'):
    item_counts = interactions_pdf[item_col].value_counts()
    interactions_pdf = interactions_pdf[interactions_pdf[item_col].isin(item_counts[item_counts >= min_items].index)]
    session_lens = interactions_pdf[user_col].value_counts()
    interactions_pdf = interactions_pdf[interactions_pdf[user_col].isin(session_lens[session_lens >= min_session_len].index)]
    return interactions_pdf
