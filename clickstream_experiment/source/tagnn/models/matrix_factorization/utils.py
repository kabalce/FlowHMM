import numpy as np

from src.models.embeddings.item2vec import Item2Vec
from src.utils.utils import normalize


def generate_embeddings(train_pdf, n_items, embedding_dim, user_col='userID', item_col='itemID'):
    rng = np.random.default_rng()
    embedding_model = Item2Vec(vector_size=embedding_dim)
    embedding_model.train(train_pdf, item_col=item_col, user_col=user_col, epochs=3)
    embeddings_pdf = embedding_model.generate_item_embeddings()
    embeddings = rng.normal(scale=0.1, size=(n_items, embedding_dim))
    embeddings[embeddings_pdf.index] = normalize(embeddings_pdf.values, new_std=0.1)
    return embeddings