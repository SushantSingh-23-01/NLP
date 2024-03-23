import torch
import pickle
import numpy as np

with open('embedding.pkl', 'rb') as fp:
    embeddings = pickle.load(fp)

def cos_sim(id1,id2):
    return np.dot(id1,id2) / (np.linalg.norm(id1) * np.linalg.norm(id2))

def get_vector(word):
    if embeddings[word] is not None:
        vector = np.array(embeddings.get(word))
    else:
        vector = np.zeros(np.shape(embeddings.values()[0]))
    return vector

v1 = get_vector('gpt')
v2 = get_vector('transformers')

print(f'Cosine similarity {cos_sim(v1,v2)}')
