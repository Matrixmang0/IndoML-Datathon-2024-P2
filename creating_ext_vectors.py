import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load the embedding model and move it to the GPU
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to('cuda')

# Load the external dataset containing titles
ext_dataset = load_dataset("Ateeqq/Amazon-Product-Description")
docs = ext_dataset['train']['TITLE']

# Set the batch size for encoding
batch_size = 32

# Initialize a list to hold the vectors
vectors = []

# Process the documents in batches with a progress bar
for i in tqdm(range(0, len(docs), batch_size), desc="Encoding Titles"):
    batch_docs = docs[i:i + batch_size]  # Get the current batch
    batch_vectors = embedding_model.encode(batch_docs, show_progress_bar=False)  # Encode the batch
    vectors.append(batch_vectors)  # Store the batch vectors

# Concatenate all the vectors into a single NumPy array
vectors = np.vstack(vectors)

# Save the vectors to a NumPy file for later use
np.save('ext_vectors.npy', vectors)

print(f"Encoded {len(docs)} titles into vectors and saved to 'ext_vectors.npy'")
