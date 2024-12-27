import pandas as pd
from datasets import load_dataset
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def return_nearest_desc(titles, desc_corpus, faiss_index):
    """
    Find the nearest descriptions in the corpus for a given list of titles using a FAISS index.

    Parameters:
    - titles (list of str): The list of titles for which nearest descriptions are to be found.
    - desc_corpus (list of str): The corpus of descriptions from which to find the nearest matches.
    - faiss_index (faiss.Index): The FAISS index built from the embeddings of the descriptions.

    Returns:
    - ret_descs (list of str): A list of nearest descriptions corresponding to the input titles.
    """
    ret_descs = []

    # Load embedding model only once outside the loop for efficiency
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to('cuda')

    # Use tqdm to track progress during the search for nearest descriptions
    for i in tqdm(range(len(titles)), desc="Finding nearest descriptions"):
        # Encode the title into its corresponding embedding
        query_embedding = embedding_model.encode([titles[i]])

        # Search for the nearest description in the FAISS index
        distance, index = faiss_index.search(query_embedding, 1)

        # Retrieve the index of the nearest description
        index = index[0][0]

        # Append the nearest description to the list
        ret_descs.append(desc_corpus[index])

    return ret_descs

# Load training and final test feature data from JSON files
train_features = pd.read_json('./phase_2_input_data(final)/training_data/train.features', lines=True)
final_test_features = pd.read_json('./phase_2_input_data(final)/final_test_data/final_test_data.features', lines=True)

# Create a 'title' column in train_features by copying 'description' column and drop 'description'
train_features['title'] = train_features['description']
train_features = train_features.drop(['description'], axis=1)

# Create a 'title' column in final_test_features by copying 'description' column and drop 'description'
final_test_features['title'] = final_test_features['description']
final_test_features = final_test_features.drop(['description'], axis=1)

# Load external dataset containing product descriptions
ext_dataset = load_dataset("Ateeqq/Amazon-Product-Description")

# Load pre-computed vectors for the descriptions from a NumPy file
vectors = np.load('ext_vectors.npy')

# Create a FAISS index for L2 distance with 384 dimensions
index = faiss.IndexFlatL2(384)

# Add the description vectors to the FAISS index
index.add(vectors)

# Convert titles from train_features DataFrame to a list
train_titles = train_features['title'].to_list()
# Extract descriptions from the external dataset
descriptions = ext_dataset['train']['DESCRIPTION']

# Find the nearest descriptions for the training titles
train_descs = return_nearest_desc(train_titles, descriptions, index)

# Add the nearest descriptions to the train_features DataFrame
train_features['description'] = train_descs

# Convert titles from final_test_features DataFrame to a list
test_titles = final_test_features['title'].to_list()
# Extract descriptions again from the external dataset
descriptions = ext_dataset['train']['DESCRIPTION']

# Find the nearest descriptions for the test titles
test_descs = return_nearest_desc(test_titles, descriptions, index)

# Add the nearest descriptions to the final_test_features DataFrame
final_test_features['description'] = test_descs

# Save the processed training features and final test features to CSV files
train_features.to_csv('processed_train_features.csv', index=False)
final_test_features.to_csv('processed_test_features.csv', index=False)
