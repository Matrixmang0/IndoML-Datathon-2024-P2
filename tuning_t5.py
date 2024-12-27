import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
import torch
import numpy as np
import os
import json
import zipfile

# Loading DataFrames
train_features = pd.read_csv('./data/processed_train_features.csv')
train_labels = pd.read_json('./data/train.labels', lines=True)
final_test_features = pd.read_csv('./data/processed_test_features.csv')

# Imputing NaN Values
train_features['price'].fillna(value='Unknown', inplace=True)
final_test_features['price'].fillna(value='Unknown', inplace=True)

def combine_infos(row):
    """
    Combines product details into a structured text format.

    Args:
        row (pd.Series): A row from the DataFrame containing product details.

    Returns:
        str: A formatted string containing the product title, description, retailer, and price.
    """
    text = (
        'The title of the product is ' + row['title'] + 
        '. The description of the product is ' + row['description'] + 
        '. This product was bought from the retailer ' + row['retailer'] + 
        '. The price of this product is ' + str(row['price']) + '.'
    )
    return text

def preprocess_data(data):
    """
    Preprocesses the input DataFrame to create a new DataFrame with combined product information.

    Args:
        data (pd.DataFrame): The DataFrame containing product features.

    Returns:
        pd.DataFrame: A new DataFrame with 'indoml_id' and 'input_text' columns.
    """
    df = pd.DataFrame()
    df['indoml_id'] = data['indoml_id']  # Retaining the unique identifier
    df['input_text'] = data.apply(combine_infos, axis=1)  # Combining product information into text
    return df

def preprocess_target(solution):
    """
    Preprocesses the target DataFrame to create a structured target text format.

    Args:
        solution (pd.DataFrame): The DataFrame containing target labels.

    Returns:
        pd.DataFrame: A new DataFrame with 'indoml_id' and 'target_text' columns.
    """
    df = pd.DataFrame()
    df['indoml_id'] = solution['indoml_id']  # Retaining the unique identifier
    df['target_text'] = solution.apply(
        lambda row: f"supergroup: {row['supergroup']} group: {row['group']} module: {row['module']} brand: {row['brand']}", 
        axis=1  # Creating structured target text
    )
    return df

# Preprocess the training features and labels
trftrs_processed = preprocess_data(train_features)
trtgt_processed = preprocess_target(train_labels)
tstftrs_processed = preprocess_data(final_test_features)

# Split the processed training features and targets into training and validation sets
trftrs_processed, valftrs_processed, trtgt_processed, valtgt_processed = train_test_split(
    trftrs_processed,
    trtgt_processed,
    test_size=0.05,
    random_state=42  # Ensuring reproducibility of the split
)

# Create training and validation datasets by merging features and targets on 'indoml_id'
train_dataset = Dataset.from_pandas(pd.merge(trftrs_processed, trtgt_processed, on='indoml_id'))
val_dataset = Dataset.from_pandas(pd.merge(valftrs_processed, valtgt_processed, on='indoml_id'))
test_dataset = Dataset.from_pandas(tstftrs_processed)

# Create a DatasetDict to organize the datasets
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

# Load the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')

def preprocess_function(examples):
    """
    Preprocesses input examples for the T5 model by tokenizing input and target texts.

    Args:
        examples (dict): A dictionary containing 'input_text' and 'target_text' fields.

    Returns:
        dict: A dictionary containing tokenized inputs and targets suitable for the model.
    """
    inputs = examples['input_text']
    targets = examples['target_text']
    
    # Tokenizing input texts with padding and truncation
    model_inputs = tokenizer(inputs, max_length=256, padding='max_length', truncation=True)
    
    # Tokenizing target texts with padding and truncation
    labels = tokenizer(targets, max_length=256, padding='max_length', truncation=True)

    # Adding the tokenized labels to model inputs
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Tokenizing the datasets using the preprocess_function
tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)

# Setting up training arguments for the Trainer
training_args = TrainingArguments(
    output_dir='./tuned_fT5small',  # Directory to save model checkpoints
    eval_strategy='epoch',  # Evaluate at the end of each epoch
    eval_accumulation_steps=32,  # Accumulate evaluation results for specified steps
    save_strategy='epoch',  # Save model checkpoints at the end of each epoch
    save_total_limit=3,  # Limit the total number of checkpoints to save
    learning_rate=2e-3,  # Learning rate for the optimizer
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    num_train_epochs=3,  # Number of training epochs
    weight_decay=0.01,  # Weight decay for regularization
    logging_first_step=True,  # Log the first step
    logging_dir='./tensorboard_logs',  # Directory for storing TensorBoard logs
    logging_steps=500,  # Log every 500 steps
    report_to='tensorboard'  # Reporting logs to TensorBoard
)

class LoggingCallback(TrainerCallback):
    """
    A callback class to log training information during the training process.

    Args:
        log_dir (str): Directory to save the log file.
        log_file (str): Name of the log file.
    """
    
    def __init__(self, log_dir='./logs', log_file='training_log_T5Small_3E.txt'):
        super().__init__()
        self.log_dir = log_dir
        self.log_file = log_file
        os.makedirs(self.log_dir, exist_ok=True)  # Create log directory if it does not exist
        self.log_path = os.path.join(self.log_dir, self.log_file)
        
        # Initialize the log file with training statistics
        with open(self.log_path, 'w') as f:
            f.write("Number of Training Datapoints: " + str(len(tokenized_datasets['train'])) + "\n")
            f.write("Number of Validation Datapoints: " + str(len(tokenized_datasets['validation'])) + "\n\n")
            f.write("Training Logs\n\n")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Logs the training information at specified steps.

        Args:
            args: The training arguments.
            state: The current training state.
            control: The current control state.
            logs (dict, optional): Logs to write to the log file.
        """
        if logs is not None:
            with open(self.log_path, 'a') as f:
                f.write(f"Step: {state.global_step}\n")
                print(f"Step: {state.global_step}\n")
                
                for key, value in logs.items():
                    f.write(f"{key}: {value}\n")
                    print(f"{key}: {value}\n")
                f.write("\n")
                print("\n")

# Initializing the Trainer with the model, training arguments, datasets, and callbacks
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    callbacks=[LoggingCallback()]  # Adding the logging callback
)

# Starting the training process
trainer.train()

# Save the fine-tuned model and tokenizer to a specified directory
model.save_pretrained('./finetuned_t5small_3E')
tokenizer.save_pretrained('./finetuned_t5small_3E')

# Determine the device to be used (GPU or CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the fine-tuned model and tokenizer from the saved directory
model = T5ForConditionalGeneration.from_pretrained('./finetuned_t5small_3E').to(device)
tokenizer = T5Tokenizer.from_pretrained('./finetuned_t5small_3E')

# Set the model to evaluation mode
model.eval()

# Extracting input texts from the test dataset
test_input = test_dataset['input_text']

def generate_text(inputs):
    """
    Generates text outputs from the model based on the input texts.

    Args:
        inputs (list of str): A list of input texts to generate outputs for.

    Returns:
        list of str: A list of generated text outputs from the model.
    """
    # Tokenize the inputs, returning PyTorch tensors with padding and truncation
    inputs = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True, truncation=True, max_length=352)
    
    # Move the input tensors to the specified device (GPU or CPU)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():  # Disable gradient calculation for inference
        # Generate outputs from the model
        outputs = model.generate(**inputs, max_length=128)
    
    # Decode the generated outputs into human-readable text
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generated_texts

def extract_details(text):
    """
    Extracts structured details from the generated text using regex.

    Args:
        text (str): The generated text containing structured information.

    Returns:
        tuple: A tuple containing extracted details (supergroup, group, module, brand).
               If no match is found, returns ('na', 'na', 'na', 'na').
    """
    # Define a regex pattern to extract details from the text
    pattern = r'supergroup: (.*?) group: (.*?) module: (.*?) brand: (.*)'
    match = re.match(pattern, text)
    
    if match:
        # Return the extracted details, defaulting to 'na' if any are missing
        return tuple(item if item is not None else 'na' for item in match.groups())
    
    # If no match is found, return default values
    return 'na', 'na', 'na', 'na'

# Set the batch size for processing
batch_size = 128
generated_details = []  # List to store extracted details from generated texts
target_details = []  # List to store target details (if needed later)

# Process the test input in batches
for i in tqdm(range(0, len(test_input), batch_size), desc="Processing test data"):
    batch_inputs = test_input[i:i + batch_size]  # Select a batch of inputs
    
    # Generate texts for the current batch
    generated_texts = generate_text(batch_inputs)
    
    # Extract details from each generated text and store them
    for generated_text in generated_texts:
        generated_details.append(extract_details(generated_text))

print('Generated info extracted.............')

# Define the categories for the extracted details
categories = ['supergroup', 'group', 'module', 'brand']

# Write the extracted details to a JSON file
with open('test_final_v2.predict', 'w') as file:
    for indoml_id, details in enumerate(generated_details):
        result = {"indoml_id": indoml_id}  # Create a dictionary for each entry
        for category, value in zip(categories, details):
            result[category] = value  # Map each category to its extracted value
        
        file.write(json.dumps(result) + '\n')  # Write the result as a JSON string

# Zip the output file for submission
file_to_zip = 'test_final_v2.predict'
zip_file_name = 'submission_tf_V2.zip'

with zipfile.ZipFile(zip_file_name, 'w') as zipf:
    zipf.write(file_to_zip, arcname=file_to_zip)  # Add the file to the zip archive