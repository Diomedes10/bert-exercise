import pandas as pd
import os
import numpy as np

def process_sst(raw_dir, output_file):
    print("Loading files...")
    sentences = pd.read_csv(os.path.join(raw_dir, 'datasetSentences.txt'), sep='\t', header=0, names=['sentence_id', 'sentence'])
    labels = pd.read_csv(os.path.join(raw_dir, 'sentiment_labels.txt'), sep='|', header=0, names=['phrase_id', 'sentiment'])
    dictionary = pd.read_csv(os.path.join(raw_dir, 'dictionary.txt'), sep='|', header=0, names=['phrase', 'phrase_id'])
    splits = pd.read_csv(os.path.join(raw_dir, 'datasetSplit.txt'), sep=',', header=0, names=['sentence_id', 'split'])

    print(f"Initial data shapes:")
    print(f"Sentences: {sentences.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Dictionary: {dictionary.shape}")
    print(f"Splits: {splits.shape}")

    print("Merging data...")
    # Merge dictionary with labels to get sentiment for each phrase
    data = pd.merge(dictionary, labels, on='phrase_id')
    print(f"After dictionary-labels merge: {data.shape}")

    # Add sentence_id to the data
    data = pd.merge(data, sentences[['sentence_id', 'sentence']], 
                    left_on='phrase', right_on='sentence', how='left')
    print(f"After adding sentence_id: {data.shape}")

    # Check for missing sentences
    missing_sentences = data[data['sentence_id'].isnull()]
    print(f"Missing sentences after merging: {missing_sentences.shape}")

    # Merge with splits
    data = pd.merge(data, splits, on='sentence_id', how='left')
    print(f"After merging with splits: {data.shape}")

    print("Original sentiment distribution:")
    print(data['sentiment'].describe())

    print("Binning sentiment scores...")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
    labels = [0, 1, 2, 3, 4]
    data['label'] = pd.cut(data['sentiment'], bins=bins, labels=labels, include_lowest=True)
    print(f"Data shape after binning: {data.shape}")
    print(f"Label distribution after binning:\n{data['label'].value_counts().sort_index()}")

    # Handle any NaN values in split (assign to train by default)
    data['split'] = data['split'].fillna(1)
    data['split'] = data['split'].map({1: 'train', 2: 'test', 3: 'dev'})

    # Select final columns
    data = data[['sentence', 'label', 'split', 'sentiment']]
    data = data.rename(columns={'sentence': 'text'})

    print("Saving processed data...")
    data.to_csv(output_file, index=False)
    print(f"Processed SST data saved to {output_file}")
    print(f"Final data shape: {data.shape}")
    print(f"Final label distribution:\n{data['label'].value_counts().sort_index()}")
    print(f"Final split distribution:\n{data['split'].value_counts()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess the SST dataset")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the SST dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed files')
    args = parser.parse_args()
    
    output_file = os.path.join(args.output_dir, 'sst_processed.csv')
    process_sst(args.data_dir, output_file)
