import pandas as pd
import os

def process_sst(raw_dir, output_file):
    print("Loading files...")
    sentences = pd.read_csv(os.path.join(raw_dir, 'datasetSentences.txt'), sep='\t', header=0, names=['sentence_id', 'text'])
    labels = pd.read_csv(os.path.join(raw_dir, 'sentiment_labels.txt'), sep='|', header=0, names=['phrase_id', 'label'])
    dictionary = pd.read_csv(os.path.join(raw_dir, 'dictionary.txt'), sep='|', header=0, names=['phrase', 'phrase_id'])
    splits = pd.read_csv(os.path.join(raw_dir, 'datasetSplit.txt'), sep=',', header=0, names=['sentence_id', 'split'])

    print(f"Initial data shapes:")
    print(f"Sentences: {sentences.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Dictionary: {dictionary.shape}")
    print(f"Splits: {splits.shape}")

    print("Merging data...")
    data = pd.merge(dictionary, labels, on='phrase_id')
    print(f"After dictionary-labels merge: {data.shape}")
    data = pd.merge(data, sentences, left_on='phrase', right_on='text', how='inner')
    print(f"After merging with sentences: {data.shape}")

    print("Binning sentiment scores...")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
    labels = [0, 1, 2, 3, 4]
    data['label'] = pd.cut(data['label'], bins=bins, labels=labels)
    print(f"Data shape after binning: {data.shape}")
    print(f"Label distribution:\n{data['label'].value_counts().sort_index()}")

    print("Merging with splits...")
    data = pd.merge(data, splits, on='sentence_id')
    print(f"Final data shape: {data.shape}")

    data = data[['text', 'label', 'split']]
    data['split'] = data['split'].map({1: 'train', 2: 'test', 3: 'dev'})

    print("Saving processed data...")
    data.to_csv(output_file, index=False)
    print(f"Processed SST data saved to {output_file}")
    print(f"Final data shape: {data.shape}")
    print(f"Final label distribution:\n{data['label'].value_counts().sort_index()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess the SST dataset")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the SST dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed files')
    args = parser.parse_args()
    output_file = os.path.join(args.output_dir, 'sst_processed.csv')
    process_sst(args.data_dir, output_file)