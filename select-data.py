import os
import pandas as pd
import argparse

# Pandas options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_seq_items', None)

# Get args
parser = argparse.ArgumentParser(prog='present-data.py')
parser.add_argument('--path', default='processed-videos.csv', help='File path for processed videos CSV file')
parser.add_argument('--threshold', type=int, help='Select all glosses with frequencies equal or greater than threshold')
parser.add_argument('--top', help='Select top N most frequent glosses')
args = parser.parse_args()

# Parse and present data
gloss_and_id = pd.read_csv(args.path, dtype={'id':'object', 'gloss':'category'})
num_gestures = gloss_and_id['gloss'].nunique()

print(f'Total number of gestures in file {args.path}: {num_gestures}')

if args.threshold is not None:
    counts = gloss_and_id['gloss'].value_counts()
    frequent_glosses = counts[counts >= args.threshold].index

    filtered = gloss_and_id[gloss_and_id['gloss'].isin(frequent_glosses)]
    filtered['gloss'] = filtered['gloss'].cat.remove_unused_categories()

    print(f'Number of selected glosses: {len(filtered)}')
    print(f'Number of unique glosses: {filtered["gloss"].nunique()}')
    print(f'\nSelected glosses:\n{filtered["gloss"].value_counts()}')

    filtered.to_csv('processed-videos-filtered.csv', index=False)
    print('Data saved to CSV.')
