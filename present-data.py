import os
import pandas as pd
import argparse

# Pandas options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_seq_items', None)

# Get filename as argument
parser = argparse.ArgumentParser(prog='present-data.py')
parser.add_argument('path')
args = parser.parse_args()

# Parse and present data
gloss_and_id = pd.read_csv(args.path, dtype={'id':'object', 'gloss':'category'})
num_gestures = gloss_and_id['gloss'].nunique()

print(f'Total number of gestures in file {args.path}: {num_gestures}')
print(gloss_and_id['gloss'].value_counts())
