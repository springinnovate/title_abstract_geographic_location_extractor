"""Extract geogrphic locations from research titles."""
from collections import defaultdict
import chardet
import os
import re

from transformers import pipeline
from datasets import Dataset
import pandas as pd
import textwrap


TABLE_LIST = [
    ('data/Civitello_.xlsx', 0),
    ('data/Methorst et al. 2020.csv', 'Title'),
    ('data/Mahon.csv', 'Citation'),
    ('data/Non-quantitative reviews.xlsx', 0),
    ('data/Quantitative non-located reviews.xlsx', 0),
    ('data/Ratto.xlsx', 0),
    ('data/Ripple et al. 2014_.xlsx', 0),
]


LOCATION_HEADER = 'geographic location(s)'
BROKEN_LINES_TABLES_SHEETS = ['Chardonnet']


def detect_encoding(csv_path):
    # Read a portion of the file to detect encoding
    with open(csv_path, 'rb') as f:
        result = chardet.detect(f.read(100000))  # You can adjust the number of bytes to read
        encoding = result['encoding']
    return encoding


def get_column(df, column):
    if isinstance(column, int):
        col_df = df.iloc[:, [column]].copy()
        col_df.columns = [f'col_{column}']
        return Dataset.from_pandas(col_df).to_list()
    elif isinstance(column, str):
        return Dataset.from_pandas(df[[column]]).to_list()


def chunk_texts(texts, max_length=512):
    chunked_texts = []
    text_indices = []
    for idx, text_dict in enumerate(texts):
        text = next(iter(text_dict.values()))
        if text is None:
            text = ''
        chunks = textwrap.wrap(
            text,
            width=max_length,
            subsequent_indent=' ',
            break_long_words=False,
            break_on_hyphens=False
        )
        chunked_texts.extend(chunks)
        text_indices.extend([idx] * len(chunks))
    return chunked_texts, text_indices


def extract_locations_from_chunks(ner_pipeline, chunked_texts, batch_size=8):
    # Process the chunks in batches
    ner_results = ner_pipeline(chunked_texts, batch_size=batch_size)
    return ner_results


def aggregate_locations(ner_results, text_indices):
    location_sets = defaultdict(set)

    for result, idx in zip(ner_results, text_indices):
        for entity in result:
            if entity['entity_group'] == 'LOC':
                location = entity['word'].strip().replace('##', '')
                location_sets[idx].add(location)

    num_texts = max(text_indices) + 1
    location_list = []
    for idx in range(num_texts):
        locations = ';'.join(location_sets[idx]) if idx in location_sets else ''
        location_list.append(locations)
    return location_list


def extract_locations_large_text(ner_pipeline, text, max_length=512, stride=256):
    chunks = textwrap.wrap(
        text, width=max_length, subsequent_indent=' ',
        break_long_words=False, break_on_hyphens=False)
    location_set = set()
    for chunk in chunks:
        ner_results = ner_pipeline(chunk)
        for entity in ner_results:
            if entity['entity_group'] == 'LOC':
                # Clean up the entity word
                location = entity['word'].strip()
                # Remove any leading '##' from subword tokens (shouldn't be necessary with grouping)
                location = location.replace('##', '')
                if location not in location_set:
                    location_set.add(location)
    return ';'.join(location_set)


def rootname(path):
    return os.path.basename(os.path.splitext(path)[0])


def process_df(df, title_column, ner_pipeline):
    title_list = get_column(df, title_column)
    chunked_texts, text_indices = chunk_texts(title_list, max_length=512)
    ner_results = extract_locations_from_chunks(ner_pipeline, chunked_texts, batch_size=8)
    location_list = aggregate_locations(ner_results, text_indices)
    df[LOCATION_HEADER] = location_list


def process_csv(csv_path, title_column, ner_pipeline):
    df = pd.read_csv(
        csv_path,
        encoding=detect_encoding(csv_path),
        header=None if isinstance(title_column, int) else 0)
    process_df(df, title_column, ner_pipeline)
    df.to_csv(f'output{rootname(csv_path)}.csv', index=False)


def merge_broken_rows(df):
    # Initialize variables
    entries = []
    current_entry = ''

    for row in df.itertuples(index=False):
        line = str(row[0]).strip()

        if re.match(r'^\d+\.', line):
            if current_entry:
                entries.append(current_entry.strip())
            current_entry = line
        else:
            current_entry += ' ' + line

    if current_entry:
        entries.append(current_entry.strip())

    combined_df = pd.DataFrame(entries, columns=['Entry'])
    return combined_df


def process_xlxs(file_path, title_column, ner_pipeline):
    xls = pd.ExcelFile(file_path)

    processed_sheets = {}
    for sheet_name in xls.sheet_names:
        print(f'processing {file_path}:{sheet_name}')
        header = None if isinstance(title_column, int) else 0
        df = pd.read_excel(xls, sheet_name=sheet_name, header=header)
        if sheet_name in BROKEN_LINES_TABLES_SHEETS:
            df = merge_broken_rows(df)
        process_df(df, title_column, ner_pipeline)
        processed_sheets[sheet_name] = df

    with pd.ExcelWriter(file_path, mode='w', engine='openpyxl') as writer:
        for sheet_name, df in processed_sheets.items():
            # Write each DataFrame back to the Excel file
            header = True if df.columns.any() else False
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=header)


def main():
    ner_pipeline = pipeline(
        "ner",
        grouped_entities=True,
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        device='cuda')

    for table_path, title_header in TABLE_LIST:
        if table_path.endswith('.csv'):
            process_csv(table_path, title_header, ner_pipeline)
        else:
            process_xlxs(table_path, title_header, ner_pipeline)


if __name__ == '__main__':
    main()
