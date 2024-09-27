"""Extract geogrphic locations from research titles."""
from collections import defaultdict
import pickle
import chardet
import os
import re

import geograpy
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

BROKEN_LINES_TABLES_SHEETS = ['Chardonnet']

LOCATION_FIELD = 'geographic location(s)'
OUTPUT_TABLE_PATH = 'geolocated_citations.csv'
CITATION_FIELD = 'citation'
SOURCE_FIELD = 'source file'
CONTINENT_FIELD = 'continent(s)'
COUNTRY_FIELD = 'country(ies)'

CORE_GEOGRAPHIC_INFO_TSV_PATH = 'geoinformation/allCountries.tsv'
COUNTRY_INFO_TSV_PATH = 'geoinformation/countryInfo.tsv'
CORE_GEOGRAPHIC_INFO_PKL_PATH = f'{os.path.basename(CORE_GEOGRAPHIC_INFO_TSV_PATH)}.pkl'

CONTINENT_CODE_TO_NAME = {
    'NA': 'North America',
    'SA': 'South America',
    'EU': 'Europe',
    'AS': 'Asia',
    'AF': 'Africa',
    'OC': 'Oceania',
}

PLACES_TO_REJECT = {
    'northern': "SD",
}


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
        return Dataset.from_pandas(col_df.astype(str)).to_list()
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


def process_df(df, citation_column, ner_pipeline):
    citation_list = get_column(df, citation_column)
    chunked_texts, text_indices = chunk_texts(citation_list, max_length=512)
    ner_results = extract_locations_from_chunks(ner_pipeline, chunked_texts, batch_size=8)
    location_list = aggregate_locations(ner_results, text_indices)
    location_index = 'col_0' if isinstance(citation_column, int) else citation_column
    return [x[location_index] for x in citation_list], location_list


def process_csv(csv_path, citation_column, ner_pipeline):
    df = pd.read_csv(
        csv_path,
        encoding=detect_encoding(csv_path),
        header=None if isinstance(citation_column, int) else 0)
    citation_list, location_list = process_df(df, citation_column, ner_pipeline)
    source_list = len(citation_list)*[os.path.basename(csv_path)]
    return source_list, citation_list, location_list


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


def process_xlxs(file_path, citation_column, ner_pipeline):
    xls = pd.ExcelFile(file_path)
    citation_list = []
    location_list = []
    source_list = []
    for sheet_name in xls.sheet_names:
        print(f'processing {file_path}:{sheet_name}')
        header = None if isinstance(citation_column, int) else 0
        df = pd.read_excel(xls, sheet_name=sheet_name, header=header)
        if sheet_name in BROKEN_LINES_TABLES_SHEETS:
            df = merge_broken_rows(df)
        local_citation_list, local_location_list = process_df(df, citation_column, ner_pipeline)
        local_source_list = len(local_citation_list) * [f'{os.path.basename(file_path)}:{sheet_name}']

        citation_list.append(local_citation_list)
        source_list.append(local_source_list)
        location_list.append(local_location_list)
    return source_list, citation_list, location_list


def main():
    country_info_df = pd.read_csv(
        COUNTRY_INFO_TSV_PATH,
        sep='\t',
        keep_default_na=False,
        usecols=['#ISO', 'Country', 'Continent'])
    country_info_dict = dict(zip(
        country_info_df['#ISO'], zip(country_info_df['Country'], country_info_df['Continent'])))
    if not os.path.exists(CORE_GEOGRAPHIC_INFO_PKL_PATH):
        core_geographic_df = pd.read_csv(
            CORE_GEOGRAPHIC_INFO_TSV_PATH,
            sep='\t',
            header=None,
            keep_default_na=False,
            usecols=[1, 7, 8])
        core_geographic_df = core_geographic_df[(core_geographic_df[7] == 'ADM1')]

        core_geographic_df = core_geographic_df.drop(columns=[7])

        core_geographic_dict = dict(zip(core_geographic_df[1].str.lower(), core_geographic_df[8]))
        with open(CORE_GEOGRAPHIC_INFO_PKL_PATH, 'wb') as f:
            pickle.dump(core_geographic_dict, f)
    else:
        with open(CORE_GEOGRAPHIC_INFO_PKL_PATH, 'rb') as f:
            core_geographic_dict = pickle.load(f)

    for country_code, (country_name, _) in country_info_dict.items():
        core_geographic_dict[country_name.lower()] = country_code

    if os.path.exists(OUTPUT_TABLE_PATH):
        os.remove(OUTPUT_TABLE_PATH)

    print('starting pipeline')
    ner_pipeline = pipeline(
        "ner",
        grouped_entities=True,
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        device='cuda')

    for table_path, title_header in TABLE_LIST:
        print(f'processing {table_path}')
        if table_path.endswith('.csv'):
            source_list, citation_list, location_list = process_csv(table_path, title_header, ner_pipeline)
        else:
            source_list, citation_list, location_list = process_xlxs(table_path, title_header, ner_pipeline)

        countries_list = []
        continent_list = []
        for location_str in location_list:
            local_country_set = set()
            local_continent_set = set()

            for location in location_str.split(';'):
                location = location.lower()
                if location in core_geographic_dict:
                    country_code = core_geographic_dict[location]
                    country_name, continent_id = country_info_dict[country_code]
                    local_country_set.add(country_name)
                    local_continent_set.add(CONTINENT_CODE_TO_NAME[continent_id])
            if not any([local_country_set, local_continent_set]) and location_str != '':
                # Create an Extractor object with the text
                extractor = geograpy.Extractor(text=location_str)
                extractor.find_entities()
                for place in extractor.places:
                    place = place.lower()
                    if place in core_geographic_dict:
                        country_code = core_geographic_dict[place]
                        if place in PLACES_TO_REJECT and PLACES_TO_REJECT[place] == country_code:
                            continue
                        country_name, continent_id = country_info_dict[country_code]
                        local_country_set.add(country_name)
                        local_continent_set.add(CONTINENT_CODE_TO_NAME[continent_id])

            countries_list.append(';'.join(local_country_set))
            continent_list.append(';'.join(local_continent_set))

        new_entries = pd.DataFrame({
            SOURCE_FIELD: source_list,
            CITATION_FIELD: citation_list,
            LOCATION_FIELD: location_list,
            COUNTRY_FIELD: countries_list,
            CONTINENT_FIELD: continent_list,
        })
        if os.path.exists(OUTPUT_TABLE_PATH):
            df = pd.read_csv(OUTPUT_TABLE_PATH)
            new_entries = pd.concat([df, new_entries], ignore_index=True)
        new_entries.to_csv(OUTPUT_TABLE_PATH, index=False)
    print('all done')

if __name__ == '__main__':
    main()
