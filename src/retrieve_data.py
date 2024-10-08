import re
import pandas as pd
import os
import time
from string import Template
import urllib.request


def get_documents_from_csv(original_metadata_file_path: str, documents_directory: str):
    """
    Given the csv file generated by the portal for a specific query, downloads and stores the documents mentioned within.
    Documents are stored as html files.
    :param original_metadata_file_path: The path to the generated csv file.
    :param documents_directory: The path to which the documents will be stored.
    :return:
    """
    if os.path.exists(documents_directory):
        raise Exception('Documents directory not empty')
    os.mkdir(documents_directory)

    df = pd.read_csv(original_metadata_file_path, na_values='Not available')
    df = df.drop_duplicates()
    df = df.dropna(how='all', axis=1)
    df['CELEX number_clean'] = df['CELEX number'].str.replace(r'\W', '')
    assert len(df['CELEX number_clean'].tolist()) == len(set(df['CELEX number_clean'].tolist())), 'During conversion of CELEX codes, duplicates have been created'
    df.to_csv(original_metadata_file_path, index=False)

    request_dict = {
        'NAT_CASE_LAW': Template('https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=URISERV:${celex}'),
        'EU_CASE_LAW': Template('https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:${celex}'),
        'CONSLEG': Template('https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:${celex}')
    }
    for typee, celex, celex_clean in zip(df['document_type'], df['CELEX number'], df['CELEX number_clean']):
        file_pathh = os.path.join(documents_directory, f'{celex_clean}.html')
        if not os.path.exists(file_pathh):
            urllib.request.urlretrieve(request_dict[typee].substitute({'celex': celex}), file_pathh)
            time.sleep(5)


def get_individual_documents(CELEX_uri_list: [str, ...], documents_directory: str):
    """
    Given a list of CELEX URIs, downloads and stores the documents.
    :param CELEX_uri_list: The list of the CELEX URIs.
    :param documents_directory: The directory where the documents will be stored.
    :return:
    """
    if not os.path.exists(documents_directory):
        os.mkdir(documents_directory)

    for CELEX_uri in CELEX_uri_list:
        urllib.request.urlretrieve(
            CELEX_uri,
            os.path.join(
                documents_directory,
                re.sub(r'\W', '', re.findall(r'CELEX:(.*)', CELEX_uri)[0]) + '.' + re.findall(r'TXT\/(.*)\/', CELEX_uri_list[0])[0].lower()
            )
        )
        time.sleep(5)
