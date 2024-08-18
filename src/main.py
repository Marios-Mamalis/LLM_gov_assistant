import pandas as pd
from langchain_experimental.agents import create_csv_agent
from langchain_community.chat_models import ChatOpenAI
import numpy as np
import os

import src.retrieve_data as rd
import src.rag_system as rag
import src.agent_query_system as aqs


rag_questions = [
    'What are the primary objective and the EU regulatory framework of the General Data Protection Regulation (GDPR)?',
    'How does the GDPR define "consent" in the context of data processing, and what are the requirements for obtaining valid consent?',
    'Describe the role, responsibilities, and qualifications of a Data Protection Officer (DPO) under the GDPR.',
    'Specify the framework for the protection of childs’ rights under GDPR and provide the relevant provisions.'
]
query_questions = [
    "How many requests for preliminary rulings have been submitted to the Court during the period of reference?",
    "For the requests for preliminary rulings, for which a court ruling has been issued, calculate the amount of time in days between date of submission and date of issuance, and present the average time for Court response.",
    "For the entire number of preliminary rulings to the EU Court, identify the cases where a court ruling has been issued.",
    "For the period from 2015-2024, identify and classify the requests for preliminary rulings to the EU Court using the following criteria: a. member state courts, b. number of requests per year.",
    "For the requests for preliminary rulings submitted to the Court by Member State courts, identify overlapping areas as regards the interpretation of specific articles of GDPR, and present the most frequently addressed articles. Classify the findings by article GDPR.",
    "From the requests for preliminary rulings submitted to the Court by Member State courts, derive the key principles that govern the processing of personal data.",
    "Taking into account the GDPR and the submitted requests for preliminary rulings, identify potential risks factors regarding data protection rights.",
    "What is the relation between the EU Court of Justice and the national courts as regards the interpretation of GDPR provisions."
]
chroma_dir = 'data/chroma'
metadata_file_path = 'data/METADATA.csv'
merged_metadata_file_path = 'data/METADATA_fa.csv'
aq_documents_directory = 'data/aq_documents'
rag_documents_directory = 'data/rag_documents'

# retrieve data
rd.get_documents_from_csv(original_metadata_file_path=metadata_file_path, documents_directory=aq_documents_directory)
print(pd.read_csv(metadata_file_path).groupby('Form').agg({'Form': 'count'}))
rd.get_individual_documents(CELEX_uri_list=['https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679'], documents_directory=rag_documents_directory)

# rag inference
if not os.path.exists(chroma_dir):
    rag.load_to_chroma(
        pdf_paths=[i for i in os.listdir(rag_documents_directory) if i.endswith('pdf')],
        chroma_dir=chroma_dir,
        llm_name='gpt-3.5-turbo-0125',
        embedding_model_name='text-embedding-3-large'
    )

for rag_question in rag_questions:
    print(rag.RAG(question=rag_question, chroma_dir=chroma_dir, llm_name='gpt-3.5-turbo-0125', embedding_model_name='text-embedding-3-large'))

# agentic query inference
if not os.path.exists(merged_metadata_file_path):
    metadata = pd.read_csv(metadata_file_path)
    metadata_for_agent = metadata.drop(
        ['ELI', 'CELEX number', 'ECLI identifier', 'Number of pages', 'document_type', 'Publication Reference'], axis=1)

    new_columns = aqs.infer_needed_information(
        query_questions,
        existing_columns=metadata_for_agent.columns.tolist(),
        llm_name='gpt-4-0125-preview'
    )

    raw_answers = aqs.extract_necessary_information(
        information_to_be_extracted=str(new_columns),
        llm_name='gpt-3.5-turbo-0125',
        model_context_size=16000,
        document_names=[i for i in metadata_for_agent['CELEX number_clean'].tolist()],
        document_dir=aq_documents_directory
    )

    # save merged metadata (original + extracted)
    refined_answers = aqs.refine_extracted_information(raw_answers=raw_answers, llm_name='gpt-3.5-turbo-0125', llm_output_size=4000)
    extracted_information_dataframe = pd.DataFrame(refined_answers).T
    extracted_information_dataframe = extracted_information_dataframe.replace('', np.NaN)
    extracted_information_dataframe = extracted_information_dataframe.reset_index().rename(columns={'index': 'CELEX number_clean'})
    metadata_for_agent = metadata_for_agent.merge(extracted_information_dataframe, on='CELEX number_clean', how='outer')
    metadata_for_agent.drop('CELEX number_clean', axis=1, inplace=True)
    metadata_for_agent.to_csv(merged_metadata_file_path, index=False)

agent = create_csv_agent(
    ChatOpenAI(model_name='gpt-4-0125-preview', temperature=0),
    merged_metadata_file_path,
    verbose=False
)

for query_question in query_questions:
    print(agent.invoke(query_question))
