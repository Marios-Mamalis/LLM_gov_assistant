import pandas as pd
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
metadata_file_path = 'data/metadata.csv'
merged_metadata_file_path = 'data/metadata_fa.csv'
aq_documents_directory = 'data/aq_documents'
rag_documents_directory = 'data/rag_documents'

# retrieve data
rd.get_documents_from_csv(original_metadata_file_path=metadata_file_path, documents_directory=aq_documents_directory)
print(pd.read_csv(metadata_file_path).groupby('Form').agg({'Form': 'count'}))
rd.get_individual_documents(CELEX_uri_list=['https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679'], documents_directory=rag_documents_directory)


rag_responses = rag.RAG(llm_name='gpt-3.5-turbo-0125', chroma_directory=chroma_dir,
                        pdf_paths=[os.path.join(rag_documents_directory, i) for i in os.listdir(rag_documents_directory) if i.endswith('pdf')],
                        embedding_model_name='text-embedding-3-large', questions=rag_questions)

metadata = pd.read_csv(metadata_file_path)
metadata = metadata.drop(['ELI', 'CELEX number', 'ECLI identifier', 'Number of pages', 'document_type', 'Publication Reference'], axis=1)
aq_responses = aqs.AQS(aq_questions=query_questions, processed_metadata_dataframe=metadata, information_inference_llm_name='gpt-4-0125-preview',
                       information_extractor_llm_name='gpt-3.5-turbo-0125', information_extractor_llm_context_size=16000, aq_documents_dir=aq_documents_directory,
                       refiner_llm_name='gpt-3.5-turbo-0125', refiner_llm_output_size=4000,
                       merged_metadata_save_file_path=merged_metadata_file_path, agent_llm_name='gpt-4-0125-preview')

print(rag_responses)
print(aq_responses)
