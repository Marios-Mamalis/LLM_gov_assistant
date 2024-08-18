# -*- coding: utf-8 -*-
import itertools
import ast
import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import CharacterTextSplitter

import src.prompts as prompts
import src.utils as utils


load_dotenv(find_dotenv())


def infer_needed_information(questions: [str, ...], existing_columns: str, llm_name: str) -> [str, ...]:
    """
    Given a batch of questions and existing information about documents, infers what information about the documents
    is missing in order to accurately answer the given questions. At every step, the function ensures that the model
    output can be evaluated as a Python object.
    :param questions: The list of questions.
    :param existing_columns: What information is currently available.
    :param llm_name: The name of the LLM model to be used to perform the inference.
    :return:
    """

    new_necessary_columns = []
    for question in questions:
        new_necessary_columns.append(ast.literal_eval(
            utils.chatqa(
                prompts.infer_new_columns_prompt.substitute(
                    question=question,
                    columns=str([i for i in existing_columns if i != 'CELEX number_clean']),
                ),
                llm_name=llm_name
            )
        ))

    # filter duplicates
    new_necessary_columns = ast.literal_eval(utils.chatqa(
        prompts.remove_duplicate_new_cols_prompt.substitute(dictt=str(list(itertools.chain(*new_necessary_columns)))),
        llm_name=llm_name
    ))

    return new_necessary_columns


def extract_necessary_information(information_to_be_extracted: str, document_names: [str, ...], document_dir: str, llm_name: str, model_context_size: int) -> str:
    """
    Extracts requested information from a list of documents in a batched manner.
    :param information_to_be_extracted: Information to be extracted in a list format.
    :param document_names: The document names. (assumes that the corresponding files are in html format).
    :param document_dir: The directory containing the documents.
    :param llm_name: The name of the LLM to be used for inference.
    :param model_context_size: The context size of the LLM to be used.
    :return: The extracted information as a string in the format of {document_name: {info1: extracted_info1, ...}, ...}
    """
    # TODO break down

    document_size = model_context_size * 0.95 - utils.get_token_count(information_to_be_extracted, llm_name=llm_name) * 2
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=document_size, chunk_overlap=0, model_name=llm_name, separator="\n",
    )

    raw_answers = ''
    for i in document_names:
        with open(os.path.join(document_dir, f'{i}.html'), 'r', encoding="utf8") as f:
            text = BeautifulSoup(f.read(), 'html.parser').get_text()
            if utils.get_token_count(text, llm_name=llm_name) >= document_size:
                texts = text_splitter.split_text(text)
                for ind, j in enumerate(texts):
                    if ind == 0:
                        output2 = utils.chatqa(
                            prompts.extract_data_prompt.substitute(context=j, suggested_column_list=information_to_be_extracted),
                            llm_name=llm_name
                        )
                    else:
                        output2 = utils.chatqa(
                            prompts.extract_data_prompt_continuous.substitute(context=j, suggested_column_list=information_to_be_extracted, answers=output2),
                            llm_name=llm_name
                        )
                raw_answers += f"'{i}': {output2},\n"
            else:
                output2 = utils.chatqa(
                    prompts.extract_data_prompt.substitute(context=text, suggested_column_list=information_to_be_extracted),
                    llm_name=llm_name
                )
                raw_answers += f"'{i}': {output2},\n"

    return raw_answers


def refine_extracted_information(raw_answers: str, llm_name: str, llm_output_size: int) -> dict:
    """
    Refines the extracted information derived from `extract_necessary_information`, to ensure homogeinity, in a batched
    manner.
    :param raw_answers: The answers that result from calling `extract_necessary_information`.
    :param llm_name: The LLM to perform the refinement.
    :param llm_output_size: The LLM's specified max output token count.
    :return: The refined extracted information as a dictionary.
    """

    raw_answers = raw_answers.split('},')
    refined_answers = []
    s = 0
    raw_answers_subset = []
    for i in raw_answers:
        s += utils.get_token_count(i, llm_name=llm_name)
        if s < llm_output_size * 0.9:
            raw_answers_subset.append(i)
        else:
            refined_answers.append(
                utils.chatqa(
                    prompts.refine_answers_prompt.substitute(raw_answers='{' + '}, '.join(raw_answers_subset) + '},}'),
                    llm_name=llm_name)
            )

            s = utils.get_token_count(i, llm_name=llm_name)
            raw_answers_subset = [i]

    refined_answers = [ast.literal_eval(i) for i in refined_answers]
    refined_answers = utils.merge_dicts(*refined_answers)

    return refined_answers
