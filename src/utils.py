import tiktoken
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def get_token_count(string: str, llm_name: str) -> int:
    """
    Returns the number of tokens in a given string for a specified language model.
    :param string: The input text to be tokenized.
    :param llm_name: The name of the language model to determine the encoding.
    :return: The count of tokens in the input string.
    """
    encoding = tiktoken.encoding_for_model(llm_name)
    return len(encoding.encode(string))


def chatqa(query: str, llm_name: str) -> str:
    """
    Basic OpenAI inference through langchain.
    :param query: The user's query.
    :param llm_name: The name of the OpenAI model to be used.
    :return: The model's output
    """

    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    chain = LLMChain(llm=llm, prompt=PromptTemplate(template="""{q}""", input_variables=["q"]))
    res = chain.apply([{'q': query}])[0]['text']

    return res


def merge_dicts(*dicts: dict) -> dict:
    _ = {}
    for dictt in dicts:
        _.update(dictt)
    return _
