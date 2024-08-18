from string import Template

infer_new_columns_prompt = Template("""
You have a collection of European Union legal documents and a pandas dataframe containing metadata about these documents.
The pandas dataframe contains these columns: ${columns}.
Some questions cannot be answered fully just by using the metadata dataframe.

Given this question:
"${question}"
, your task is to detect the data (columns) missing from the dataframe that would make answering this question possible, solely with the metadata dataframe.

Which columns are missing from the metadata dataframe?
Answer ONLY with the columns' Python data types and the questions that must be posed to each individual document for the creation of those columns.
Available datatypes are: 'str' for string, 'int' for integer, 'float' for floating point numbers, and 'bool' for boolean.
The format specified below must always be returned and nothing else:

["question_that_should_be_posed_for_column1(data type)",
"question_that_should_be_posed_for_column2(data type)"]

The questions must be formulated as if they are targeted for one and only one document.
The column names must be completely informative of the content.
""")

remove_duplicate_new_cols_prompt = Template("""
Given this list of questions:

${dictt}

, remove entries that would be considered duplicates. Return only the resulting list and nothing else.
""")

extract_data_prompt = Template("""
Context: ${context}

Considering the document provided above, and the following list, where its values are questions posed to the document:

${suggested_column_list}

Create a new dictionary that contains the list items as keys while the values are the corresponding answered questions for the document. 
Never change the list items. Never add or remove items to the list.
Respect the Python data type of each column, that is provided inside the parentheses, making sure to create outputs parsable by Python. Never return values that are lists themselves.
In case of open ended questions be as concise as possible.
In case that the answer is null and only then, input None, however, using None should be avoided as much as possible.

Reply with the dictionary and nothing else.
""")

extract_data_prompt_continuous = Template("""
Context: ${context}

Considering the document provided above, and the following list, where its values are questions posed to the document:

${suggested_column_list}

Create a new dictionary that contains the list items as keys while the values are the corresponding answered questions for the document. 
Never change the list items. Never add or remove items to the list.
Respect the Python data type of each column, that is provided inside the parentheses, making sure to create outputs parsable by Python. Never return values that are lists themselves.
In case of open ended questions be as concise as possible.
In case that the answer is null and only then, input None, however, using None should be avoided as much as possible.

This document chunk provided as context, is a continuation of a document that already received answers noted below:

${answers}

By taking all information under account, update, modify or replace the values in the previous dictionary according to the added context, however, the dictionary keys must remain the same.

Reply with the new dictionary and nothing else.
""")

refine_answers_prompt = Template("""
${raw_answers}

Given the above dictionary, that follows the structure of {'id': {'column_name(column_data_type)': 'column_value'}}, homogenize the column_values,
respecting the data type of the column provided.

Some rules for the homogenization include but are not limited to: 
Values must not be lists. Dates must be in the format of dd/mm/YYYY. Boolean values must be either True or False, case sensitive.

Reply only with the homogenized dictionary.

""")
