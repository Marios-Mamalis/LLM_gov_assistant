# An LLM agent-based governace assistant
Official implementation for the systems presented in "[A Large Language Model Agent Based Legal Assistant for Governance Applications](https://doi.org/10.1007/978-3-031-70274-7_18)".

## Overview
This repository includes two systems: a basic retrieval-augmented generation pipeline for answering questions based on an external corpus, and an agent-based system specialized in answering questions about aggregated metrics across multiple documents.  

RAG is implemented in a standard manner by using the initial query's embeddings to retrieve relevant documents for inclusion in the question prompt. The agent-based subsystem operates in three steps: first, it defines the information to be extracted from each document based on the user's query. Next, it extracts this information from each document, homogenizes the results, and stores them in a structured format. Finally, a Python agent is used to answer the user's queries based on the structured data. The system also supports integrating existing metadata alongside the extracted, structured data, and is optimized for batched inference. Both systems utilize OpenAI models for text and embeddings generation.

The case presented in the paper is contained in `main.py`. 

## Installation
Requires Python 3.9
Install dependencies with:
```
pip install -r requirements.txt
```

## Citation
If you use the code, please cite the corresponding paper:  
```
@inproceedings{mamalis2024large,
  title={A Large Language Model Agent Based Legal Assistant for Governance Applications},
  author={Mamalis, Marios Evangelos and Kalampokis, Evangelos and Fitsilis, Fotios and Theodorakopoulos, Georgios and Tarabanis, Konstantinos},
  booktitle={International Conference on Electronic Government},
  pages={286--301},
  year={2024},
  organization={Springer}
}
```
