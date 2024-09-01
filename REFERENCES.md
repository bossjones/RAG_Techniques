### RAG evaluation summary[](https://docs.smith.langchain.com/concepts/evaluation#rag-evaluation-summary)

| Use Case            | Detail                                            | Reference-free? | LLM-as-judge?                                                | Pairwise relevant |
| ------------------- | ------------------------------------------------- | --------------- | ------------------------------------------------------------ | ----------------- |
| Document relevance  | Are documents relevant to the question?           | Yes             | Yes - [prompt](https://smith.langchain.com/hub/langchain-ai/rag-document-relevance) | No                |
| Answer faithfulness | Is the answer grounded in the documents?          | Yes             | Yes - [prompt](https://smith.langchain.com/hub/langchain-ai/rag-answer-hallucination) | No                |
| Answer helpfulness  | Does the answer help address the question?        | Yes             | Yes - [prompt](https://smith.langchain.com/hub/langchain-ai/rag-answer-helpfulness) | No                |
| Answer correctness  | Is the answer consistent with a reference answer? | No              | Yes - [prompt](https://smith.langchain.com/hub/langchain-ai/rag-answer-vs-reference) | No                |
| Chain comparison    | How do multiple answer versions compare?          | Yes             | Yes - [prompt](https://smith.langchain.com/hub/langchain-ai/pairwise-evaluation-rag) | Yes               |
