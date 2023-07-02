import os

kb = """
Querying the index or a graph involves a three main components:

| Retreivers $\to$                                                        | Response Synthesizer $\to$                                                  | Query Engine                                                                                                                              |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| A retriever class retrieves a set of Nodes from an index given a query. | This class takes in a set of Nodes and synthesizes an answer given a query. | This class takes in a query and returns a Response object. It can make use of Retrievers and Response Synthesizer modules under the hood. |

For the query logic itself we will use maximum marginal relevance or $\mathcal{MMR}$. In this we iteratively find documents that are dissimilar to previous results. It has been shown to improve performance for LLM retrievals [[2]](https://arxiv.org/pdf/2211.13892.pdf).
"""

indices = [
    {
        "path": "storage/vedvyas_Index/",
        "index_id": "VedVyas",
        "summary": os.getenv("VYAS_SUMMARY") or "",
    },
    {
        "path": "storage/tulsidas_Index/",
        "index_id": "Tulsidas",
        "summary": os.getenv("TULSI_SUMMARY") or "",
    },
    {
        "path": "storage/valmiki_Index/",
        "index_id": "ShriValmiki",
        "summary": os.getenv("VALMIKI_SUMMARY") or "",
    },
    {
        "path": "storage/vedas_Index/",
        "index_id": "vedas",
        "summary": os.getenv("VEDAS_SUMMARY") or "",
    },
    {
        "path": "storage/mb_Index/",
        "index_id": "mb",
        "summary": os.getenv("MB_SUMMARY") or "",
    },
    {
        "path": "storage/puranas_Index/",
        "index_id": "puranas",
        "summary": os.getenv("PURANAS_SUMMARY") or "",
    },
    {
        "path": "storage/geeta_Index/",
        "index_id": "geeta",
        "summary": os.getenv("GEETA_SUMMARY") or "",
    },
]
