import datetime
import os

import streamlit as st
from dotenv import load_dotenv

from composable_graphs import ComposableGraphs

load_dotenv()

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


@st.cache_resource
def compose_graph():
    return ComposableGraphs(indices_path=indices)


def inference(
    prompt,
    graph_index_type="List",
    return_sources=True,
    use_langchain=False,
    use_mmr=False,
    mmr_index=0,
):
    graph = compose_graph()

    return graph.call_inference(
        prompt=prompt,
        return_sources=return_sources,
        use_langchain=use_langchain,
        use_mmr=use_mmr,
        graph_index_type=graph_index_type,
        mmr_index=mmr_index,
    )


st.set_page_config(layout="wide")
st.markdown(
    "## $\mathbb{C}$omposable $\mathbb{G}$raphs with $\mathbb{V}$ector indices for $\mathcal{LLM}$ queries."
)
if "history" not in st.session_state:
    st.session_state["history"] = []

st.sidebar.image("https://i.imgur.com/4D2ikLS.png", clamp=True, width=150)
st.sidebar.markdown("# $\mathbb{Parameters}$")
# -- Select Llama index or langchain
st.sidebar.markdown("### $\mathbf{Retreiver}$")
select_retreiver = st.sidebar.selectbox(
    "Python Retreival Augmentation Library",
    ["MMR", "LlamaIndex", "Langchain"],
)
mmr_index = st.sidebar.number_input(
    f"MMR Index {'(MMR Not Selected)' if select_retreiver!='MMR' else ''}",
    min_value=0,
    max_value=len(indices) - 1,
    step=1,
    value=0,
    disabled=select_retreiver != "MMR",
)

st.sidebar.divider()
st.sidebar.markdown("### $\mathbf{Graph \ Index}$")
st.sidebar.text(
    f"{'(Select Non-MMR retreiver to use graphs)' if select_retreiver=='MMR' else ''}"
)
select_graph_index = st.sidebar.selectbox(
    "Graph Index",
    ["List", "Tree", "Keyword"],
    disabled=select_retreiver == "MMR",
)
st.sidebar.divider()
st.sidebar.markdown("### $\mathbf{Info}$")
with st.sidebar.expander("ℹ What is this about ? "):
    st.markdown(kb)
    st.image("img/composable-graphs.png")

with st.container():
    with st.form("query_form"):
        st.markdown("### $\mathbb{PROMPT}$")
        text = st.text_area(
            "Enter any subject or topic query related to Indic History.",
            "What are the origins of manu?",
            label_visibility="hidden",
        )
        submitted = st.form_submit_button("Submit")

        if submitted:
            with st.container():
                st.divider()
                if select_retreiver == "Langchain":
                    with st.spinner("Querying Composable graph with Langchain."):
                        re = inference(
                            prompt=text,
                            return_sources=False,
                            use_langchain=True,
                            graph_index_type=select_graph_index,
                        )
                    st.success(re)
                    st.session_state["history"].append(re)

                elif select_retreiver == "LlamaIndex":
                    with st.spinner("Querying Composable Graph with Llamaindex"):
                        re = inference(prompt=text, graph_index_type=select_graph_index)

                    st.success(re)
                    st.session_state["history"].append(re)
                    st.divider()
                    with st.expander("Retreived Sources"):
                        st.info(re.get_formatted_sources())
                else:
                    # print(select_graph_index)
                    # print(select_retreiver)
                    with st.spinner(
                        f"Querying Index {indices[mmr_index]['index_id']} using MMR"
                    ):
                        re = inference(
                            prompt=text,
                            graph_index_type=select_graph_index,
                            use_mmr=True,
                            mmr_index=mmr_index,
                        )
                    st.success(re)
                    st.session_state["history"].append(re)
                    st.divider()
                    with st.expander("Retreived Sources"):
                        st.info(re.get_formatted_sources())

        st.divider()
        with st.container():
            hist = ""
            st.markdown("### $\mathbb{HISTORY}$")
            st.divider()
            for i in reversed(st.session_state.history):
                with st.container():
                    st.markdown(f"- Inference @ `{datetime.datetime.now()}`")
                    st.info(i)
