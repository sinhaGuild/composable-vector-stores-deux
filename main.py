import datetime
import os

import streamlit as st
from dotenv import load_dotenv

from composable_graphs import ComposableGraphs
from indices import indices, kb

load_dotenv()


@st.cache_resource
def compose_graph():
    return ComposableGraphs(indices_path=indices)


# def inference():
#     return compose_graph()


st.set_page_config(layout="wide")
st.markdown(
    "## $\mathbb{C}$omposable $\mathbb{G}$raphs with $\mathbb{V}$ector indices for $\mathcal{LLM}$ queries."
)
if "history" not in st.session_state:
    st.session_state["history"] = []
    st.session_state["sources"] = []

st.sidebar.image("https://i.imgur.com/4D2ikLS.png", clamp=True, width=150)
st.sidebar.markdown("# $\mathbb{Parameters}$")
# -- Select Llama index or langchain
st.sidebar.markdown("### $\mathbf{Retreiver}$")
select_retreiver = st.sidebar.selectbox(
    "Python Retreival Augmentation Library",
    ["MMR", "LlamaIndex", "Langchain", "SQQ"],
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
    disabled=select_retreiver == "MMR" or select_retreiver == "SQQ",
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
                        re = compose_graph()._inference_LANGCHAIN_(
                            prompt=text, graph_index_type=select_graph_index
                        )
                    st.success(re)
                    st.session_state["history"].append(re)
                    st.session_state["sources"].append("Langchain.")

                else:
                    if select_retreiver == "LlamaIndex":
                        with st.spinner("Querying Composable Graph with Llamaindex"):
                            # graph = inference()
                            re = compose_graph()._inference_LLAMA_(
                                prompt=text, graph_index_type=select_graph_index
                            )
                        # st.success(re)
                        # st.session_state["history"].append(re)
                        # st.divider()
                        # with st.expander("Retreived Sources"):
                        #     st.info(re.get_formatted_sources())
                    elif select_retreiver == "SQQ":
                        with st.spinner("Querying Graph with Sub-Question-Query."):
                            # graph = inference()
                            re = compose_graph()._inference_SQQ_(prompt=text)
                        # st.success(re)
                        # st.session_state["history"].append(re)
                        # with st.expander("Retreived Sources"):
                        #     st.info(re.get_formatted_sources())

                    else:
                        with st.spinner(
                            f"Querying Index {indices[mmr_index]['index_id']} using MMR"
                        ):
                            re = compose_graph()._inference_MMR_(
                                prompt=text, mmr_index=mmr_index
                            )
                        # st.success(re)
                        # st.session_state["history"].append(re)
                        # st.divider()
                        # with st.expander("Retreived Sources"):
                        #     st.info(re.get_formatted_sources())
                    st.success(re)
                    st.session_state["history"].append(re)
                    st.session_state["sources"].append(re.get_formatted_sources())
                    st.divider()
                    with st.expander("Retreived Sources"):
                        st.info(re.get_formatted_sources())

        st.divider()
        with st.container():
            hist = ""
            st.markdown("### $\mathbb{HISTORY}$")
            st.divider()
            for idx, i in enumerate(st.session_state.history):
                with st.container():
                    st.markdown(f"- Inference @ `{datetime.datetime.now()}`")
                    st.info(i)
                    with st.expander("Retreived Sources"):
                        st.markdown(f"{st.session_state.sources[idx]}")
