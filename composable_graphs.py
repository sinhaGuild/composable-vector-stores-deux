import logging
import os
import sys

import openai
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from llama_index import (
    SimpleKeywordTableIndex,
    StorageContext,
    TreeIndex,
    load_index_from_storage,
)
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.list.base import ListIndex
from llama_index.playground import Playground
from rich.console import Console
from rich.markdown import Markdown

from inference import InferenceLlamaIndex

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "OPENAI_API_KEY"

# Set environment variables and logging config.
openai.api_key = OPENAI_API_KEY
console = Console()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
memory = ConversationBufferMemory(memory_key="chat_history")
llm = ChatOpenAI(temperature=0, max_tokens=500, model="gpt-3.5-turbo-16k")
# llm = ChatOpenAI(temperature=0, callbacks=[StreamingStdOutCallbackHandler()], streaming=True)


class ComposableGraphs:
    """
    Composable Graphs
    """

    # Rebuild indices
    def _rebuild_from_storage_(self, path, index_id):
        storage_context = StorageContext.from_defaults(persist_dir=path)
        return load_index_from_storage(storage_context, index_id=index_id)

    # Compose graphs
    def _build_composable_graph_(self, parent_index):
        return ComposableGraph.from_indices(
            parent_index,
            self.indices,
            self.index_summaries,
            storage_context=StorageContext.from_defaults(),
        )

    # initialize
    def __init__(self, indices_path) -> None:
        self.indices = []
        self.custom_query_engines = {}
        self.index_summaries = []

        # indexes
        for idx in indices_path:
            index = self._rebuild_from_storage_(idx.get("path"), idx.get("index_id"))
            self.indices.append(index)
            self.index_summaries.append(idx.get("summary"))
            self.custom_query_engines[index.index_id] = index.as_query_engine(
                similarity_top_k=3,
                response_mode="tree_summarize",
            )

        # compose graphs
        self.graph_list_parent = self._build_composable_graph_(ListIndex)
        self.graph_tree_parent = self._build_composable_graph_(TreeIndex)
        self.graph_keyword_parent = self._build_composable_graph_(
            SimpleKeywordTableIndex
        )

        self.pg = Playground(indices=self.indices)

    # graphs and inference
    def call_inference(
        self,
        prompt,
        return_sources=False,
        use_langchain=False,
        use_mmr=False,
        graph_index_type="List",
        mmr_index=0,
    ):
        def call_inference_method(inf, use_langchain, use_mmr):
            if use_langchain:
                return inf._inference_langchain_()
            elif use_mmr:
                print(f"{inf.prompt=}")
                return inf._inference_MMR_(self.indices[mmr_index])
            else:
                return inf._inference_(self.custom_query_engines)

        if graph_index_type == "List":
            inf = InferenceLlamaIndex(
                prompt=prompt,
                return_sources=return_sources,
                graph=self.graph_list_parent,
            )
            return call_inference_method(inf, use_langchain, use_mmr)

        elif graph_index_type == "Tree":
            inf = InferenceLlamaIndex(
                prompt=prompt,
                return_sources=return_sources,
                graph=self.graph_tree_parent,
            )
            return call_inference_method(inf, use_langchain, use_mmr)

        elif graph_index_type == "Keyword":
            inf = InferenceLlamaIndex(
                prompt=prompt,
                return_sources=return_sources,
                graph=self.graph_keyword_parent,
            )

            return call_inference_method(inf, use_langchain, use_mmr)

    def playground(self, prompt):
        re = self.pg.compare(prompt, to_pandas=True)
        self._md_(f"### Output: {re['Output']}")
        self._md_(f"> {re}")
