import logging
import os
import sys

import openai
from dotenv import load_dotenv
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
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.tools import QueryEngineTool, ToolMetadata
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
        self.query_engine_tools = []

        # indexes
        for idx in indices_path:
            index = self._rebuild_from_storage_(idx.get("path"), idx.get("index_id"))
            self.indices.append(index)
            self.index_summaries.append(idx.get("summary"))
            self.custom_query_engines[index.index_id] = index.as_query_engine(
                similarity_top_k=3,
                response_mode="tree_summarize",
            )
            self.query_engine_tools.append(
                QueryEngineTool(
                    query_engine=self.custom_query_engines[index.index_id],
                    metadata=ToolMetadata(
                        name=f"{idx.get('index_id')}",
                        description=f"{idx.get('summary')}",
                    ),
                )
            )

        # compose sub-question query engine
        self.s_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=self.query_engine_tools
        )

        # compose graphs
        self.graph_list_parent = self._build_composable_graph_(ListIndex)
        self.graph_tree_parent = self._build_composable_graph_(TreeIndex)
        self.graph_keyword_parent = self._build_composable_graph_(
            SimpleKeywordTableIndex
        )

        self.pg = Playground(indices=self.indices)

    def create_inference_by_graph_type(self, prompt, graph_index_type):
        if graph_index_type == "Tree":
            return InferenceLlamaIndex(
                prompt=prompt,
                graph=self.graph_tree_parent,
            )
        elif graph_index_type == "Keyword":
            return InferenceLlamaIndex(
                prompt=prompt,
                graph=self.graph_keyword_parent,
            )
        else:
            return InferenceLlamaIndex(
                prompt=prompt,
                graph=self.graph_list_parent,
            )

    def _inference_LLAMA_(self, prompt, graph_index_type):
        return self.create_inference_by_graph_type(
            prompt=prompt, graph_index_type=graph_index_type
        )._inference_(self.custom_query_engines)

    def _inference_LANGCHAIN_(self, prompt, graph_index_type):
        return self.create_inference_by_graph_type(
            prompt=prompt, graph_index_type=graph_index_type
        )._inference_langchain_(self.custom_query_engines)

    def _inference_MMR_(self, prompt, mmr_index):
        inf = InferenceLlamaIndex(
            prompt=prompt,
            graph=self.graph_list_parent,
        )

        return inf._inference_MMR_(self.indices[mmr_index])

    def _inference_SQQ_(self, prompt):
        inf = InferenceLlamaIndex(
            prompt=prompt,
            graph=self.graph_list_parent,
        )

        return inf._inference_SQQ_(self.s_engine)

    def playground(self, prompt):
        re = self.pg.compare(prompt, to_pandas=True)
        self._md_(f"### Output: {re['Output']}")
        self._md_(f"> {re}")
        return re
