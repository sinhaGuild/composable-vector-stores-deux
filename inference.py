from langchain.agents import Tool, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from rich.console import Console
from rich.markdown import Markdown


class InferenceLlamaIndex:
    def __init__(
        self,
        graph,
        prompt,
        return_sources,
        memory_key="chat_history",
        temperature=0,
        max_tokens=500,
        model="gpt-3.5-turbo-16k",
    ) -> None:
        self.console = Console()
        self.memory = ConversationBufferMemory(memory_key=memory_key)
        self.llm = ChatOpenAI(
            temperature=temperature, max_tokens=max_tokens, model=model
        )
        self.graph = graph
        self.prompt = prompt
        self.return_sources = return_sources

    def _set_graph_(self, graph):
        self.graph = graph

    # display markdown
    def _md_(self, re, return_sources=False):
        self.console.print(
            Markdown(f"Sources: <br> {re.get_formatted_sources()} <br>  ")
        ) if return_sources else print("")

        self.console.print(Markdown((f"Full Response: <br> {re} <br>  ")))

    # Inference through vector indices
    def _inference_(self, custom_query_engines):
        qe = self.graph.as_query_engine(custom_query_engines=custom_query_engines)
        re = qe.query(self.prompt)
        self._md_(re, self.return_sources)
        return re

    # Inference through vector indices
    def _inference_MMR_(self, index):
        qe = index.as_query_engine(
            vector_store_query_mode="mmr",
            vector_store_kwargs={"mmr_threshold": 0.8},
        )
        print(f"{self.prompt=}")
        re = qe.query(self.prompt)
        self._md_(re, self.return_sources)
        return re

    # Inference through langchain
    def _inference_langchain_(self):
        qe = self.graph.as_query_engine(custom_query_engines=self.custom_query_engines)

        tool = [
            Tool(
                name="LlamaIndex",
                func=lambda q: str(qe.query(q)),
                description="useful for when you want to answer questions. The input to this tool should be a complete english sentence.",
                return_direct=True,
            ),
        ]
        agent = initialize_agent(
            tools=tool,
            llm=self.llm,
            agent="conversational-react-description",
            memory=self.memory,
        )
        re = agent.run(self.prompt)
        self._md_(re)
        return re
