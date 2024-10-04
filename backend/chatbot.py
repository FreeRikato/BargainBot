import os
from dotenv import load_dotenv
from typing import Annotated
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Load environment variables from the .env file
load_dotenv()

# Access the environment variable
groq_api_key = os.getenv("GROQ_API_KEY")


class State(TypedDict):
    messages: Annotated[list, add_messages]


# Build the graph
graph_builder = StateGraph(State)

# Use DuckDuckGo Search Results instead of TavilySearchResults
tool = DuckDuckGoSearchResults(max_results=10)
tools = [tool]
llm = ChatGroq(model="llama-3.1-70b-versatile")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")


class SqliteSaverManager:
    def __init__(self, conn_string):
        self.conn_string = conn_string
        self.saver = None

    def __enter__(self):
        self.saver = SqliteSaver.from_conn_string(self.conn_string)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.saver:
            self.saver.close()

    def flush_memory(self):
        db_path = self.saver.db_file
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM checkpoints")
            conn.commit()
        print("Memory flushed.")

    def get_saver(self):
        return self.saver


# Use SqliteSaverManager with a context manager
with SqliteSaverManager("../logs/chat_history.db") as memory_manager:
    # Compile the graph with SQLite memory saver
    graph = graph_builder.compile(checkpointer=memory_manager.get_saver())
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        pass

    # Function to stream graph updates
    def stream_graph_updates(user_input: str, thread_id="default"):
        config = {"configurable": {"thread_id": thread_id}}
        for event in graph.stream({"messages": [("user", user_input)]}, config=config):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)

    # Chat loop to interact with the bot
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            elif user_input.lower() == "flush memory":
                memory_manager.flush_memory()
                continue

            stream_graph_updates(user_input)
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    # Flush memory at the end of the conversation
    memory_manager.flush_memory()
    print("Memory flushed at the end of the conversation.")
