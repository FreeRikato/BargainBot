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

# Use SqliteSaver with a context manager
with SqliteSaver.from_conn_string("../logs/chat_history.db") as memory:
    # Compile the graph with SQLite memory saver
    graph = graph_builder.compile(checkpointer=memory)

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

            stream_graph_updates(user_input)
        except:
            # Fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break
