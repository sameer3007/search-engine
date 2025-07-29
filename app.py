import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()
#loading environment variables
os.environ['HUGGING_FACE_API_SE']=os.getenv("HUGGING_FACE_API_SE")
os.environ['GROQ_API_KEY_SE']=os.getenv("GROQ_API_KEY_SE")
os.environ['LANGCHAIN_API_KEY_SE']=os.getenv("LANGCHAIN_API_KEY_SE")
os.environ['LANGSMITH_TRACING_SE']=os.getenv("LANGSMITH_TRACING_SE")
os.environ['LANGSMITH_ENDPOINT_SE']=os.getenv("LANGSMITH_ENDPOINT_SE")
os.environ['LANGSMITH_API_KEY_SE']=os.getenv("LANGSMITH_API_KEY_SE")
os.environ['LANGSMITH_PROJECT_SE']=os.getenv("LANGSMITH_PROJECT_SE")

## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Search")


st.title("🔎 LangChain - search engine")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""


# --- 💬 Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a chatbot that can search the web. How can I help you?"}
    ]

# Display all previous messages in the chat interface
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- ✍️ Handle New User Input ---
if prompt := st.chat_input(placeholder="Ask me anything..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize the Chat Model (Groq with Llama3)
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY_SE"),
        model_name="Llama3-8b-8192",
        streaming=True
    )

    # List of tools the agent can use
    tools = [search, arxiv, wiki]

    # Create a LangChain agent with tool access
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    # --- 🤖 Generate and Display Assistant's Response ---
    with st.chat_message("assistant"):
        # Set up a Streamlit callback to show real-time actions
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # Run the agent on the chat history (entire message thread)
        response = agent.run(st.session_state.messages, callbacks=[st_cb])

        # Save assistant's reply to the session
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)