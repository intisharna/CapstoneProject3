import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage
from qdrant_client import models
from difflib import get_close_matches

st.set_page_config(
    page_title="FilmFinder Movie Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

IS_DEPLOYED = os.environ.get('HOSTNAME') == 'streamlit' or os.environ.get('USER') == 'streamlit'

QDRANT_URL = ""
QDRANT_API_KEY = ""
OPENAI_API_KEY = ""

if IS_DEPLOYED:
    QDRANT_URL = st.secrets.get("QDRANT_URL")
    QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY")
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
else:
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY or not OPENAI_API_KEY:
    st.error("🚨 Missing essential API keys or URL! Check .env file (local) or Streamlit secrets (deployed).")
    st.stop()

@st.cache_resource(show_spinner="Initializing AI models...")
def initialize_models():
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            temperature=0.2
        )
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )
        return llm, embeddings
    except Exception as e:
        st.error(f"🚨 Error initializing OpenAI models: {e}")
        st.stop()

llm, embeddings = initialize_models()

collection_name = "imdb_movies"

@st.cache_resource(show_spinner="Connecting to Movie Database...")
def initialize_qdrant_connection(_embeddings, _collection_name, _url, _api_key):
    try:
        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=_embeddings,
            collection_name=_collection_name,
            url=_url,
            api_key=_api_key
        )
        return qdrant
    except Exception as e:
        st.error(f"🚨 Error connecting to Qdrant collection '{_collection_name}': {e}")
        st.stop()

qdrant = initialize_qdrant_connection(embeddings, collection_name, QDRANT_URL, QDRANT_API_KEY)

movie_genres = sorted([
    "Action", "Adventure", "Sci-Fi", "Drama", "Comedy", "Thriller",
    "Romance", "Horror", "Mystery", "Fantasy", "Animation", "Family",
    "Crime", "Biography", "History", "War", "Music", "Musical", "Sport", "Western"
])

@tool
def search_movies(question: str):
    """Search for relevant movies based on a description, title, plot keywords, actors, or director."""
    try:
        results = qdrant.similarity_search(question, k=5)
        return results
    except Exception as e:
        return f"Error during movie search: {e}"

@tool
def search_movies_by_genre(question: str, genre: str):
    """
    Cari film berdasarkan deskripsi dan genre-nya.
    Selalu gunakan alat 'genre_list' terlebih dahulu untuk mengonfirmasi nama genre yang tepat sebelum menggunakan tools ini.
    """
    genre_lower = genre.lower()
    correct_genre = next((g for g in movie_genres if g.lower() == genre_lower), None)
    if not correct_genre:
        matches = get_close_matches(genre, movie_genres, n=1, cutoff=0.7)
        suggestion = f" Did you perhaps mean '{matches[0]}'?" if matches else ""
        return f"Error: Invalid genre '{genre}'. Please use the 'genre_list' tool to see valid options.{suggestion}"
    try:
        results = qdrant.similarity_search(
            question,
            k=5,
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.genre",
                        match=models.MatchValue(value=correct_genre),
                    ),
                ]
            ),
        )
        return results
    except Exception as e:
        return f"Error during genre search for '{correct_genre}': {e}"

@tool
def genre_list() -> list[str]:
    """Lihat daftar genre film yang tersedia untuk difilter."""
    return movie_genres

tools = [search_movies, search_movies_by_genre, genre_list]

def chat_movie_assistant(question, history_string):
    """Handles agent creation, invocation, and response processing for the movie assistant."""
    try:
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt= f'''You are FilmFinder, a knowledgeable and friendly movie assistant.
            Your goal is to help users find movies based on their descriptions, preferences (like genre, actors, director), or plot details.

            Available Tools:
            - search_movies: Use this for general searches based on keywords, plot, actors, etc. It returns a list of relevant movie documents.
            - genre_list: Use this FIRST if the user mentions a genre, to check the exact spelling and availability.
            - search_movies_by_genre: Use this ONLY AFTER confirming the genre exists with 'genre_list', to search within that specific genre. Provide both the search query and the exact genre name. It returns a list of relevant movie documents.

            Instructions:
            1. Understand the user's request.
            2. If a genre is mentioned, ALWAYS use 'genre_list' first to verify.
            3. Choose the most appropriate search tool ('search_movies' or 'search_movies_by_genre') based on the request and genre verification.
            4. Formulate a clear query for the chosen tool.
            5. Examine the list of Document objects returned by the tool. Extract relevant information (like title, year, overview from page_content or metadata) from the documents.
            6. Provide a concise and engaging summary to the user based on the extracted information. Mention 2-3 relevant movies found, perhaps using markdown bullet points.
            7. If the tool returns an empty list or an error, politely inform the user that no relevant movies were found or that there was an issue.
            8. Keep responses focused on the user's movie query.

            Use the following chat history for context if needed:
            {history_string}'''
        )

        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )

        answer = "Sorry, I encountered an issue and couldn't generate a response. Please try again."
        if result["messages"] and hasattr(result["messages"][-1], 'content'):
             last_message = result["messages"][-1]
             if hasattr(last_message, 'type') and last_message.type == 'ai':
                 answer = last_message.content

        total_input_tokens = 0
        total_output_tokens = 0
        for message in result["messages"]:
            usage = None
            if hasattr(message, 'response_metadata') and message.response_metadata:
                 usage = message.response_metadata.get("token_usage") or message.response_metadata.get("usage_metadata")
            if usage and isinstance(usage, dict):
                 total_input_tokens += usage.get("prompt_tokens", usage.get("input_tokens", 0))
                 total_output_tokens += usage.get("completion_tokens", usage.get("output_tokens", 0))

        price_idr = 17_000*(total_input_tokens*0.15 + total_output_tokens*0.6)/1_000_000

        tool_messages_content = []
        for message in result["messages"]:
            if isinstance(message, ToolMessage):
                tool_output_str = str(message.content)
                if len(tool_output_str) > 1000:
                    tool_output_str = tool_output_str[:1000] + "...(truncated)"
                tool_messages_content.append(f"🛠️ Tool Used: {message.name}\n💬 Output:\n{tool_output_str}")

        response_dict = {
            "answer": answer,
            "price_idr": price_idr,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "tool_messages": "\n---\n".join(tool_messages_content)
        }
        return response_dict

    except Exception as e:
        st.error(f"⚠️ An error occurred while processing your request: {e}")
        import traceback
        print(traceback.format_exc())
        return {
            "answer": "Sorry, I ran into an unexpected error. Please try asking differently.",
            "price_idr": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "tool_messages": f"Error: {e}"
        }

with st.sidebar:
    st.image("./img/FilmFinderIMG.png", use_container_width=True)
    st.header("🎬 FilmFinder Assistant")
    st.caption("Your friendly guide to the world of movies!")
    st.markdown("---")

    with st.expander("How I Work", expanded=False):
        st.markdown("""
            1. Ask me to find movies (ex: "Suggest a sci-fi movie about space").
            2. I use AI to understand your request.
            3. I search a database of top IMDb movies.
            4. I provide recommendations based on the search!
        """)

    with st.expander("Suggestion", expanded=False):
        st.markdown("""
            **Try asking about:**
            - Genres (ex: "comedy movies from the 90s")
            - Directors (ex: "movies directed by Christopher Nolan")
            - Actors (ex: "films starring Tom Hanks")
            - Plot details (ex: "a movie about time travel")
        """)

    st.markdown("---")

st.header("Chat with FilmFinder! 🍿")
st.markdown("Ask me anything about the top 1000 IMDb movies.")

container = st.container(height=500, border=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

with container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Suggest a movie!", key="movie_input"):
    messages_history = st.session_state.get("messages", [])[-10:]
    history_string = "\n".join([f'{msg["role"].capitalize()}: {msg["content"]}' for msg in messages_history]) or "No previous conversation."

    with container:
        with st.chat_message("user"):
            st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with container:
        with st.chat_message("ai"):
            with st.spinner("FilmFinder is searching.."):
                response_data = chat_movie_assistant(prompt, history_string)
                answer = response_data["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "ai", "content": answer})

    st.divider()
    with st.expander("Show Debugging Info", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("🛠️ Tool Calls")
            if response_data["tool_messages"]:
                st.text_area(
                    "Tool Interactions:",
                    value=response_data["tool_messages"],
                    height=200,
                    disabled=True
                )
            else:
                st.info("No tools were called.")

        with col2:
            st.subheader("📜 History Context")
            st.text_area(
                "History Sent to Agent:",
                value=history_string,
                height=200,
                disabled=True
            )

        with col3:
            st.subheader("📊 Usage (Est.)")
            st.metric(label="Input Tokens", value=response_data["total_input_tokens"])
            st.metric(label="Output Tokens", value=response_data["total_output_tokens"])
            st.metric(label="Estimated Cost (Rp)", value=f"Rp {response_data['price_idr']:.2f}")