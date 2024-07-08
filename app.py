import streamlit as st
import os
from llama_index.core import (
    Document,
    SummaryIndex,
    load_index_from_storage,
    # TODO update this in docs
    VectorStoreIndex,
    StorageContext,
)
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings


from llama_index.core import (
    PromptTemplate,
    SelectorPromptTemplate,
    ChatPromptTemplate,
    SimpleDirectoryReader,
)
from llama_index.core.prompts.utils import is_chat_model
from llama_index.core.llms import ChatMessage, MessageRole

from PIL import Image
from llama_index.readers.file import ImageReader


# Text QA templates
DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information answer the following question "
    "(if you don't know the answer, use the best of your knowledge): {query_str}\n"
)
TEXT_QA_TEMPLATE = PromptTemplate(DEFAULT_TEXT_QA_PROMPT_TMPL)

# Refine templates
DEFAULT_REFINE_PROMPT_TMPL = (
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context and using the best of your knowledge, improve the existing answer. "
    "If you can't improve the existing answer, just repeat it again."
)
DEFAULT_REFINE_PROMPT = PromptTemplate(DEFAULT_REFINE_PROMPT_TMPL)

CHAT_REFINE_PROMPT_TMPL_MSGS = [
    ChatMessage(content="{query_str}", role=MessageRole.USER),
    ChatMessage(content="{existing_answer}", role=MessageRole.ASSISTANT),
    ChatMessage(
        content="We have the opportunity to refine the above answer "
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Given the new context and using the best of your knowledge, improve the existing answer. "
        "If you can't improve the existing answer, just repeat it again.",
        role=MessageRole.USER,
    ),
]

CHAT_REFINE_PROMPT = ChatPromptTemplate(CHAT_REFINE_PROMPT_TMPL_MSGS)

# refine prompt selector
REFINE_TEMPLATE = SelectorPromptTemplate(
    default_template=DEFAULT_REFINE_PROMPT,
    conditionals=[(is_chat_model, CHAT_REFINE_PROMPT)],
)
DEFAULT_TERM_STR = (
    "Make a list of terms and definitions that are defined in the context, "
    "with one pair on each line. "
    "If a term is missing it's definition, use your best judgment. "
    "Write each line as as follows:\nTerm: <term> Definition: <definition>"
)


def get_llm(
    llm_name,
    model_temperature,
    api_key,
    max_tokens=256,
):
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI(
        temperature=model_temperature,
        model=llm_name,
        max_tokens=max_tokens,
    )


def extract_terms(
    documents,
    term_extract_str,
    llm_name,
    model_temperature,
    api_key,
):
    llm = get_llm(
        llm_name,
        model_temperature,
        api_key,
        max_tokens=1024,
    )

    temp_index = SummaryIndex.from_documents(
        documents,
    )
    query_engine = temp_index.as_query_engine(
        response_mode="tree_summarize",
        llm=llm,
    )
    terms_definitions = str(query_engine.query(term_extract_str))
    terms_definitions = [
        x
        for x in terms_definitions.split("\n")
        if x and "Term:" in x and "Definition:" in x
    ]
    # parse the text into a dict
    terms_to_definition = {
        x.split("Definition:")[0]
        .split("Term:")[-1]
        .strip(): x.split("Definition:")[-1]
        .strip()
        for x in terms_definitions
    }
    return terms_to_definition


DEFAULT_TERMS = {
    "New York City": "The most populous city in the United States, located at the southern tip of New York State, and the largest metropolitan area in the U.S. by both population and urban area.",
    "boroughs": "Five administrative divisions of New York City, each coextensive with a respective county of the state of New York: Brooklyn, Queens, Manhattan, The Bronx, and Staten Island.",
    "metropolitan statistical area": "A geographical region with a relatively high population density at its core and close economic ties throughout the area.",
    "combined statistical area": "A combination of adjacent metropolitan and micropolitan statistical areas in the United States and Puerto Rico that can demonstrate economic or social linkage.",
    "megacities": "A city with a population of over 10 million people.",
    "United Nations": "An intergovernmental organization that aims to maintain international peace and security, develop friendly relations among nations, achieve international cooperation, and be a center for harmonizing the actions of nations.",
    "Pulitzer Prizes": "A series of annual awards for achievements in journalism, literature, and musical composition in the United States.",
    "Times Square": "A major commercial and tourist destination in Manhattan, New York City.",
    "New Netherland": "A Dutch colony in North America that existed from 1614 until 1664.",
    "Dutch West India Company": "A Dutch trading company that operated as a monopoly in New Netherland from 1621 until 1639-1640.",
    "patroon system": "A system instituted by the Dutch to attract settlers to New Netherland, whereby wealthy Dutchmen who brought 50 colonists would be awarded land and local political autonomy.",
    "Peter Stuyvesant": "The last Director-General of New Netherland, who served from 1647 until 1664.",
    "Treaty of Breda": "A treaty signed in 1667 between the Dutch and English that resulted in the Dutch keeping Suriname and the English keeping New Amsterdam (which was renamed New York).",
    "African Burying Ground": "A cemetery discovered in Foley Square in the 1990s that included 10,000 to 20,000 graves of colonial-era Africans, some enslaved and some free.",
    "Stamp Act Congress": "A meeting held in New York in 1765 in response to the Stamp Act, which imposed taxes on printed materials in the American colonies.",
    "Battle of Long Island": "The largest battle of the American Revolutionary War, fought on August 27, 1776, in Brooklyn, New York City.",
    "New York Police Department": "The police force of New York City.",
    "Irish immigrants": "People who immigrated to the United States from Ireland.",
    "lynched": "To kill someone, especially by hanging, without a legal trial.",
    "civil unrest": "A situation in which people in a country are angry and likely to protest or fight.",
    "megacity": "A very large city, typically one with a population of over ten million people.",
    "World Trade Center": "A complex of buildings in Lower Manhattan, New York City, that were destroyed in the September 11 attacks.",
    "COVID-19": "A highly infectious respiratory illness caused by the SARS-CoV-2 virus.",
    "monkeypox outbreak": "An outbreak of a viral disease similar to smallpox, which occurred in the LGBT community in New York City in 2022.",
    "Hudson River": "A river in the northeastern United States, flowing from the Adirondack Mountains in New York into the Atlantic Ocean.",
    "estuary": "A partly enclosed coastal body of brackish water with one or more rivers or streams flowing into it, and with a free connection to the open sea.",
    "East River": "A tidal strait in New York City.",
    "Five Boroughs": "Refers to the five counties that make up New York City: Bronx, Brooklyn, Manhattan, Queens, and Staten Island.",
    "Staten Island": "The most suburban of the five boroughs, located southwest of Manhattan and connected to it by the free Staten Island Ferry.",
    "Todt Hill": "The highest point on the eastern seaboard south of Maine, located on Staten Island.",
    "Manhattan": "The geographically smallest and most densely populated borough of New York City, known for its skyscrapers, Central Park, and cultural, administrative, and financial centers.",
    "Brooklyn": "The most populous borough of New York City, located on the western tip of Long Island and known for its cultural diversity, independent art scene, and distinctive neighborhoods.",
    "Queens": "The largest borough of New York City, located on Long Island north and east of Brooklyn, and known for its ethnic diversity, commercial and residential prominence, and hosting of the annual U.S. Open tennis tournament.",
    "The Bronx": "The northernmost borough of New York",
}

if "all_terms" not in st.session_state:
    st.session_state["all_terms"] = DEFAULT_TERMS


def insert_terms(terms_to_definition):
    for term, definition in terms_to_definition.items():
        doc = Document(text=f"Term: {term}\nDefinition: {definition}")
        st.session_state["llama_index"].insert(doc)


@st.cache_resource
def initialize_index(llm_name, model_temperature, api_key):
    """Create the VectorStoreIndex object."""
    Settings.llm = get_llm(llm_name, model_temperature, api_key)

    # create a vector store index for each folder
    try:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./initial_index")
        )
    except Exception as e:
        print(e)
        docs = [
            Document(text=f"Term: {key}\nDefinition: {value}")
            for key, value in DEFAULT_TERMS.items()
        ]
        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir="./initial_index")
    return index


@st.cache_resource
def get_file_extractor():
    image_parser = ImageReader(keep_image=True, parse_text=True)
    file_extractor = {
        ".jpg": image_parser,
        ".png": image_parser,
        ".jpeg": image_parser,
    }
    return file_extractor


file_extractor = get_file_extractor()


st.title("🦙 Llama Index Term Extractor 🦙")

setup_tab, terms_tab, upload_tab, query_tab = st.tabs(
    ["Setup", "All Terms", "Upload/Extract Terms", "Query Terms"]
)

with setup_tab:
    st.subheader("LLM Setup")
    api_key = st.text_input("Enter your OpenAI API key here", type="password")
    llm_name = st.selectbox("Choose an LLM", ["gpt-3.5-turbo", "gpt-4"])
    model_temperature = st.slider(
        "Model Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1
    )
    term_extract_str = st.text_area(
        "Enter your term extraction prompt here",
        value=DEFAULT_TERM_STR,
    )
with upload_tab:
    st.subheader("Extract and Query Definitions")
    if st.button("Initialize Index and Reset Terms"):
        st.session_state["llama_index"] = initialize_index(
            llm_name, model_temperature, api_key
        )
        st.session_state["all_terms"] = {}
    if "llama_index" in st.session_state:
        st.markdown(
            "Either upload an image/screenshot of a document, or enter the text manually."
        )
        uploaded_file = st.file_uploader(
            "Upload an image/screenshot of a document:",
            type=["png", "jpg", "jpeg"],
        )
        document_text = st.text_area("Or enter raw text")
        # TODO remove uploaded_file in docs and update the text
        if st.button("Extract Terms and Definitions") and (
            document_text or uploaded_file
        ):
            st.session_state["terms"] = {}
            terms_docs = {}
            with st.spinner("Extracting (images may be slow)..."):
                if document_text:
                    terms_docs.update(
                        extract_terms(
                            [Document(text=document_text)],
                            term_extract_str,
                            llm_name,
                            model_temperature,
                            api_key,
                        )
                    )
                if uploaded_file:
                    Image.open(uploaded_file).convert("RGB").save("temp.png")
                    img_reader = SimpleDirectoryReader(
                        input_files=["temp.png"], file_extractor=file_extractor
                    )
                    img_docs = img_reader.load_data()
                    os.remove("temp.png")
                    terms_docs.update(
                        extract_terms(
                            img_docs,
                            term_extract_str,
                            llm_name,
                            model_temperature,
                            api_key,
                        )
                    )
            st.session_state["terms"].update(terms_docs)

    if "terms" in st.session_state and st.session_state["terms"]:
        st.markdown("Extracted terms")
        st.json(st.session_state["terms"])

        if st.button("Insert terms?"):
            with st.spinner("Inserting terms"):
                insert_terms(st.session_state["terms"])
            st.session_state["all_terms"].update(st.session_state["terms"])
            st.session_state["terms"] = {}
            st.experimental_rerun()

with terms_tab:
    with terms_tab:
        st.subheader("Current Extracted Terms and Definitions")
        st.json(st.session_state["all_terms"])

with query_tab:
    st.subheader("Query for Terms/Definitions!")
    st.markdown(
        (
            "The LLM will attempt to answer your query, and augment it's answers using the terms/definitions you've inserted. "
            "If a term is not in the index, it will answer using it's internal knowledge."
        )
    )
    if st.button("Initialize Index and Reset Terms", key="init_index_2"):
        st.session_state["llama_index"] = initialize_index(
            llm_name, model_temperature, api_key
        )
        st.session_state["all_terms"] = {}

    if "llama_index" in st.session_state:
        query_text = st.text_input("Ask about a term or definition:")
        if query_text:
            query_text = query_text
            # breakpoint()
            with st.spinner("Generating answer..."):
                response = (
                    st.session_state["llama_index"]
                    .as_query_engine(
                        similarity_top_k=5,
                        response_mode="compact",
                        text_qa_template=TEXT_QA_TEMPLATE,
                        refine_template=DEFAULT_REFINE_PROMPT,
                    )
                    .query(query_text)
                )
            st.markdown(str(response))
