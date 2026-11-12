import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")


class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str


parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
    ('system', """
Extract movie information from the paragraph
{format_instructions}
"""),
    ("human", "{paragraph}")
])

st.set_page_config(page_title="Movie Information Extractor", page_icon="🎬")

st.title("🎬 Movie Information Extractor")
st.caption("Paste a paragraph about a movie to extract its details.")

para = st.text_area("Your paragraph", height=220, placeholder="Paste your paragraph here...")

if st.button("Extract", type="primary"):
    if not para.strip():
        st.warning("Please enter a paragraph first.")
    else:
        with st.spinner("Extracting..."):
            final_prompt = prompt.invoke(
                {"paragraph": para,
                 "format_instructions": parser.get_format_instructions()
                }
            )
            response = model.invoke(final_prompt)

        st.subheader("Result")
        st.code(response.content, language="json")