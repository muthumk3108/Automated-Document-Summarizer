import streamlit as st
from transformers import pipeline

# Instantiating summarization pipeline with the bart-finetuned-samsum model
summarizer = pipeline(task="summarization", model="luisotorres/bart-finetuned-samsum")

# Title
st.title("📝 Text Summarization with BART")

# Creating a sidebar for input
with st.sidebar:
    st.header("Input")
    input_text = st.text_area("Enter a text or dialogue for summarization.")

# Creating a button to start the summarization
if st.button("Summarize"):
    # If the input box isn't empty, process the input and generate a summary
    if input_text:
        summary = summarizer(input_text, max_length=1024, min_length=0, do_sample=False)
        st.subheader("Original Text")
        st.write(input_text)
        st.subheader("Summary")
        st.write(summary[0]["summary_text"])
    else:
        st.warning("Enter a text or dialogue for summarization.")
