import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cohere
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Directly set the Cohere API key here
cohere_api_key = "UvQcnNHSF42oPDGTWv6P9OpGrOCMb9lPgKOjxj3m"  # Replace with your actual Cohere API key
co = cohere.Client(cohere_api_key)

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to get Cohere embeddings and store them in FAISS
def get_vector_store(text_chunks):
    embeddings = get_cohere_embeddings(text_chunks)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get Cohere embeddings
def get_cohere_embeddings(text_chunks):
    response = co.embed(
        model='embed-english-v2.0',  # Cohere embedding model
        texts=text_chunks
    )
    embeddings = response.embeddings
    return embeddings

# Function to create a QA chain with Cohere
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Cohere's language model for text generation (answering questions)
    def cohere_answer(context, question):
        prompt = prompt_template.format(context=context, question=question)
        response = co.generate(
            model="xlarge",  # Use an appropriate Cohere model for generation
            prompt=prompt,
            max_tokens=200,
            temperature=0.3
        )
        return response.generations[0].text.strip()

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(cohere_answer, chain_type="stuff", prompt=prompt)

    return chain

# Function to process user input and get a response
def user_input(user_question):
    embeddings = get_cohere_embeddings

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    # Get response from Cohere model based on the question and context
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

# Main function for Streamlit UI
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Cohere💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
