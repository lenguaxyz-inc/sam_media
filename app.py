##app.py##

import os
import streamlit as st
import pandas as pd
import PyPDF2
from PyPDF2 import PdfReader
from pathlib import Path
import logging
import configparser

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


# Set Streamlit app title, icon, and page configuration
st.set_page_config(
    page_title="SAM - Screenplay Analysis Tool",
    page_icon="✍️",
    layout="wide"
)

# Configure logging
logging.basicConfig(filename='sam.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read configuration from a config.ini file
config = configparser.ConfigParser()
config.read('config.ini')  # Create a config.ini file with your specific settings

# Constants for file paths and configuration
PDF_FILE_PATH = config.get('Paths', 'PDF_FILE_PATH', fallback='')
TXT_FILE_PATH = config.get('Paths', 'TXT_FILE_PATH', fallback='')
PPT_FILE_PATH = config.get('Paths', 'PPT_FILE_PATH', fallback='')
OUTPUT_DIRECTORY = config.get('Paths', 'OUTPUT_DIRECTORY', fallback='')
OPENAI_API_KEY = config.get('API', 'OPENAI_API_KEY', fallback='')

# Function to extract text from a PDF file
@st.cache
def extract_text_from_pdf(file_path):
    try:
        text = ""
        with open(file_path, "rb") as file:
            pdf_reader = PdfReader(file)
            num_pages = len(pdf_reader.pages)
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        st.error(f"Error extracting text from PDF: {str(e)}")
        raise

# Function to extract text from a TXT file
@st.cache
def extract_text_from_txt(file_path):
    try:
        with open(file_path, "r") as file:
            text = file.read()
        return text
    except Exception as e:
        logging.error(f"Error reading text from TXT: {str(e)}")
        st.error(f"Error reading text from TXT: {str(e)}")
        raise

# Function to extract text from a PPT file (not implemented here)
@st.cache
def extract_text_from_ppt(file_path):
    # You can implement text extraction from PPT files using a library like python-pptx.
    # This function is a placeholder.
    st.warning("Text extraction from PPT files is not implemented.")
    return ""

# Function to write a list to a file
def write_list_to_file(items, filename):
    try:
        with open(filename, 'w') as file:
            for item in items:
                file.write(str(item) + '\n')
    except Exception as e:
        logging.error(f"Error writing to file: {str(e)}")
        st.error(f"Error writing to file: {str(e)}")
        raise

# Main function
def main():
    st.title("SAM - Screenplay Analysis Tool")
    
    # Set app header
    st.markdown("### Analyze Screenplays with SAM")
    
    # Sidebar with file uploader
    st.sidebar.title("Upload a File")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "txt", "ppt"])

    if uploaded_file is not None:
        # Extract text based on file type
        st.sidebar.header("Processing...")
        st.sidebar.text("Please wait while SAM analyzes the file.")
        
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()

        if ext == "pdf":
            roteiro = extract_text_from_pdf(uploaded_file)
        elif ext == "txt":
            roteiro = extract_text_from_txt(uploaded_file)
        elif ext == "ppt":
            roteiro = extract_text_from_ppt(uploaded_file)
        else:
            st.error(f"Unsupported file format: {ext}")
            return

        # Initialize ChatOpenAI and create a prompt template
        chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
        template = """You are SAM, a screenplay expert, your function is to analyze a script and break it down...
                      """
        prompt_template = PromptTemplate.from_template(template=template)

        # Create an LLMChain
        llm_chain = LLMChain(llm=chat, prompt=prompt_template)

        # Process the script and extract scene information
        resultado = []

        # CharacterTextSplitter configuration
        text_splitter = CharacterTextSplitter(
            separator="CUT TO:",
            chunk_size=150,
            chunk_overlap=50,
            length_function=len,
        )

        roteiro_split = text_splitter.create_documents([roteiro])
        num_roteiro_split = len(roteiro_split)

        for i in range(1, len(roteiro_split)):
            result = llm_chain.run({"input": roteiro_split[i]})
            cena = 'Cena: ' + str(i)
            store = cena + '\n' + result + '\n'
            resultado.append(store)

        # Parse the results and store them in a DataFrame
        data = []

        for text in resultado:
            # Split the text into lines
            lines = text.strip().split("\n")
            # Initialize a dictionary to store the information for this scene
            scene_data = {}
            # Process each line in the text
            for line in lines:
                if ": " in line:
                    key, value = line.split(": ", 1)
                    scene_data[key] = value
            # Append the dictionary to the list
            data.append(scene_data)

        # Display the scene information in a DataFrame
        st.sidebar.header("Screenplay Analysis Result")
        df = pd.DataFrame(data)
        st.sidebar.dataframe(df)

        # Export the data to an Excel file (if desired)
        if st.sidebar.button("Export Analysis to Excel"):
            output_excel_path = os.path.join(OUTPUT_DIRECTORY, "analise.xlsx")
            df.to_excel(output_excel_path, index=False)
            st.sidebar.success(f"Analysis exported to {output_excel_path}")

if __name__ == "__main__":
    main()
