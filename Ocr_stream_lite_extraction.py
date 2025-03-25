import fitz  # PyMuPDF
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytesseract
import random
import re
import time
import streamlit as st
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\maxik\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
class Ocr:
    def __init__(self, pdf_path, output_text_file):
        self.pdf_path = pdf_path
        self.output_text_file = output_text_file

    def convert_pdf_to_text(self):
        try:
            # Open the PDF file
            pdf_document = fitz.open(self.pdf_path)
            extracted_text = ""
            for page_number in range(len(pdf_document)):
                # Render each page as an image
                page = pdf_document.load_page(page_number)
                pix = page.get_pixmap()
                image_path = f"page_{page_number + 1}.png"
                pix.save(image_path)

                # Open the saved image and extract text using pytesseract
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image)
                extracted_text += text

            # Write the extracted text to a file
            with open(self.output_text_file, "w", encoding="utf-8") as f:
                f.write(extracted_text)

            return extracted_text
        except Exception as e:
            return f"An error occurred: {e}"

class StreamliteGUI:
    def __init__(self):
        st.title("PDF to Text OCR Tool")
        self.upload_file()

    def upload_file(self):
        # Allow the user to upload a PDF file
        file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if file is not None:
            # Save the uploaded file to a temporary location
            with open("uploaded_file.pdf", "wb") as f:
                f.write(file.read())

            # Specify output text file
            output_file = "extracted_text.txt"

            # Process the uploaded file using the Ocr class
            ocr_processor = Ocr("uploaded_file.pdf", output_file)
            extracted_text = ocr_processor.convert_pdf_to_text()

            # Display the extracted text in the Streamlit app
            st.text_area("Extracted Text", extracted_text, height=300)

            # Print the extracted text to the console
            self.print_text(extracted_text)

    def print_text(self, text):
        print("Extracted Text from the PDF:")
        print(text)


# Run the Streamlit app
if __name__ == "__main__":
    StreamliteGUI()
