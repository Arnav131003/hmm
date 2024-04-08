import re , os
from typing import BinaryIO
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import google.generativeai as ai
from transformers import pipeline
from genai import safety_settings
import openai
from dotenv import load_dotenv
import json

app = FastAPI()

# Load the text summarization pipeline
summarizer = pipeline("summarization", model="t5-small")
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
# Configure Google Gemini Vision Pro
generation_config = ai.GenerationConfig(
    temperature=0.4,
    top_p=1,
    top_k=32,
    max_output_tokens=2000,
)

model = ai.GenerativeModel(
    model_name="gemini-pro-vision",
    generation_config=generation_config,
)

class ImageData(BaseModel):
    data: bytes

def extract_text_from_image(image_data: ImageData):
    response = model.generate_content(
        [
            "What is written on this prescription image?",
            {
                "mime_type": "image/jpeg",
                "data": image_data.data,
            },
        ]
    )
    
    if response.candidates and response.candidates[0].content.parts:
        return response.candidates[0].content.parts[0].text
    else:
        return ""
    
    
def summarize_prescription(text: str):
    prompt = f"""
    Extract the following entities from the prescription text in JSON format:

    - patient_name
    - disease
    - disease_description
    - medicines (as a list of objects with name, dose, and frequency)

    Prescription: 
    {text}

    """

    response = openai.Completion.create(
        engine="interviewtest",
        prompt=prompt, 
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text.strip()

@app.post("/upload")
async def upload_prescription(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_data = ImageData(data=await file.read())
    
    # Extract text from the image using Google Gemini Vision Pro
    text = extract_text_from_image(image_data)
    
    # Generate summary of the prescription
    summary = summarize_prescription(text)
    
    print("Summary:", summary)
    
    # Parse the summary JSON string
    # summary_json = json.loads(summary)
    
    return 0
