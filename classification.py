import os
import json
import requests
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
import streamlit as st
load_dotenv()

from sklearn.metrics.pairwise import cosine_similarity

aiVisionApiKey = os.getenv("AI_VISION_API_KEY")
aiVisionRegion=os.getenv("AI_VISION_REGION")
aiVisionEndpoint=os.getenv("AI_VISION_ENDPOINT")
# Define image URL
def classifiy_iamges(image_paths):
    # Create an Image Analysis client
    client = ImageAnalysisClient(
        endpoint=aiVisionEndpoint,
        credential=AzureKeyCredential(aiVisionApiKey)
    )
    
    st.write("Analyzing image...")
    st.write(image_paths)
    with open(image_paths, 'rb') as f:
            body = f.read()
    

    # with open(image_path, "rb") as image_stream:
    #     result = client.analyze_image_in_stream(
    #         image_stream,
    #         visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ],
    #         gender_neutral_caption=True  # Optional (default is False)
    #     )


    # Get a caption for the image. This will be a synchronously (blocking) call.
    result = client.analyze(image_data=body,
        visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ,VisualFeatures.TAGS],
        gender_neutral_caption=True,  # Optional (default is False)
    )

    print("Image analysis results:")
    # Print caption results to the console
    print(" Caption:")
    if result.caption is not None:
        print(f"   '{result.caption.text}', Confidence {result.caption.confidence:.4f}")
        # st.write(result.tags)
    first_value = None
    if hasattr(result.tags, 'values'):  # Check if the method exists
        tag_values = list(result.tags.values())  # Convert to a list
        if tag_values:  # Check if the list is not empty
            first_value = tag_values[0]  # Get the first tag
    
    return result.caption.text,result.caption.confidence,first_value
