import streamlit as st
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve environment variables
endpoint = os.environ["VISION_TRAINING_ENDPOINT"]
training_key = os.environ["VISION_TRAINING_KEY"]
prediction_key = os.environ["VISION_PREDICTION_KEY"]
prediction_resource_id = os.environ["VISION_PREDICTION_RESOURCE_ID"]
project_id = os.environ["VISION_PROJECT_ID"]
published_name = os.environ["VISION_PUBLISHED_NAME"]

# Construct the prediction endpoint URL
# endpoint_url = f"{endpoint}/customvision/v3.0/Prediction/{project_id}/classify/iterations/{published_name}/image"

# # Path to the test image
# base_image_location = "Images/pexels-veeterzy-303383.jpg"

# # Set up the headers for the request
# headers = {
#     "Prediction-Key": prediction_key,
#     "Content-Type": "application/octet-stream"
# }

# # Open the image file in binary mode and send it to the API
# with open(base_image_location, "rb") as image_contents:
#     response = requests.post(endpoint_url, headers=headers, data=image_contents)

# # Check if the request was successful
# if response.status_code == 200:
#     results = response.json()
#     # Display the results
#     for prediction in results['predictions']:
#         st.write(f"{prediction['tagName']}: {prediction['probability'] * 100:.2f}%")
# else:
#     st.write(f"Error: {response.status_code}, {response.text}")
