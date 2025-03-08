
# Image Classification using Azure Computer Vision API and Cosine Similarity with recommendation system.


import os
import json
import requests
import http.client,urllib.parse
# from tenacity import retry, wait_fixed, stop_after_attempt
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures



load_dotenv()

from sklearn.metrics.pairwise import cosine_similarity

aiVisionApiKey = os.getenv("AI_VISION_API_KEY")
aiVisionRegion=os.getenv("AI_VISION_REGION")
aiVisionEndpoint=os.getenv("AI_VISION_ENDPOINT")
credentials = DefaultAzureCredential()


FilePath = 'Images'

# @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
def get_image_vector(image_path,key,region):
    headers = {
        'Ocp-Apim-Subscription-Key': key,
    }
    params = urllib.parse.urlencode({
        'model-version': 'latest',
    })
    try:
        if image_path.startswith(('http://','https://')):
            headers['Content-Type'] = 'application/json'
            body = json.dumps({'url': image_path})
        else:
            headers['Content-Type'] = 'application/octet-stream'
            with open(image_path, 'rb') as f:
                body = f.read()
        conn = http.client.HTTPSConnection(f'{region}.api.cognitive.microsoft.com')
        conn.request("POST", "/computervision/retrieval:vectorizeImage?api-version=2023-04-01-preview&%s" % params, body, headers)
        response = conn.getresponse()
        data = json.load(response)
        conn.close()

        if response.status != 200:
            raise Exception(f"Error processing image {image_path}: {data.get('message', '')}")

        return data.get("vector")

    except (requests.exceptions.Timeout, http.client.HTTPException) as e:
        print(f"Timeout/Error for {image_path}. Retrying...")
        raise
    
def generate_embeddings(text, aiVisionEndpoint, aiVisionApiKey):  
    url = f"{aiVisionEndpoint}/computervision/retrieval:vectorizeText"  
  
    params = {  
        "api-version": "2023-02-01-preview"  
    }  
  
    headers = {  
        "Content-Type": "application/json",  
        "Ocp-Apim-Subscription-Key": aiVisionApiKey  
    }  
  
    data = {  
        "text": text  
    }  
  
    response = requests.post(url, params=params, headers=headers, json=data)  
  
    if response.status_code == 200:  
        embeddings = response.json()["vector"]  
        return embeddings  
    else:  
        print(f"Error: {response.status_code} - {response.text}")  
        return None  
   

    
# image_embedding = {
    
# }
# FILES = os.listdir(FilePath)
# DIRPATH = os.path.join(os.getcwd(), FilePath)
# for file in FILES:
#     image_embedding[file] = get_image_vector(os.path.join(DIRPATH,file),aiVisionApiKey,aiVisionRegion)
# with open("image_embedding.json", "w") as outfile:
#     json.dump(image_embedding, outfile)
    
    
    
# keys = list(image_embedding.keys())
# print(keys)
# arr = [image_embedding[keys[0]]]
# sim_dict={}
# for file in FILES:
#     sim = cosine_similarity(arr,[image_embedding[file]])
#     sim_dict[file] = sim[0][0]
    
    
#     print(f"Similarity between {keys[0]} and {file}: {sim[0][0]}")

# sim_sort = dict(sorted(sim_dict.items(), key=lambda x: x[1], reverse=True))



# display(Image(filename = os.path.join(DIRPATH,keys[0])))

# print(sim_sort.keys()[1])

# print('-'*100)
# print(sim_sort)
# print('-'*100)

    