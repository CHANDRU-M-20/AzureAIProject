import streamlit as st
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from app import get_image_vector,generate_embeddings
import time
from classification import classifiy_iamges
import pandas as pd 
# from twisted.python.filepath import FilePath


aiVisionApiKey = os.getenv("AI_VISION_API_KEY")
aiVisionRegion=os.getenv("AI_VISION_REGION")
aiVisionEndpoint=os.getenv("AI_VISION_ENDPOINT")
credentials = DefaultAzureCredential()

st.title('Automatic Product Recommendations and Product Classification Using Azure Vision')
tab1, tab2 , tab3 = st.tabs(["Image Search", "Search System",'Image Classification'])
with open('image_embedding.json','r') as fils:
    image_embedding = json.load(fils)

embed_data={}
FilePath = 'uploads'

def save_uploaded_image(images):
    if not os.path.exists(FilePath):
        os.makedirs(FilePath)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    file_path=os.path.join("uploads",f"{timestamp}_{images.name}")
    with open(file_path, "wb") as f:
        f.write(images.getbuffer())
        
def get_last_uploaded_file(directory):
    files = os.listdir(directory)
    files.sort(key=lambda x:os.path.getmtime(os.path.join(directory,x)),reverse=True)
    if files:
        return files
    else:
        return None
    
def display_data(sim_dict):
    st.divider()
    st.header("Recommanded Images")
    sim_sort = dict(sorted(sim_dict.items(), key=lambda x: x[1], reverse=True))
    # st.write(sim_sort)
    count = 0
    # for i,j in sim_sort.items():
    #     if count==3:
    #         break
    #     st.write(f"Similarity Score : {round(j,2)}")
    #     st.image(os.path.join('Images',i))
    #     st.divider()
    #     count+=1
    flag=0
    for i,j in sim_sort.items():
        if j>0.25 and j<1:
            st.write(f"Similarity Score : {round(j,2)}")            
            st.image(os.path.join('Images',i))
            st.divider()
            flag=1
            count+=1
        # elif j> 0.65:
        #     st.write(f"Similarity Score : {round(j,2)}")
        #     st.image(os.path.join('Images', i))
        #     st.divider()    
        #     flag=1
        # elif j>0.25:
        #     st.write(f"Similarity Score : {round(j,2)}")
        #     st.image(os.path.join('Images', i))
        #     st.divider()    
        #     flag=1
            
    if flag==0:
        st.write("No similar images found")
    else:
        st.write("Similar images found")

def text_recommand(data):
    if data is not None:
        # st.write(data)
        vector_data = generate_embeddings(text=data, aiVisionEndpoint=aiVisionEndpoint, aiVisionApiKey=aiVisionApiKey)
        # st.write(vector_data)
        
        TrainedImages_FilePath = 'Images'
        sim_dict={}
        FILES = os.listdir(TrainedImages_FilePath)
        DIRPATH = os.path.join(os.getcwd(), TrainedImages_FilePath)
        for file in FILES:
            sim = cosine_similarity([vector_data], [tuple(image_embedding[file])])
            sim_dict[file] = sim[0][0]
            
        return display_data(sim_dict)

def create_sidebar():
    with st.sidebar:
        st.title("Product Classification")
        st.write("Upload the Image for Classifications..")
        files = st.file_uploader("Choose an image...", type="jpg")
        if files:
            st.success(f"File '{files.name}' uploaded successfully!")
        else:
            st.warning("No files were uploaded.")
    return files


    

with tab1:  
    uploaded_file = create_sidebar()
    if uploaded_file is not None:
        save_uploaded_image(uploaded_file)
        last_uploaded_file = get_last_uploaded_file(FilePath)        
        images = f'uploads/{last_uploaded_file[0]}'
        st.image(f'uploads/{last_uploaded_file[0]}')
        if st.button("Predict"):
            result,classify,tags=classifiy_iamges(images)
            st.write(f"Image Classifications : {result}")
            st.write(f"Confidence Score: {classify}")
            # st.write(f"Image Tags: {tags}")
            st.divider()
            tag = tags[0]['name']
            st.write(f"Image Classifications : {tag}")
            text_recommand(result)    
    else:
       st.warning("Please Upload the Image for classification and get the similar images 😊")
        
        
        # ** ------------ Get the Image Vectors-----------------**        
        # embed_data[uploaded_file.name]=get_image_vector(image_path=images,key=aiVisionApiKey,region=aiVisionRegion)
        # # st.write(embed_data)
        # # st.write(embed_data.values())
        
        # TrainedImages_FilePath = 'Images'
        # sim_dict={}
        # FILES = os.listdir(TrainedImages_FilePath)
        # DIRPATH = os.path.join(os.getcwd(), TrainedImages_FilePath)
        # keys = list(embed_data.keys())
        # for file in FILES:
        #     sim = cosine_similarity([embed_data[keys[0]]],[image_embedding[file]])
        #     sim_dict[file] = sim[0][0]
            
        # display_data(sim_dict)
        # print(f"Similarity between {keys[0]} and {file}: {sim[0][0]}")

    
with tab2:
    data = st.text_input("Enter the product name")
    submit_button = st.button('Submit')
    if len(data)!=0 and submit_button:
        text_recommand(data)
    else:
        st.write("Please Enter a product name")
    
            


with tab3:
    pass