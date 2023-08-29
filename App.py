import os
import pickle
import tensorflow
import numpy as np
from PIL import Image
import streamlit as st
from annoy import AnnoyIndex
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

os.system("cls")
base_path = "Uploads"
try:
    os.mkdir(base_path)
except:
    print("Folder Already Exists!!!")

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))
annoy_index = AnnoyIndex(2048,"angular")
annoy_index.load("features_annoy.ann")

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommendation System')

def save_image(uploaded_image):

    try:
        image_path = os.path.join(base_path,uploaded_image.name).replace("\\","/")
        print(image_path)

        with open(image_path,'wb') as file:
            file.write(uploaded_image.getbuffer())
            st.success("Image Saved Sucessfully")
        return 1
    
    except:
        return 0

def feature_extraction(img_path):

    img = image.load_img(img_path, target_size=(224, 224))

    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features):

    return annoy_index.get_nns_by_vector(features,5)

uploaded_image = st.file_uploader("Upload Your Image",type=["png","jpg","jepg","webp"])

if uploaded_image is not None:

    if save_image(uploaded_image):

        st.image(uploaded_image)

        features = feature_extraction(os.path.join(base_path,uploaded_image.name))

        recommendations = recommend(features)

        st.write("---")
        st.subheader(':red[Your] :green[Top] :blue[Recommendations]')

        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(filenames[recommendations[0]])
        with col2:
            st.image(filenames[recommendations[1]])
        with col3:
            st.image(filenames[recommendations[2]])
        with col4:
            st.image(filenames[recommendations[3]])
        with col5:
            st.image(filenames[recommendations[4]])
    else:
        st.error("Image Upload Error") 