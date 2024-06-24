import cv2
import supervision as sv
from inference import get_model
import streamlit as st
from PIL import Image


# 上傳檔案
uploaded_file = st.file_uploader(
            "**上傳圖片**", type=['png', 'jpeg', 'jpg'])
if uploaded_file is not None:
    is_valid = True
    with st.spinner(text='Loading...'):
        st.image(uploaded_file)
        picture = Image.open(uploaded_file)
        # picture = picture.save(f'data/images/{uploaded_file.name}')
        # source = f'data/images/{uploaded_file.name}'
        print(picture)
else:
    is_valid = False
    
st.write(cv2.__version__)