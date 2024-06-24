import cv2
import supervision as sv
from inference import get_model
import streamlit as st
from PIL import Image

model = get_model(model_id="fish-ku7kf/1", api_key='ZAlitxVtkbZWqNvDDxOw')

# 上傳檔案
uploaded_file = st.file_uploader(
            "**上傳圖片**", type=['png', 'jpeg', 'jpg'])
if uploaded_file is not None:
    is_valid = True
    with st.spinner(text='Loading...'):
        st.image(uploaded_file)
        image = Image.open(uploaded_file)
        # picture = image.save(f'data/images/{uploaded_file.name}')
        # source = f'data/images/{uploaded_file.name}'
        result = model.infer(image, confidence=0.1)[0]
        # load the results into the supervision Detections api
        detections = sv.Detections.from_inference(result)

        label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)
        # create supervision annotators
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        # annotate the image with our inference results
        annotated_image = image.copy()
        annotated_image = bounding_box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        st.image(annotated_image)
else:
    is_valid = False
    
st.write(cv2.__version__)