import cv2
import supervision as sv
from inference import get_model
import streamlit as st
from PIL import Image
import numpy as np

model = get_model(model_id="fish-ku7kf/1", api_key='ZAlitxVtkbZWqNvDDxOw')

# 上傳檔案
uploaded_file = st.file_uploader(
            "**上傳圖片**", type=['png', 'jpeg', 'jpg'])
if uploaded_file is not None:
    is_valid = True
    with st.spinner(text='Loading...'):
        # st.image(uploaded_file)
        # image = Image.open(uploaded_file)
        # picture = image.save(f'data/images/{uploaded_file.name}')
        # source = f'data/images/{uploaded_file.name}'
        # 辨識
        
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        # 取得原始影像的尺寸
        original_height, original_width = image.shape[:2]
        # 目標寬度
        target_width = 640
        # 計算等比例縮放後的高度
        scale_ratio = target_width / original_width
        target_height = int(original_height * scale_ratio)
        # 縮放影像
        resized_image = cv2.resize(image, (target_width, target_height))

        # 執行模型推論
        result = model.infer(resized_image, confidence=0.1)[0]

        # 解析推論結果
        detections = sv.Detections.from_inference(result)

        # 檢查是否有超過1個物件
        if len(detections.xyxy) > 1:
            # 找到概率最大的物件
            max_confidence_index = detections.confidence.argmax()
            max_detection_xyxy = detections.xyxy[max_confidence_index].reshape(1, -1)
            max_detection_confidence = detections.confidence[max_confidence_index:max_confidence_index+1]
            max_detection_class_id = detections.class_id[max_confidence_index:max_confidence_index+1]
            max_detection_data = {key: value[max_confidence_index:max_confidence_index+1] for key, value in detections.data.items()}

            # 創建一個新的 Detections 物件，只包含最大物件
            detections = sv.Detections(
                xyxy=max_detection_xyxy,
                confidence=max_detection_confidence,
                class_id=max_detection_class_id,
                data=max_detection_data
            )

        # 在標籤中加入機率值
        def add_confidence_to_label(detections):
            labels = []
            for i in range(len(detections.xyxy)):
                class_name = detections.data['class_name'][i]
                confidence = detections.confidence[i]
                label = f"Fish {confidence*100:.0f}%"
                labels.append(label)
            return labels

        # 設置標籤
        labels = add_confidence_to_label(detections)
        label_annotator = sv.LabelAnnotator(text_color=sv.Color.WHITE, text_position=sv.Position.TOP_CENTER, color=sv.Color(r=251, g=81, b=163))

        # 建立標註器
        # bounding_box_annotator = sv.BoundingBoxAnnotator()
        corner_annotator = sv.BoxCornerAnnotator(color=sv.Color(r=251, g=81, b=163))


        # 標註圖片
        annotated_image = resized_image.copy()
        # annotated_image = bounding_box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = corner_annotator.annotate( scene=annotated_image, detections=detections )
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # 顯示標註後的圖片
        st.image(annotated_image[:,:,::-1])


        # 裁切最大的物件
        if len(detections.xyxy) > 0:
            x1, y1, x2, y2 = map(int, detections.xyxy[0])
            cropped_image = resized_image[y1:y2, x1:x2]

            
            
        # 模擬分類結果
        results = [
            {"name": "Surf bream", "scientific_name": "Acanthopagrus australis", "confidence": 0.92},
            {"name": "Sooty grunter", "scientific_name": "Hephaestus fuliginosus", "confidence": 0.89},
            {"name": "Sheepshead", "scientific_name": "Archosargus probatocephalus", "confidence": 0.10},
        ]
        # 設定應用程式標題
        st.markdown('#### 分析結果')
        # 顯示分類結果
        table_col1, table_col2 = st.columns([3, 1])
        with table_col1:
            for result in results:
                col1, col2, col3 = st.columns([3, 1, 3])

                with col1:
                    st.subheader(result["name"])
                    st.caption(result["scientific_name"])

                with col2:
                    st.write("\n\n")
                    st.write(f"{result['confidence']*100}%")

                with col3:
                    st.write("\n\n")
                    st.progress(result["confidence"])
                
        with table_col2:
            st.image(cropped_image[:,:,::-1])
else:
    is_valid = False
    

    
st.write(cv2.__version__)