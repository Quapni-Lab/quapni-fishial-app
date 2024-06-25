import cv2
import supervision as sv
from inference import get_model
import streamlit as st
from PIL import Image
import numpy as np
## classification_task
import os
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from classification_task.inference_class import EmbeddingClassifier

class YoloDetector:
    """YOLO 檢測器類別，封裝了圖像處理和檢測的主要方法。

    Attributes:
        model: YOLO 模型，用於進行物體檢測。
    """

    def __init__(self, model_id, api_key):
        """
        初始化 YoloDetector 物件。

        Args:
            model_id (str): 模型 ID。
            api_key (str): API 金鑰。
        """
        self.model = get_model(model_id=model_id, api_key=api_key)

    def load_image(self, uploaded_file):
        """
        從上傳的文件中讀取圖像。

        Args:
            uploaded_file: 上傳的圖像文件。

        Returns:
            numpy.ndarray: 讀取的圖像。
        """
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        return image

    def resize_image(self, image, target_width=640):
        """
        將圖像等比例縮放到指定寬度。

        Args:
            image (numpy.ndarray): 原始圖像。
            target_width (int, optional): 目標寬度。默認為 640。

        Returns:
            numpy.ndarray: 縮放後的圖像。
        """
        original_height, original_width = image.shape[:2]
        scale_ratio = target_width / original_width
        target_height = int(original_height * scale_ratio)
        resized_image = cv2.resize(image, (target_width, target_height))
        return resized_image

    def run_inference(self, image):
        """
        執行模型推論，檢測圖像中的物體。

        Args:
            image (numpy.ndarray): 輸入圖像。

        Returns:
            supervision.Detections: 檢測結果。
        """
        result = self.model.infer(image, confidence=0.1)[0]
        detections = sv.Detections.from_inference(result)
        return detections

    def get_max_confidence_detection(self, detections):
        """
        從檢測結果中找到最大置信度的物體。

        Args:
            detections (supervision.Detections): 檢測結果。

        Returns:
            supervision.Detections: 包含最大置信度物體的檢測結果。
        """
        if len(detections.xyxy) > 1:
            max_confidence_index = detections.confidence.argmax()
            max_detection_xyxy = detections.xyxy[max_confidence_index].reshape(1, -1)
            max_detection_confidence = detections.confidence[max_confidence_index:max_confidence_index+1]
            max_detection_class_id = detections.class_id[max_confidence_index:max_confidence_index+1]
            max_detection_data = {key: value[max_confidence_index:max_confidence_index+1] for key, value in detections.data.items()}
            detections = sv.Detections(
                xyxy=max_detection_xyxy,
                confidence=max_detection_confidence,
                class_id=max_detection_class_id,
                data=max_detection_data
            )
        return detections

    def add_confidence_to_label(self, detections):
        """
        將置信度添加到標籤中。

        Args:
            detections (supervision.Detections): 檢測結果。

        Returns:
            list: 標籤列表，包含置信度信息。
        """
        labels = []
        for i in range(len(detections.xyxy)):
            class_name = detections.data['class_name'][i]
            confidence = detections.confidence[i]
            label = f"Fish {confidence*100:.0f}%"
            labels.append(label)
        return labels

    def annotate_image(self, image, detections, labels):
        """
        標註圖像，顯示檢測到的物體和置信度。

        Args:
            image (numpy.ndarray): 輸入圖像。
            detections (supervision.Detections): 檢測結果。
            labels (list): 標籤列表。

        Returns:
            numpy.ndarray: 標註後的圖像。
        """
        label_annotator = sv.LabelAnnotator(text_color=sv.Color.WHITE, text_position=sv.Position.TOP_CENTER, color=sv.Color(r=251, g=81, b=163))
        corner_annotator = sv.BoxCornerAnnotator(color=sv.Color(r=251, g=81, b=163))

        annotated_image = image.copy()
        annotated_image = corner_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        return annotated_image

def crop_max_detection(image, detections):
    """
    裁剪圖像中最大置信度的物體。

    Args:
        image (numpy.ndarray): 輸入圖像。
        detections (supervision.Detections): 檢測結果。

    Returns:
        numpy.ndarray: 裁剪後的圖像，或 None 如果沒有檢測到物體。
    """
    if len(detections.xyxy) > 0:
        x1, y1, x2, y2 = map(int, detections.xyxy[0])
        cropped_image = image[y1:y2, x1:x2]
        return cropped_image
    return None

def display_results(results, cropped_image):
    """
    顯示分類結果和裁剪後的圖像。

    Args:
        results (list): 分類結果列表。
        cropped_image (numpy.ndarray): 裁剪後的圖像。
    """
    st.markdown('#### 分析結果')
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

    if cropped_image is not None:
        with table_col2:
            st.image(cropped_image[:,:,::-1])

def main():
    """主函數，執行 Streamlit 應用程式。"""
    st.title("Quapni Fish Detection App")
    st.caption("上傳一張圖片，識別魚的種類")

    model_id = "fish-ku7kf/1"
    
    api_key = st.secrets["roboflow_api_key"]

    detector = YoloDetector(model_id, api_key)

    uploaded_file = st.file_uploader("**上傳圖片**", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        with st.spinner(text='Loading...'):
            image = detector.load_image(uploaded_file)
            resized_image = detector.resize_image(image)
            detections = detector.run_inference(resized_image)
            detections = detector.get_max_confidence_detection(detections)
            labels = detector.add_confidence_to_label(detections)
            annotated_image = detector.annotate_image(resized_image, detections, labels)
            st.image(annotated_image[:,:,::-1])

            cropped_image = crop_max_detection(resized_image, detections)

            results = [
                {"name": "Surf bream", "scientific_name": "Acanthopagrus australis", "confidence": 0.92},
                {"name": "Sooty grunter", "scientific_name": "Hephaestus fuliginosus", "confidence": 0.89},
                {"name": "Sheepshead", "scientific_name": "Archosargus probatocephalus", "confidence": 0.10},
            ]

            display_results(results, cropped_image)

if __name__ == '__main__':
    main()
