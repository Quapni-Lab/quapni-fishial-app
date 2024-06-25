import cv2
import supervision as sv
from inference import get_model
import streamlit as st
from PIL import Image
import numpy as np
# import googletrans
## classification_task
import os
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from classification_task.inference_class import EmbeddingClassifier
from pathlib import Path

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
    
class ResNetClassifier:
    """ResNet 分類器，用於圖像中的魚種辨識。"""

    def __init__(self, model_folder='classification_task/model', device='cpu'):
        """
        初始化 ResNet 分類器。

        Args:
            model_folder (str): 存放模型文件的文件夾路徑。
            device (str): 運行模型的設備 ('cpu' 或 'cuda')。
        """
        # model = Path("embeddings.pt")
        # if not model.exists():
        #     with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        #         from GD_download import download_file_from_google_drive
        #         download_file_from_google_drive("19lLNWnZs8iMibYHR_3t86VYq7Z1vgxEF", model)
        # model2 = Path("model.ts")
        # if not model2.exists():
        #     with st.spinner("Downloading model2... this may take awhile! \n Don't stop it!"):
        #         from GD_download import download_file_from_google_drive
        #         download_file_from_google_drive("1y4lkJZC97vo9XX7xpq0KvdVZJCF-oRG_", model2)
        # model3 = Path("idx.json")
        # if not model3.exists():
        #     with st.spinner("Downloading model3... this may take awhile! \n Don't stop it!"):
        #         from GD_download import download_file_from_google_drive
        #         download_file_from_google_drive("1Rtsr8mp85SO5g-joyVXo3qNbFndKi748", model3)
        self.classification_path = os.path.join(model_folder, 'model.ts')
        self.data_base_path = os.path.join(model_folder, 'embeddings.pt')
        self.data_idx_path = os.path.join(model_folder, 'idx.json')
        self.device = device
        self.model = EmbeddingClassifier(
            self.classification_path,
            self.data_base_path,
            self.data_idx_path,
            # 'model.ts',
            # 'embeddings.pt',
            # 'idx.json',
            device=self.device
        )

    def classify_image(self, image):
        """
        對單一圖像進行魚種辨識。

        Args:
            image (numpy array): 要進行辨識的圖像。

        Returns:
            list of dict: 辨識結果列表，包含魚種名稱和置信度。
        """
        single_output = self.model.inference_numpy(image)
        classifier_results = []
        for item in single_output:
            fish_id = item[0]
            fish_info = item[1]
            fish_name = fish_info[0]
            fish_confidence = fish_info[1]
            classifier_results.append({"name": fish_name, "confidence": fish_confidence})
        return classifier_results

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
                if result["name"] == 'Pogonias cromis':
                    st.subheader('黑鯛') # 顯示中文名稱
                elif result["name"] == 'Trachinotus falcatus':
                    st.subheader('金鯧魚') # 顯示中文名稱
                # else:
                #     translator = googletrans.Translator() # google翻譯
                #     translation = translator.translate(result["name"], dest='zh-tw') # 翻譯成繁體中文
                #     st.subheader(translation.text) # 顯示中文名稱
                st.caption(result["name"]) #顯示英文名稱

            with col2:
                st.write("\n\n")
                st.write(f"{result['confidence'] * 100:.1f}%")

            with col3:
                st.write("\n\n")
                st.progress(result["confidence"])

    if cropped_image is not None:
        with table_col2:
            st.image(cropped_image[:,:,::-1])

def process_and_display_example_image(image_path, detector, classifier):
    image = cv2.imread(image_path)
    resized_image = detector.resize_image(image)
    detections = detector.run_inference(resized_image)
    detections = detector.get_max_confidence_detection(detections)
    labels = detector.add_confidence_to_label(detections)
    annotated_image = detector.annotate_image(resized_image, detections, labels)
    st.image(annotated_image[:,:,::-1], caption='Annotated Image')
    cropped_image = crop_max_detection(resized_image, detections)
    if cropped_image is not None:
        classifier_results = classifier.classify_image(cropped_image)
        # st.write("Classification Results:", classifier_results)
        display_results(classifier_results, cropped_image)
        st.toast('辨識成功!', icon='🎉')
    else:
        st.error("未在圖片中找到可識別的魚類" ,icon="🚨")

def main():
    """主函數，執行 Streamlit 應用程式。"""
    st.title("Quapni Fish Detection App")
    st.caption("上傳一張圖片，識別魚的種類")
    
    model_id = "fish-ku7kf/1"
    api_key = 'ZAlitxVtkbZWqNvDDxOw'
    # api_key = st.secrets["roboflow_api_key"]
    detector = YoloDetector(model_id, api_key)
    classifier = ResNetClassifier(model_folder='classification_task/model')

    tab1, tab2 = st.tabs(["⬆️ 上傳圖片", "🖼️ Example"])
    with tab1:
        uploaded_file = st.file_uploader("**上傳圖片**", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            with st.spinner(text='Loading...'):
                image = detector.load_image(uploaded_file)
                resized_image = detector.resize_image(image)
                # YOLO物件辨識
                detections = detector.run_inference(resized_image)
                detections = detector.get_max_confidence_detection(detections)
                labels = detector.add_confidence_to_label(detections)
                annotated_image = detector.annotate_image(resized_image, detections, labels)
                st.image(annotated_image[:,:,::-1])
                # 物件剪裁
                cropped_image = crop_max_detection(resized_image, detections) 
                if cropped_image is not None:
                    # ResNet魚種分類
                    classifier_results = classifier.classify_image(cropped_image)
                    # st.write("Classification Results:", classifier_results)
                    display_results(classifier_results, cropped_image)
                    st.toast('辨識成功!', icon='🎉')
                else:
                    st.error("未在圖片中找到可識別的魚類" ,icon="🚨")
                
        with tab2:
            example_col1, example_col2, example_col3 = st.columns(3)
            example_col1.image('example/黑鯛.jpg')
            example_col2.image('example/吳郭魚.jpg')
            example_col3.image('example/金鯧魚.png')
            if example_col1.button('辨識此魚', key=1):
                process_and_display_example_image('example/黑鯛.jpg', detector, classifier)
            if example_col2.button('辨識此魚', key=2):
                process_and_display_example_image('example/吳郭魚.jpg', detector, classifier)
            if example_col3.button('辨識此魚', key=3):
                process_and_display_example_image('example/金鯧魚.png', detector, classifier)

if __name__ == '__main__':
    main()
