

![](./screenshot/demo.png)

本專案利用了兩種主要的 AI 技術：YOLOv9 物件偵測技術和 ResNet18 架構進行遷移學習，來完成魚類位置偵測和物種分類。並採用Streamlit 建立互動式的 Web 界面，用戶可以上傳照片，並即時查看辨識結果。

## 主要功能：
1. 魚類位置偵測：
    - 使用 YOLOv9 物件偵測技術來準確地識別和標註魚所在的位置，提供精確的偵測結果。
2. 魚類物種分類：
     - 採用 ResNet18 架構進行遷移學習，訓練了包含 289 種魚類的分類模型，能夠精確地識別出魚的物種。

## 操作步驟：
1. 上傳魚類圖片：
    - 用戶在應用程式上傳含有魚的圖片。
2. 魚類偵測：
    - 系統首先使用 YOLOv9 技術進行物件偵測，標註出圖片中魚的所在位置。
3. 物種識別：
    - 接著，系統會對偵測到的魚進行物種分類，使用訓練好的 ResNet18 模型來識別魚的具體種類，並顯示識別結果和概率。
## 使用技術：
1. YOLOv9：
     - 最新的物件偵測技術，用於快速而準確地偵測魚的位置。
3. ResNet18：
     - 一種深度學習架構，經過遷移學習訓練來識別多種魚類。
   
## 項目應用：
- 魚類研究：科學家和研究人員可以利用此工具快速識別和分類魚類。
- 漁業管理：幫助漁業從業者識別捕獲的魚類種類，進行資源管理。
- 教育用途：用於生物課程中魚類識別教學，提高學習的趣味性和效果。


### Fish Dataset
- [3926-1](https://universe.roboflow.com/siqi-li/fish-detector-ruldt/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
- [1850-3](https://universe.roboflow.com/fairuz/fish_model-4cl7s)
- [3322-1](https://universe.roboflow.com/mec-ywdlp/fish-c67za)
- [466-1](https://universe.roboflow.com/siqi-li/fish-detecton-2.0)

- [我自己的roboflow](https://universe.roboflow.com/yolo-zbpxw/fish-ku7kf/model/1)


## Demo
- [Roboflow Fish detect inference](https://detect.roboflow.com/?model=fish-ku7kf&version=1&api_key=ZAlitxVtkbZWqNvDDxOw)
- [fishial (魚種辨識)](https://www.fishial.ai/solutions)

- [中研院台灣魚種資料](https://fishdb.sinica.edu.tw/knowledge_home)


## 資源
- [使用機器學習辨識魚 Vize.ai 的服務(!!!可作為教學)](https://medium.com/lapis/identify-fish-using-machine-learning-13399e62e9bf)
- [fishial分類影像分割](https://github.com/fishial/fish-identification)

## google search Lens
- [lingolens](https://github.com/OSINT-mindset/lingolens)


```
streamlit run main.py
```
