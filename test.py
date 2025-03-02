from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# ResNet50 모델 불러오기
model = ResNet50(weights='imagenet')

# 예측을 위한 이미지 로드 및 전처리
img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# 이미지에 대한 예측
predictions = model.predict(img_array)

# 예측 결과 디코딩 및 출력
decoded_predictions = decode_predictions(predictions, top=3)[0]
print(decoded_predictions)
