import cv2
from cv2.ximgproc import guidedFilter
import numpy as np
import os
import model

# 폴더 경로 설정
folder_path = './wb_result/'
results_folder = './wb_pred_result/'

result = []

def process_image(image_path, des_scores):
    # load model
    HazDesNet = model.load_HazDesNet()
    HazDesNet.summary()

    # read image
    img = cv2.imread(image_path)

    # image to grayscale
    guide = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # expand dimension(0)
    img = np.expand_dims(img, axis=0)

    # color veil removal method
    balanced_img = cv2.xphoto.createSimpleWB().balanceWhite(img)

    # predict haze image
    haz_des_map = HazDesNet.predict(balanced_img)

    haz_des_map = haz_des_map[0, :, :, 0]

    # resize guide image
    guide = cv2.resize(guide, (haz_des_map.shape[1], haz_des_map.shape[0]))

    # generate hazy image using guided filter
    haz_des_map = guidedFilter(guide=guide, src=haz_des_map, radius=32, eps=500)

    # score
    des_score = np.mean(haz_des_map)

    # save
    result_filename = os.path.basename(image_path)
    result_path = os.path.join(results_folder, result_filename)
    cv2.imwrite(result_path, haz_des_map * 255)
    result.append((result_filename, des_score))
    #  des_score를 리스트에 추가
    des_scores.append((result_filename, des_score))

def main():
    # 결과 폴더가 없으면 생성
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # 이미지 폴더에서 모든 이미지 파일 목록을 가져옴
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    # 각 이미지에 대한 des_score를 저장할 리스트
    des_scores = []

    for image_path in image_files:
        process_image(image_path, des_scores)

    for chunk in [result[i:i+5] for i in range(0, len(result), 5)]:
        for result_filename, des_score in chunk:
            print("the haze density score of " + result_filename + " is: %.5f" % (des_score))
        print("-------------------------------------------------")

if __name__ == "__main__":
    main()
