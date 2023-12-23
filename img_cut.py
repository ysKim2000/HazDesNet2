import cv2
import os

# 폴더 경로 설정
I_DIR = './wb_stack_result/'
O_DIR = './wb_result/'


def crop_image(image):
    height, width, _ = image.shape

    middle = width // 2
    cropped_image = image[:, middle:]
    print(cropped_image)

    return cropped_image


def files(dir=I_DIR):
    names = os.listdir(dir)

    for name in names:
        print(name)
        I = cv2.imread(os.path.join(dir, name))
        result = crop_image(I)
        name = name.split('.')[0]
        cv2.imwrite(os.path.join(O_DIR, f'{name}.jpg'), result)


if __name__ == "__main__":
    files()
