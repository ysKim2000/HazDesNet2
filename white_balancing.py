import os
import cv2
import numpy as np
from skimage import morphology

I_DIR = './images/mrfid'
O_DIR = './wb_result/'

def white_balancing(img):
    res = img
    final = cv2.cvtColor(res, cv2.COLOR_BGR2LAB)

    avg_a = np.average(final[:, :, 1])
    avg_b = np.average(final[:, :, 2])

    for x in range(final.shape[0]):
        for y in range(final.shape[1]):
            l, a, b = final[x, y, :]
            l *= 100 / 255.0
            final[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
            final[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
    
    final = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)
    # 두개 같이 병합해서 나오기 
    # final = np.hstack((res, final))
    
    return final


def files(dir=I_DIR):
    names = os.listdir(dir)
    # 정렬화
    names.sort()
    for name in names:
        print(name)
        I = cv2.imread(os.path.join(dir, name))
        result = white_balancing(I)
        name = name.split('.')[0]
        cv2.imwrite(os.path.join(O_DIR, f'{name}.jpg'), result)

if __name__ == "__main__":
    # file('0001-0.jpg')
    files()
