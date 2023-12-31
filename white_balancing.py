import os
import cv2
import numpy as np
from skimage import morphology
#레퍼런스
# https://medium.com/@er_95882/colour-vision-lands-experiments-with-colour-constancy-white-balance-and-examples-in-python-93a71d0c4cbe

I_DIR = './images/mrfid'
# 종류에 맞게 
O_T_DIR = './wb_result/'
O_GS_DIR = './gs_result/'
O_MPR_DIR = './mpr_result/'
O_R_DIR = './r_result/'


# 화이트 밸런싱 알고리즘 종류 

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


def grey_world(image):
    image = image / 255.

    pWhite = 0.05
    
    # In OpenCV the channels order is Blue-Green-Red
    red = image[:, :, 2]
    green = image[:, :, 1]
    blue = image[:, :, 0]

    red = red / np.mean(red)
    green = green / np.mean(green)
    blue = blue / np.mean(blue)

    red_sorted = sorted(red.ravel())
    green_sorted = sorted(green.ravel())
    blue_sorted = sorted(blue.ravel())

    total = len(red_sorted)

    max_index = int(total * (1. - pWhite))
    
    image[:, :, 2] = red / red_sorted[max_index]
    image[:, :, 1] = green / green_sorted[max_index]
    image[:, :, 0] = blue / blue_sorted[max_index]

    return image * 255

def max_patch_retinex(image):
    image = image / 255.

    p_white = 0.04
    p_black = 0.0001

    # In OpenCV the channels order is Blue-Green-Red
    red = image[:, :, 2]
    green = image[:, :, 1]
    blue = image[:, :, 0]

    red_sorted = sorted(red.ravel())
    green_sorted = sorted(green.ravel())
    blue_sorted = sorted(blue.ravel())

    total = len(red_sorted)
    max_index = int(total * (1. - p_white))
    min_index = int(total * p_black)

    max_red = red_sorted[max_index]
    max_green = green_sorted[max_index]
    max_blue = blue_sorted[max_index]
    min_red = red_sorted[min_index]
    min_green = green_sorted[min_index]
    min_blue = blue_sorted[min_index]


    red = (red - min_red / (max_red - min_red))
    green = (green - min_green / (max_green - min_green))
    blue = (blue - min_blue / (max_blue - min_blue))
    

    image[:, :, 2] = red
    image[:, :, 1] = green
    image[:, :, 0] = blue
    return image * 255

def retinex(image):
    r = image[:, :, 2]
    g = image[:, :, 1]
    b = image[:, :, 0]

    max_red = r.max()
    max_green = g.max()
    max_blue = b.max()

    image[:, :, 0] = np.minimum(b * (max_green / max_blue), 255)
    image[:, :, 2] = np.minimum(r * (max_green / max_red), 255)

    return image

def files(dir=I_DIR):
    names = os.listdir(dir)
    # 정렬화
    names.sort()
    for name in names:
        print(name)
        I = cv2.imread(os.path.join(dir, name))
        
        result_wb, result_gs, result_mpr, result_r  = white_balancing(I), grey_world(I), max_patch_retinex(I), retinex(I)

        name = name.split('.')[0]
        cv2.imwrite(os.path.join(O_T_DIR, f'{name}_wb.jpg'), result_wb)
        cv2.imwrite(os.path.join(O_GS_DIR, f'{name}_gs.jpg'), result_gs)
        cv2.imwrite(os.path.join(O_MPR_DIR, f'{name}_mpr.jpg'), result_mpr)
        cv2.imwrite(os.path.join(O_R_DIR, f'{name}_r.jpg'), result_r)



if __name__ == "__main__":
    # file('0001-0.jpg')
    files()
