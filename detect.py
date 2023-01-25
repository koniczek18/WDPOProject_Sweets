import json
from pathlib import Path
from typing import Dict
import numpy as np

import click
import cv2
from tqdm import tqdm

#IMAGE PROCESSING
def colorIsolation(hsv_image):
    ''' Function to isolate green, red, yellow and purple colors (sweets) form hsv image
    Parameters
    ----------
    hsv_image

    Returns
    -------
    resultGreen,resultRed,resultYellow,resultPurple
        isolated images (G,R,Y,P)
    '''
    #MASKS
    lower_Green = np.array([30, 150, 0])
    upper_Green = np.array([70, 255, 255])
    lower_Purple = np.array([60, 30, 0])
    upper_Purple = np.array([170, 255, 255])
    lower_Yellow = np.array([0, 160, 120])
    upper_Yellow = np.array([30, 255, 255])
    lower_Red = np.array([70, 37, 110])
    upper_Red = np.array([200, 255, 255])
    #KERNELS
    kernel_erode = np.ones((3, 3), np.uint8)
    #RED
    Red_mask = cv2.inRange(hsv_image, lower_Red, upper_Red)
    RedMasked = cv2.bitwise_and(hsv_image, hsv_image, mask=Red_mask)
    Red=cv2.erode(RedMasked, kernel_erode, iterations=1)
    resultRed = cv2.medianBlur(Red, 7)
    #GREEN
    Green_mask = cv2.inRange(hsv_image, lower_Green, upper_Green)
    GreenMasked = cv2.bitwise_and(hsv_image, hsv_image, mask=Green_mask)
    Green = cv2.erode(GreenMasked, kernel_erode, iterations=1)
    resultGreen = cv2.medianBlur(Green, 7)
    #YELLOW
    Yellow_mask = cv2.inRange(hsv_image, lower_Yellow, upper_Yellow)
    YellowMasked = cv2.bitwise_and(hsv_image, hsv_image, mask=Yellow_mask)
    Yellow = cv2.erode(YellowMasked, kernel_erode, iterations=1)
    resultYellow = cv2.medianBlur(Yellow, 7)
    #PURPLE
    Purple_mask = cv2.inRange(hsv_image, lower_Purple, upper_Purple)
    PurpleMasked = cv2.bitwise_and(hsv_image, hsv_image, mask=Purple_mask)
    Purple = cv2.erode(PurpleMasked, kernel_erode, iterations=1)
    resultPurple=cv2.medianBlur(Purple, 7)
    #RETURN
    return resultGreen,resultRed,resultYellow,resultPurple

def imageInitialProcessing(image,brightnessBump=30):
    #RESIZE
    dim=(720,int((image.shape[0])*720/image.shape[1]))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    #BLUR
    blured=cv2.medianBlur(resized,5)
    #HSV
    hsv = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)
    #BRIGHT
    limit=255-brightnessBump
    h,s,v=cv2.split(hsv)
    v[v>limit]=255
    v[v<=limit]+=brightnessBump
    brightenedHSV=cv2.merge((h, s, v))
    brightenedBGR = cv2.cvtColor(brightenedHSV, cv2.COLOR_HSV2BGR)
    return resized,blured,hsv,brightenedHSV,brightenedBGR

def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.
    ###
    Parameters
    ----------
    img_path : str
        Path to processed image.
    ###
    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """

    #CODE
    ogImage= cv2.imread(img_path, cv2.IMREAD_COLOR)
    resized, blured, hsv, brightenedHSV, brightenedBGR=imageInitialProcessing(ogImage,30)
    resultGreen, resultRed, resultYellow, resultPurple=colorIsolation(brightenedHSV)
    hori1 = np.concatenate((resultRed, blured,resultPurple), axis=1)
    hori2 = np.concatenate((resultYellow, blured, resultGreen), axis=1)
    cv2.imshow('R _ P',hori1)
    cv2.imshow('Y _ G', hori2)
    cv2.waitKey()

    #END CODE
    #TODO: PROVIDE ANSWEAR
    red = 0
    yellow = 0
    green = 0
    purple = 0
    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
