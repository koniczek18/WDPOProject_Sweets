import json
from pathlib import Path
from typing import Dict
import numpy as np
import click
import cv2
from tqdm import tqdm


#IMAGE PREPROCESSING
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

def colorIsolation(hsv_image,kernel_erode,lower_Green,upper_Green,lowerRed,upperRed,lowerYellow,upperYellow,lowerPurple,upperPurple):
    # GREEN
    greenMask = cv2.inRange(hsv_image, lower_Green, upper_Green)
    GreenMasked = cv2.bitwise_and(hsv_image, hsv_image, mask=greenMask)
    Green = cv2.erode(GreenMasked, kernel_erode, iterations=1)
    resultGreen = cv2.medianBlur(Green, 7)
    hsv_image-=GreenMasked
    # YELLOW
    yellowMask = cv2.inRange(hsv_image, lowerYellow, upperYellow)
    YellowMasked=cv2.bitwise_and(hsv_image,hsv_image,mask=yellowMask)
    Yellow = cv2.erode(YellowMasked,kernel_erode,iterations=1)
    resultYellow = cv2.medianBlur(Yellow, 7)
    hsv_image-=YellowMasked
    # PURPLE
    purpleMask = cv2.inRange(hsv_image, lowerPurple, upperPurple)
    PurpleMasked=cv2.bitwise_and(hsv_image,hsv_image,mask=purpleMask)
    Purple=cv2.erode(PurpleMasked,kernel_erode,iterations=1)
    resultPurple=cv2.medianBlur(Purple,7)
    #RED
    redMask = cv2.inRange(hsv_image, lowerRed, upperRed)
    RedMasked = cv2.bitwise_and(hsv_image, hsv_image, mask=redMask)
    Red = cv2.erode(RedMasked, kernel_erode, iterations=1)
    resultRed = cv2.medianBlur(Red, 7)
    #RETURN
    return greenMask,resultGreen,yellowMask,resultYellow,redMask,resultRed,purpleMask,resultPurple


def colorValidation(color, lowerTH, upperTH ):
    if color[0][0][0] == color[0][0][1] == color[0][0][2] == 0:
        if not lowerTH[0] < color[0][0][0] < upperTH[0]:
            if not lowerTH[1] < color[0][0][1] < upperTH[1]:
                if not lowerTH[2] < color[0][0][2] < upperTH[2]:
                    return False
    return True

def aqColor(image, x, y):
    B = image[x, y][0]
    G = image[x, y][1]
    R = image[x, y][2]
    return cv2.cvtColor(np.uint8([[[B, G, R]]]), cv2.COLOR_BGR2HSV)

def findCircles(mask, isoColor, low, high, dp, minDist, param_1, param_2, minRadius, maxRadius, rangeSize):
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp, minDist, param1=param_1, param2=param_2, minRadius=minRadius, maxRadius=maxRadius)
    realCircles=[]
    try:
        circles = np.uint16(np.around(circles))
        inRange = False
        rangeSize = 3
        minusCounter = 0
        for i in circles[0, :]:
            color = aqColor(isoColor, i[1], i[0])
            inRange = colorValidation(color, low, high)
            for z in range(-rangeSize, rangeSize):
                for k in range(-rangeSize, rangeSize):
                    inRange = colorValidation(color, low, high)
                    color = aqColor(isoColor, i[1] + k, i[0] + z)
                    if inRange:
                        break
                if inRange:
                    break
            if not inRange:
                minusCounter += 1
        ### visualisation debuging ###
        #   if inRange:
        #       cv2.circle(isoColor, (i[0], i[1]), i[2], (0, 0, 255), 1)
        #       cv2.circle(isoColor, (i[0], i[1]), 2, (0, 0, 255), 1)
        # cv2.imshow('circles', isoColor)
        result = len(circles[0]) - minusCounter
        return result
    except:
        result=0
        return result

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
    #INITIALISATION
    ##VARIABLES
    minDist = 30
    minRadius = 2
    maxRadius = 18
    param_1 = 200
    param_2 = 2.5
    dp = 1
    rangeSize = 3
    ## MASKS
    ### GREEN
    lowerGreen = np.array([30, 150, 0])
    upperGreen = np.array([70, 255, 255])
    ### YELLOW
    lowerYellow = np.array([0, 120, 135])
    upperYellow = np.array([30, 255, 255])
    ### RED
    lowerRed = np.array([160, 40, 130])
    upperRed = np.array([200, 255, 255])
    ### PURPLE
    lowerPurple = np.array([155, 60, 0])
    upperPurple = np.array([180, 255, 255])
    ## KERNELS
    kernel_erode = np.ones((3, 3), np.uint8)
    #END INITIALISATION
    #CODE
    ogImage= cv2.imread(img_path, cv2.IMREAD_COLOR)
    resized, blured, hsv, brightenedHSV, brightenedBGR=imageInitialProcessing(ogImage,30)
    greenMask,resultGreen,yellowMask,resultYellow,redMask,resultRed,purpleMask,resultPurple = colorIsolation(brightenedHSV,kernel_erode,lowerGreen,upperGreen,lowerRed,upperRed,lowerYellow,upperYellow,lowerPurple,upperPurple)
    #END CODE
    #ANSWER
    redAns = findCircles(redMask, resultRed, lowerRed, upperRed,dp,minDist,param_1,param_2,minRadius,maxRadius,rangeSize)
    yellowAns = findCircles(yellowMask, resultYellow, lowerYellow, upperYellow,dp,minDist,param_1,param_2,minRadius,maxRadius,rangeSize)
    greenAns = findCircles(greenMask, resultGreen, lowerGreen, upperGreen,dp,minDist,param_1,param_2,minRadius,maxRadius,rangeSize)
    purpleAns = findCircles(purpleMask, resultPurple, lowerPurple, upperPurple,dp,minDist,param_1,param_2,minRadius,maxRadius,rangeSize)
    #END ANSWER
    #DEBUG
    cv2.waitKey()
    #DEBUG END
    #RETURN
    return {'red': redAns, 'yellow': yellowAns, 'green': greenAns, 'purple': purpleAns}

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
