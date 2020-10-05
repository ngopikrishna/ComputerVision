import argparse
import cv2
import numpy as np 


imgBlack = np.array([
                [  0,  0,  0,  0,  0,  0,  0,  0],
                [  0,  0,  0,  0,  0,  0,  0,  0],
                [  0,  0,  0,  0,  0,  0,  0,  0],
                [  0,  0,  0,  0,  0,  0,  0,  0],
                [  0,  0,  0,  0,  0,  0,  0,  0],
                [  0,  0,  0,  0,  0,  0,  0,  0],
                [  0,  0,  0,  0,  0,  0,  0,  0],
                [  0,  0,  0,  0,  0,  0,  0,  0],
])


imgDot = np.array([
                [255,255,255,255,255,255,255,255],
                [255,255,255,255,255,255,255,255],
                [255,255,  0,  0,  0,  0,255,255],
                [255,255,  0,  0,  0,  0,255,255],
                [255,255,  0,  0,  0,  0,255,255],
                [255,255,  0,  0,  0,  0,255,255],
                [255,255,255,255,255,255,255,255],
                [255,255,255,255,255,255,255,255],
])

imgPlus = np.array([
                [255,255,255,  0,  0,255,255,255],
                [255,255,255,  0,  0,255,255,255],
                [255,255,255,  0,  0,255,255,255],
                [  0,  0,  0,  0,  0,  0,  0,  0],
                [  0,  0,  0,  0,  0,  0,  0,  0],
                [255,255,255,  0,  0,255,255,255],
                [255,255,255,  0,  0,255,255,255],
                [255,255,255,  0,  0,255,255,255],
])

imgWhite = np.array([
                [255,255,255,255,255,255,255,255],
                [255,255,255,255,255,255,255,255],
                [255,255,255,255,255,255,255,255],
                [255,255,255,255,255,255,255,255],
                [255,255,255,255,255,255,255,255],
                [255,255,255,255,255,255,255,255],
                [255,255,255,255,255,255,255,255],
                [255,255,255,255,255,255,255,255],
])




def Image2Text(imgOriginalGrayScale):    

    img = cv2.resize(imgOriginalGrayScale, (0,0), fx=0.125, fy=0.125)
    q1 = np.quantile(img, 0.25)
    q2 = np.quantile(img, 0.5)
    q3 = np.quantile(img, 0.75)
    

    nRows, nCols = img.shape
    print(img.shape)
    txtArray = []
    for inx in range(nRows):
        strLine = ''
        for jnx in range(nCols):
            if img[inx][jnx] > q3:
                strLine += " "
            elif q2<=img[inx][jnx] and img[inx][jnx]<=q3:
                strLine += "."
            elif q1<=img[inx][jnx] and img[inx][jnx]<=q2:
                strLine += "+"            
            elif img[inx][jnx]<q1:
                strLine += "*"
            else:
                strLine +=" "
                print("q1:{},     q2:{},    q3:{},   pix:{}".format(q1,q2,q3,img[inx][jnx]))
        # strLine += "\n"
        txtArray.append(strLine)

    return txtArray



def CopyImage(imgTarget, imgSource, nStartRow, nStartCol):
    rows,cols = imgSource.shape

    for inx in range(rows):
        for jnx in range(cols):
            imgTarget[nStartRow+inx][nStartCol+jnx]=imgSource[inx][jnx]

    return


def DottifyImage(imgOriginal):
    txtArray = Image2Text(imgOriginal)

    nRows = len(txtArray)
    nCols = len(txtArray[0])
    

    imgDotted = np.empty((nRows*8, nCols*8), dtype=np.uint8)
    

    for inx in range(nRows):
        # print(inx, len(txtArray[inx]))
        
        for jnx in range(nCols):
            nStartRow = inx*8
            nStartCol = jnx*8
            if txtArray[inx][jnx] == "*":
                CopyImage(imgDotted, imgBlack, nStartRow, nStartCol)
            if txtArray[inx][jnx] == "+":
                CopyImage(imgDotted, imgPlus, nStartRow, nStartCol)
            if txtArray[inx][jnx] == ".":
                CopyImage(imgDotted, imgDot, nStartRow, nStartCol)
            if txtArray[inx][jnx] == " ":
                CopyImage(imgDotted, imgWhite, nStartRow, nStartCol)
    
    return imgDotted



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to the image to be pointili-fied")
    args = parser.parse_args()


    imgOriginal = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    print(imgOriginal.shape)

    imgDotted = DottifyImage(imgOriginal)



    cv2.namedWindow("Dottified", cv2.WINDOW_NORMAL)
    cv2.imshow("Dottified", imgDotted)
    cv2.waitKey(0)
    
    