import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

#out = cv2.VideoWriter('GrayEqualized.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,480))

#to plot the histogram. This function is not ussed in actual output but was used for visualization
def histplt(imd):
    lisy = list()
    for i in imd:
        for j in i:
            lisy.append(j)
    bins = 255
    n, bins, patches = plt.hist(lisy, bins, edgecolor='k', facecolor='blue', alpha=0.5)
    plt.show()


#computes the CDF of one frame and returns the CDF values in a list
def CDF(hisarr, a, mx):
    cdflis = list()
    for i in range(256):
        sum1 = np.sum(hisarr[0:(i+1),1])
        CDF = sum1/(width * height)
        cdflis.append(round(CDF*255))
    #print("cdf from function: ",CDF)
    return cdflis


#computes the histogram of one frame and returns the values in a numpy array
def histog(img):
    his = list()
    i = 0
    def recur(i):                           #computes the histogram by recurring the functon call in range 0 to 256
        coor = np.where(img == i)
        his.append([i,len(coor[0])])

        i += 1
        if i == 256:
            return
        return recur(i)
    if i == 0:
        recur(i)
    hisarr = np.array(his)
    return hisarr, i


#equalization in the RGB color space
def equalizationRGB(img12):

    img2 = img12.copy()

    #extracting the BGR channels
    bl = img2[:,:,0]
    gr = img2[:,:,1]
    re = img2[:,:,2]

    #computes the histogram of the respective channels
    fbl, maxb = histog(bl)
    fgr, maxg = histog(gr)
    fre, maxr = histog(re)

    #CDF for B channel
    cf = CDF(fbl, 5, maxr)
    cf = np.array(cf)
    img2[:,:,0] = cf[bl]        #assigning new values computed from the CDF

    #CDF for G channel
    cf = CDF(fgr, 5, maxr)
    cf = np.array(cf)
    img2[:,:,1] = cf[gr]        #assigning new values computed from the CDF

    #CDF for R channel
    cf = CDF(fre, 5, maxr)
    cf = np.array(cf)
    img2[:,:,2] = cf[re]        #assigning new values computed from the CDF

    return img2


def equalizationGRAY(img12):
    img2 = img12.copy()

    fre, maxr = histog(img2)
    cf = CDF(fre, 5, maxr)
    cf = np.array(cf)
    img2[:,:] = cf[img12[:,:]]      #assigning new values computed from the CDF

    return img2


#equalization in the HSV color space
def equalizationHSV(img12):
    img2 = img12.copy()
    #sh = img2.shape
    #img3 = np.zeros((sh[0],sh[1],sh[2]))

    re = img2[:,:,2]                #extracting the V channel values of the image

    fre, maxr = histog(re)         #computes the histogram of the V channel
    cf = CDF(fre, 5, maxr)         #return the CDF values for each pixel
    cf = np.array(cf)
    img2[:,:,2] = cf[re]            #assigning new pixel values to the V channel

    img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)

    return img2


### main body of the program
#reads the video and and calls the respective equalization function
vid = cv2.VideoCapture("Night Drive - 2689.mp4")
count = 0
while(True):
    r, frame = vid.read()
    if frame is None:
        break
    width = 640
    height = 480
    img1 = cv2.resize(frame, (width,height), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)

    count += 1
    cv2.imshow("video",img1)
    cv2.waitKey(1)
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #histplt(gray)
    #histplt(hsv[:,:,2])
    #final_img = equalizationRGB(img1)
    final_img = equalizationHSV(hsv)
    #final_img = equalizationGRAY(gray)
    #histplt(final_img)


    #histplt(final_img[:,:,1])


    cv2.imshow("equalized",final_img)

    #out.write(final_img)
    cv2.waitKey(1)

#out.release()
vid.release()
cv2.destroyAllWindows()
