# Histogram-Equalization

## Use the link below to download the data
https://drive.google.com/drive/folders/1Mh7s8fDo0LdvLaiVOPu4grDvvQVMqpCy?usp=sharing

## Instructions:
Make sure to download the data required to run the code and save it in the same directory as that of the code.


## Libraries Required
* Numpy
* OpenCV
* Matplotlib
* imutils

## Notes:
1) The three channels BGR are then separated, and the histogram of each individual channel is computed separately.
2) The histogram values are used to compute the cumulative distributed function (CDF) and thus the new value of each pixel is obtained from the CDF.
3) For each pixel value the CDF is computed as the histogram of all the pixel values less than and equal to the current pixel value divided by the total number of pixels.
4) This method works fine in the case of a grayscale image. For a BGR image the values of each individual channel are redistributed separately thus causing colors to mix up in the resulting image.
5) In order to resolve this issue, the image is converted to the HSV color space and then the histogram equalization is applied only to the V channel of the image. This results in a better image quality.
