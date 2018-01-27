#include "ImageOperations.h"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    ImageOperations imageOperations;

    Mat srcImage = imread("1.jpg");

    //转化为灰度图
    Mat grayImage;
    cvtColor(srcImage, grayImage, CV_RGB2GRAY);

    imshow("srcImage", srcImage);
    imshow("resultImage", imageOperations.Graylayered(srcImage));

    waitKey(0);
    return 0;
}