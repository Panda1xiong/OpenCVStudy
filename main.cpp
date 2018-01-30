﻿#include "ImageOperations.h"
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{

    string imageDir = "D:\\xsl\\Codes\\ImageOcr\\Images\\Images\\";
    string imageFile = imageDir + "2(2).jpg";
//	string imageFile = imageDir + "2.jpg";

    Mat srcImage = imread(imageFile.c_str());
    if (!srcImage.data)
    {
        return 1;
    }
    imshow("srcImage:", srcImage);

    Mat grayMat;
    cvtColor(srcImage, grayMat, CV_RGB2GRAY);
    imshow("grayimage", grayMat);

    ImageOperations imOperation;
//    imOperation.MoiveImage(srcImage);
//    imshow("contrastStetch", imOperation.contrastStetch(grayMat));
//    imshow("graylayered", imOperation.Graylayered(grayMat))
//    imshow("result", imOperation.MaxEntropySegMentatio(grayMat));
//    imshow("result", imOperation.NNeighbourInterpolation(srcImage));
//    imOperation.Pyramid(srcImage);
//    imshow("result", imOperation.Filter2D_(srcImage));
//	  imshow("resultImage", imOperation.DFT(srcImage));

    grayMat.convertTo(grayMat, CV_32F);
    //定义卷积核算子
    Mat kernel = (Mat_<float>(3, 3) << 1, 1, 1,
            1, 1, 1,
            1, 1, 1);
    Mat resultImage;
    imOperation.Convolution(grayMat, kernel, resultImage);
    //归一化结果输出
    normalize(resultImage, resultImage, 0, 1, CV_MINMAX);
    imshow("result", resultImage);

//    Mat grayMat;
//    cvtColor(srcImage, grayMat, CV_RGB2GRAY);
//    imshow("grayImage", grayMat);

//    //旋转中心
//    Point2f center = Point2f(srcImage.cols / 2, srcImage.rows / 2);
//    //旋转角度
//    double angle = 60;
//    //缩放尺度
//    double scale = 0.5;
//    Mat rotateImage = getRotationMatrix2D(center, angle, scale);
//    //仿射变换
//    Mat rotateImg;
//    warpAffine(srcImage, rotateImg, rotateImage, srcImage.size());

//    imshow("srcImage",srcImage);
//    imshow("rotateImg", rotateImg);
    waitKey(0);
    return 0;
}







