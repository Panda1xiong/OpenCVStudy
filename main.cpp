#include "ImageOperations.h"
#include <iostream>
#include <string>
using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
#ifdef _windows
    string imageDir = "D:\\xsl\\Codes\\ImageOcr\\Images\\Images\\";
#else
    string imageDir = "./";
#endif
//    string imageFile = imageDir + "2.jpg";
	string imageFile = imageDir + "1.jpg";

    Mat srcImage = imread(imageFile.c_str());
    if (!srcImage.data)
    {
        return 1;
    }
//    imshow("srcImage:", srcImage);

    Mat grayMat;
    cvtColor(srcImage, grayMat, CV_RGB2GRAY);
//    imshow("grayimage", grayMat);

    ImageOperations imOperation;
//    imOperation.MoiveImage(srcImage);
//    imshow("contrastStetch", imOperation.contrastStetch(grayMat));
//    imshow("graylayered", imOperation.Graylayered(grayMat))
//    imshow("result", imOperation.MaxEntropySegMentatio(grayMat));
//    imshow("result", imOperation.NNeighbourInterpolation(srcImage));
//    imOperation.Pyramid(srcImage);
//    imshow("result", imOperation.Filter2D_(srcImage));
//	  imshow("resultImage", imOperation.DFT(srcImage));
     imOperation.CorrectImageDirection(grayMat);


//    grayMat.convertTo(grayMat, CV_32F);
//    //定义卷积核算子
//    float m[3][3] ={{0,1,0},{1,0,1},{0,1,0}};
//    Mat kernel = Mat(3, 3, CV_32F, m) / 4;
//    Mat resultImage;
//    imOperation.Convolution(grayMat, kernel, resultImage);
//    //归一化结果输出
//    normalize(resultImage, resultImage, 0, 1, CV_MINMAX);
//    imshow("result", resultImage);

//    imshow("result", imOperation.AddSaltNoise(srcImage, 5000));
//    imshow("result", imOperation.AddGaussianNoise(srcImage));

//    Mat mBMat;
//    medianBlur(grayMat, mBMat, 3);
//    imshow("mbmat", mBMat);

//    Mat bFMat;
//    bilateralFilter(srcImage, bFMat, 7, 20.0, 2.0);
//    imshow("bFMatresult", bFMat);


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







