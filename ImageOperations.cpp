#include "ImageOperations.h"
#include <iostream>

using namespace std;
using namespace cv;


ImageOperations::ImageOperations()
{
}


ImageOperations::~ImageOperations()
{
}

void ImageOperations::ZoomImage(cv::Mat& image)
{

}

void ImageOperations::MoiveImage(cv::Mat& image)
{
	int nRows = image.rows;
	int nCols = image.cols;
	
	int xOffset = 50;
	int yOffset = 50;

    //改变图片大小
	//nRows += abs(yOffset);
	//nCols += abs(xOffset);
	//Mat resultImage(nRows,nCols,image.type());

	Mat resultImage(image.size(), image.type());
	
	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{
			int x = j - xOffset;
			int y = i - yOffset;

			if (x >= 0 && y >= 0 && x < nCols && y < nRows)
			{
				resultImage.at<Vec3b>(i, j) = image.at<Vec3b > (y,x);
			}
		}

	}
	imshow("MovedImage", resultImage);

}

cv::Mat ImageOperations::GammaTransform(cv::Mat& image, float kFactor)
{
    //建立查表文件
	unsigned char LUT[256];
	for (int i = 0; i < 256; i++)
	{
        //Gamma变换表达式
		LUT[i] = saturate_cast<uchar>(pow((float)(i / 255.0), kFactor)*255.0f);
	}
	Mat resultImage = image.clone();
    //输入通道为单向时直接进行变换
	if(image.channels() == 1)
	{
		MatIterator_<uchar> iterator = resultImage.begin<uchar>();
		MatIterator_<uchar> iteratorEnd = resultImage.end<uchar>();
		for ( ; iterator != iteratorEnd; iterator++)
		{
			*iterator = LUT[(*iterator)];
		}
	}
	else
	{
        //输入通道为3时,需对每个通道分别进行变换
		MatIterator_<Vec3b> iterator = resultImage.begin<Vec3b>();
		MatIterator_<Vec3b> iteratorEnd = resultImage.end<Vec3b>();
        //通过查找表进行转换
		for ( ; iterator != iteratorEnd; iterator++)
		{
			(*iterator)[0] = LUT[(*iterator)[0]];
			(*iterator)[1] = LUT[(*iterator)[1]];
			(*iterator)[2] = LUT[(*iterator)[2]];
		}
	}
	return resultImage;
}

cv::Mat ImageOperations::contrastStetch(cv::Mat& grayImage)
{
	Mat resultIamge = grayImage.clone();
	int nRows = resultIamge.rows;
	int nCols = resultIamge.cols;

    //图像连续性判断
	if(resultIamge.isContinuous())
	{
		nCols = nCols * nRows;
		nRows = 1;
	}
    //图像指针操作
	uchar *pDataMat;
	int pixMax = 0, pixMin = 255;
	for (int j = 0; j < nRows; j++)
	{
		pDataMat = resultIamge.ptr<uchar>(j);
		for (int i = 0; i < nCols; i++)
		{
			if (pDataMat[i]>pixMax)
			{
				pixMax = pDataMat[i];
			}
			if (pDataMat[i]<pixMin)
			{
				pixMin = pDataMat[i];
			}
		}
	}
    //对比度拉伸映射
	for (int j = 0; j < nRows; j++)
	{
		pDataMat = resultIamge.ptr<uchar>(j);
		for (int i = 0; i < nCols; i++)
		{
			pDataMat[i] = (pDataMat[i] - pixMin) * 255 / (pixMax - pixMin);
		}
	}
	return resultIamge;
}

cv::Mat ImageOperations::Graylayered(cv::Mat& grayimage)
{
	Mat resultImage = grayimage.clone();
	int nRows = resultImage.rows;
	int nCols = resultImage.cols;

    //图像连续
	if(grayimage.isContinuous())
	{
		nCols = nCols * nRows;
		nRows = 1;
	}
    //图像指针操作
	uchar* pDataMat;
	int controlMin = 150;
    int controlMax = 200;
    //计算图像的灰度级分层
	for (int j = 0; j < nRows; j++)
	{
		pDataMat = resultImage.ptr<uchar>(j);
		for (int i = 0; i < nCols; i++)
		{
            //第一种方法:二值映射
			if(pDataMat[i]>controlMin)
			{
				pDataMat[i] = 255;
			}
			else
			{
				pDataMat[i] = 0;
			}
            ////第二种方法:区域映射
			//if(pDataMat[i]>controlMin&&
			//	pDataMat[i]<controlMax)
			//{
			//	pDataMat[i] = controlMax;
			//}
		}
	}
	return resultImage;
}
