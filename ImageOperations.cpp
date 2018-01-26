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

	//�ı�ͼƬ��С
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
	//��������ļ�
	unsigned char LUT[256];
	for (int i = 0; i < 256; i++)
	{
		//Gamma�任���ʽ
		LUT[i] = saturate_cast<uchar>(pow((float)(i / 255.0), kFactor)*255.0f);
	}
	Mat resultImage = image.clone();
	//����ͨ��Ϊ����ʱֱ�ӽ��б任
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
		//����ͨ��Ϊ3ʱ,���ÿ��ͨ���ֱ���б任
		MatIterator_<Vec3b> iterator = resultImage.begin<Vec3b>();
		MatIterator_<Vec3b> iteratorEnd = resultImage.end<Vec3b>();
		//ͨ�����ұ����ת��
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

	//ͼ���������ж�
	if(resultIamge.isContinuous())
	{
		nCols = nCols * nRows;
		nRows = 1;
	}
	//ͼ��ָ�����
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
	//�Աȶ�����ӳ��
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

cv::Mat ImageOperations::Graylayered(cv::Mat& image)
{
	Mat resultImage = image.clone();
	int nRows = resultImage.rows;
	int nCols = resultImage.cols;

	//ͼ������
	if(image.isContinuous())
	{
		nCols = nCols * nRows;
		nRows = 1;
	}
	//ͼ��ָ�����
	uchar* pDataMat;
	int controlMin = 150;
	int controlMax = 200;	
	//����ͼ��ĻҶȼ��ֲ�
	for (int j = 0; j < nRows; j++)
	{
		pDataMat = resultImage.ptr<uchar>(j);
		for (int i = 0; i < nCols; i++)
		{
			//��һ�ַ���:��ֵӳ��
			if(pDataMat[i]>controlMin)
			{
				pDataMat[i] = 255;
			}
			else
			{
				pDataMat[i] = 0;
			}
			////�ڶ��ַ���:����ӳ��
			//if(pDataMat[i]>controlMin&&
			//	pDataMat[i]<controlMax)
			//{
			//	pDataMat[i] = controlMax;
			//}
		}
	}
	return resultImage;
}
