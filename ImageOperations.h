#ifndef IMAGEOPERATIONS_H
#define IMAGEOPERATIONS_H
#include <opencv2/opencv.hpp>

class ImageOperations
{
public:
	ImageOperations();
	~ImageOperations();

public:
	//ͼƬ����
	void ZoomImage(cv::Mat& image);
	//ͼƬ�ƶ�
	void MoiveImage(cv::Mat& image);
	//Camma����
	cv::Mat GammaTransform(cv::Mat& image, float kFactor);
	//�ֶλҶ�����
	cv::Mat contrastStetch(cv::Mat& grayImage);
	//�Ҷȷֲ�
	cv::Mat Graylayered(cv::Mat& image);
};
#endif

