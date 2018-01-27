#ifndef IMAGEOPERATIONS_H
#define IMAGEOPERATIONS_H
#include <opencv2/opencv.hpp>

class ImageOperations
{
public:
	ImageOperations();
	~ImageOperations();

public:
    //图片缩放
	void ZoomImage(cv::Mat& image);

    //图片移动
	void MoiveImage(cv::Mat& image);

    //Camma矫正
	cv::Mat GammaTransform(cv::Mat& image, float kFactor);

    //分段灰度拉伸
	cv::Mat contrastStetch(cv::Mat& grayImage);

    //灰度分层
	cv::Mat Graylayered(cv::Mat& grayimage);
};
#endif

