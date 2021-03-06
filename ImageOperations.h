﻿#ifndef IMAGEOPERATIONS_H
#define IMAGEOPERATIONS_H
#include <opencv2/opencv.hpp>

extern cv::Mat g_srcImage,g_srcGray;
extern int g_thresh;
extern int g_max_thresh;
extern cv::RNG g_rng;

class ImageOperations
{
public:
	ImageOperations();
	~ImageOperations();
private:
    //计算当前的位置的能量熵
    float CaculateCurrentEntropy(cv::Mat& hit, int threshold);
    //高斯噪声
    double GenerateGaussianNoise(double mu, double sigma);
public:
	//图片缩放
	void ZoomImage(cv::Mat& image);
	//Camma矫正
	cv::Mat GammaTransform(cv::Mat& image, float kFactor);
	//图片移动
	void MoiveImage(cv::Mat& image);
	//分段灰度拉伸
	cv::Mat contrastStetch(cv::Mat& grayImage);
	//灰度分层
	cv::Mat Graylayered(cv::Mat& image);
	//灰度比特平面
	void ShowMBitPlan(cv::Mat& image);
	//最大熵阈值分割
	cv::Mat MaxEntropySegMentatio(cv::Mat& inputImage);
	//最近邻插值图像缩放
	cv::Mat NNeighbourInterpolation(cv::Mat& image);
	//图像金字塔
	void Pyramid(cv::Mat& image);
	//图像掩码操作
	cv::Mat Filter2D_(cv::Mat& image);
	//图像傅里叶变换
	cv::Mat DFT(cv::Mat& image);
	//图像卷积操作
    void Convolution(cv::Mat& graySrc,cv::Mat kernel,cv::Mat& dst);
	//添加椒盐噪声
    cv::Mat AddSaltNoise(cv::Mat& srcImage, int n);
	//添加高斯噪声
    cv::Mat AddGaussianNoise(cv::Mat& srcImage);
	//矫正旋转文本
	cv::Mat CorrectImageDirection(cv::Mat& srcImage);
	//凸包及缺陷检测
	static void thresh_callback(int,void*);
	//轮廓矩
	void CacMoments(cv::Mat& src);
	//canny边缘检测
	void CannyEdgeDetection(cv::Mat& src);
	//点多边形测试
	
};
#endif

