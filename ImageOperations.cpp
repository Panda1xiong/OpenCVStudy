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

float ImageOperations::CaculateCurrentEntropy(cv::Mat& hist, int threshold)
{
	float BackgroundSum = 0, targetSum = 0;
	const float* pDataHist = (float*)hist.ptr<float>(0);
	for (int i = 0; i < 256; i++)
	{
		//累计背景值
		if (i< threshold)
		{
			BackgroundSum += pDataHist[i];
		}
		else
		{
			targetSum += pDataHist[i];
		}
	}
	//std::cout << BackgroundSum << " " << targetSum << std::endl;
	float BackgroundEntropy = 0, targetEntropy = 0;
	for (int i = 0; i < 256; i++)
	{
		//计算背景熵
		if (i < threshold)
		{
			if (pDataHist[i] == 0)
			{
				continue;
			}
			float ratiol = pDataHist[i] / BackgroundSum;
			//计算当前能量熵
			BackgroundEntropy += -ratiol* logf(ratiol);
		}else //计算目标熵
		{
			if (pDataHist[i]==0)
			{
				continue;
			}
			float ratio2 = pDataHist[i] / targetSum;
			targetEntropy += -ratio2 * logf(ratio2);
		}
	}
	return (targetEntropy + BackgroundEntropy);
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

cv::Mat ImageOperations::Graylayered(cv::Mat& image)
{
	Mat resultImage = image.clone();
	int nRows = resultImage.rows;
	int nCols = resultImage.cols;

	//图像连续
	if(image.isContinuous())
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

void ImageOperations::ShowMBitPlan(cv::Mat& image)
{
	int nRows = image.rows;
	int nCols = image.cols;
	//图像连续性判断
	if (image.isContinuous())
	{
		nCols = nCols * nRows;
		nRows = 1;
	}
	//图像指针操作
	uchar*  pSrcMat;
	uchar* pResultMat;
	Mat  resultImage = image.clone();
	int pixMax = 0, pixMin = 0;
	for (int n = 0; n < 8; n++)
	{
		//比特平面分层 像素构成
		pixMin = (int)pow(2, n - 1);
		pixMax = (int)pow(2, n);
		for (int j = 0;j< nRows;j++)
		{
			pSrcMat = image.ptr<uchar>(j);
			pResultMat = resultImage.ptr<uchar>(j);
			for (int i = 0; i < nCols; i++)
			{
				//相应比特平面二值化
				if (pSrcMat[i]>=pixMin&& pSrcMat[i]<pixMax)
				{
					pResultMat[i] = 255;
				}
				else
				{
					pResultMat[i] = 0;
				}
			}
		}
		//比特平面输出
		char windowName[20] = { 0 };
		sprintf(windowName, "BitPlane %d", n);
		imshow(windowName, resultImage);
	}

}

cv::Mat ImageOperations::MaxEntropySegMentatio(cv::Mat& inputImage)
{
	//初始化直方图参数
	const int channels[1] = { 0 };
	const int histSize[1] = { 256 };
	float pranges[2] = { 0,256 };
	const float* ranges[1] = { pranges };
	Mat hist;
	//计算直方图
	calcHist(&inputImage, 1, channels, Mat(), hist, 1, histSize, ranges);
	float maxentropy = 0;
	int max_index = 0;
	Mat result;
	//遍历得到最大熵阈值分割的最佳阈值
	for (int i = 0; i < 256; i++)
	{
		float cur_entropy = CaculateCurrentEntropy(hist, i);
		//计算当前最大值的位置
		if (cur_entropy > maxentropy)
		{
			maxentropy = cur_entropy;
			max_index = i;
		}
	}
	//二值化分割
	threshold(inputImage, result, max_index, 255, CV_THRESH_BINARY);
	return result;
}

cv::Mat ImageOperations::NNeighbourInterpolation(cv::Mat& image)
{
	//判断输入有效性
	CV_Assert(image.data != NULL);
	int rows = image.rows;
	int cols = image.cols;
	//构建目标图像
	Mat dstImage = Mat(Size(150, 150), image.type(),Scalar::all(0));
	int dstRows = dstImage.rows;
	int dstCols = dstImage.cols;
	//坐标转换,求取缩放倍数
	float cx = (float)cols / dstCols;
	float ry = (float)rows / dstRows;
	cout << "cx:" << cx << " ry:" << ry << endl;
	//遍历图像,完成缩放操作
	for (int  i = 0; i < dstCols; i++)
	{
		//取整,获取目标图像在原图像对应坐标
		int ix = cvFloor(i * cx);
		for (int j = 0; j < dstRows; j++)
		{
			int jy = cvFloor(j * ry);
			//边界处理，防止指针越界
			if (ix > cols - 1)
			{
				ix = cols - 1;
			}
			if (jy > rows -1)
			{
				jy = rows - 1;
			}
			//映射矩阵
			dstImage.at<Vec3b>(j, i) = image.at<Vec3b>(jy, ix);
		}
	}
	return dstImage;
}

void ImageOperations::Pyramid(cv::Mat& image)
{
	//根据尺寸判断是否需要缩放
	if (image.rows> 400&& image.cols>400)
	{
		resize(image, image, Size(), 0.5, 0.5);
	}
	else
	{
		//不需要进行缩放
		resize(image, image, Size(), 1, 1);
	}
	imshow("srcImage", image);
	Mat pyrDownImage, pyrUpImage;
	//下采样过程
	pyrDown(image, pyrDownImage, Size(image.cols / 2, image.rows / 2));
	imshow("pyrDown", pyrDownImage);
	//上采样过程
	pyrUp(image, pyrUpImage, Size(image.cols * 2, image.rows * 2));
	imshow("pyrUp", pyrUpImage);
	//对下采样进行重构
	Mat pyrBuildImage;
	pyrUp(pyrDownImage, pyrBuildImage, Size(pyrDownImage.cols * 2, pyrDownImage.rows * 2));
	imshow("pyrBuild", pyrBuildImage);
	//比较重构后的性能
	Mat diffImage;
	absdiff(image, pyrBuildImage, diffImage);
	imshow("diffImage", diffImage);
	waitKey(0);

}

cv::Mat ImageOperations::Filter2D_(cv::Mat& image)
{
	Mat resultImage(image.size(), image.type());
	//构造核函数因子
    float m[3][3] ={{0,1,0},{1,0,1},{0,1,0}};
    Mat kern = Mat(3, 3, CV_32F, m) / 4;
    cout << "kern:" << kern << endl;
    filter2D(image, resultImage, image.depth(), kern);
	return resultImage;
}

cv::Mat ImageOperations::DFT(cv::Mat& image)
{
	cv::Mat srcGray;
	cvtColor(image, srcGray, CV_RGB2GRAY);
	//将输入图像延扩到最佳的尺寸
	int nRows = getOptimalDFTSize(srcGray.rows);
	int nCols = getOptimalDFTSize(srcGray.cols);
	Mat resultImage;
	//把灰度图像放在左上角,向右边和下边扩展图像
	//将添加的像素初始化为0 [Scalar::all(0)]
	copyMakeBorder(srcGray, resultImage, 0, nRows - srcGray.rows,
		0, nCols - srcGray.cols, BORDER_CONSTANT, Scalar::all(0));


    //	为傅里叶变换的结果(实部和虚部)分配存储空间
	Mat planes[] = { Mat_<float>(resultImage),Mat::zeros(resultImage.size(), CV_32F) };
	Mat completeI;
	//为延扩后的图像增添一个初始化为0的通道
	merge(planes, 2, completeI);
	//进行离散傅里叶变换
	dft(completeI, completeI);
	//将复数转换为幅度
	split(completeI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat dftResultImage = planes[0];
	//对数尺度(logarithmic scale)缩放
	dftResultImage += 1;
	log(dftResultImage, dftResultImage);
	//剪切和重分布幅度图象限
	dftResultImage = dftResultImage(Rect(0,
		0, srcGray.cols, srcGray.rows));
	//归一化图像
	normalize(dftResultImage, dftResultImage,
		0, 1, CV_MINMAX);
	int cx = dftResultImage.cols / 2;
	int cy = dftResultImage.rows / 2;
	Mat tmp;
	//Top-Left-为每一个象限创建ROI
	Mat q0(dftResultImage, Rect(0, 0, cx, cy));
	//Top-Right
	Mat q1(dftResultImage, Rect(cx, 0, cx, cy));
	//Bootom-Left
	Mat q2(dftResultImage, Rect(0, cy, cx, cy));
	//Bootom-Right
	Mat q3(dftResultImage, Rect(cx, cy, cx, cy));
	//交换象限(Top-Right with Bootom-Left)
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	return dftResultImage;
}

void ImageOperations::Convolution(cv::Mat& src, cv::Mat kernel, cv::Mat& dst)
{
    //输出图像定义
    dst.create(abs(src.rows - kernel.rows) + 1, abs(src.cols - kernel.cols) + 1, src.type());
    Size dftSize;
    //计算傅里叶变换尺寸
    dftSize.width = getOptimalDFTSize(src.cols + kernel.cols - 1);
    dftSize.height = getOptimalDFTSize(src.rows + kernel.rows - 1);
    //创建临时图像，初始化为0
    Mat tempA(dftSize, src.type(), Scalar::all(0));
    Mat tempB(dftSize, kernel.type(), Scalar::all(0));
    //对区域进行复制
    Mat rolA(tempA, Rect(0, 0, src.cols, src.rows));
    src.copyTo(rolA);
    Mat rolB(tempB, Rect(0, 0, kernel.cols, kernel.rows));
    kernel.copyTo(rolB);
    //傅里叶变换
	dft(tempA, tempA, 0, src.rows);
	dft(tempB, tempB, 0, kernel.rows);
    //对频谱中每个元素进行惩罚操作
    mulSpectrums(tempA, tempB, tempA, DFT_COMPLEX_OUTPUT);
    //变换结果,所有行非零
    dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, dst.rows);
    //复制结果到输出图像
    tempA(Rect(0, 0, dst.cols, dst.rows)).copyTo(dst);
}

Mat ImageOperations::AddSaltNoise(Mat& srcImage, int n)
{
    Mat resultImage = srcImage.clone();
    for (int k = 0; k < n; k++)
    {
        int i = rand() % resultImage.cols;
        int j = rand() % resultImage.rows;
        //图像通道判定
        if (resultImage.channels() == 1)
        {
            resultImage.at<uchar>(j, i) = 255;
        } else
        {
            resultImage.at<Vec3b>(j, i)[0] = 255;
            resultImage.at<Vec3b>(j, i)[1] = 255;
            resultImage.at<Vec3b>(j, i)[2] = 255;
        }
    }
    return resultImage;
}

double ImageOperations::GenerateGaussianNoise(double mu, double sigma)
{
    //定义小值
    const double epsilon = std::numeric_limits<double>::min();
    static double z0, z1;
    static bool flag = false;
    flag = !flag;
    //flag为假构造高斯随机变量X
    if (!flag)
    {
        return z1 * sigma + mu;
    }
    double u1, u2;
    do
    {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while (u1 <= epsilon);
    //flag为真构造高斯随机变量X
    z0 = sqrt(-2.0 * log(u1)) * cos(2 * CV_PI * u2);
    z1 = sqrt(-2.0 * log(u1)) * cos(2 * CV_PI * u2);
    return z0 * sigma + mu;
}

Mat ImageOperations::AddGaussianNoise(Mat& srcImage)
{
    Mat resultImage = srcImage.clone();
    int channels = resultImage.channels();
    int nRows = resultImage.rows;
    int nCols = resultImage.cols * channels;
    //判断图像连续性
    if (resultImage.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }
    for (int i = 0; i < nRows; ++i)
    {
        for (int j = 0; j < nCols; ++j)
        {
            //添加高斯噪声
            int val = resultImage.ptr<uchar>(i)[j] + GenerateGaussianNoise(2, 0.8) * 32;
            if (val < 0)
            {
                val = 0;
            }
            if (val > 255)
            {
                val = 255;
            }
            resultImage.ptr<uchar>(i)[j] = (uchar) val;
        }
    }
    return resultImage;
}

cv::Mat ImageOperations::CorrectImageDirection(cv::Mat& srcImage)
{
    /*************************图像DFT尺寸转换*************************/
    const int nRows = srcImage.rows;
    const int nCols = srcImage.cols;
    //获取DFR尺寸
    int cRows = getOptimalDFTSize(nRows);
    int cCols = getOptimalDFTSize(nCols);
    //复制图像,超过边界区域填充为0
    Mat sizeConvMat;
    copyMakeBorder(srcImage, sizeConvMat, 0, cRows - nRows, 0,
                   cCols - nCols, BORDER_CONSTANT, Scalar::all(0));
    /*************************图像DFT变换*******************************/
    Mat groupMats[] = {Mat_<float>(sizeConvMat),
                       Mat::zeros(sizeConvMat.size(), CV_32F)};
    Mat mergeMat;
    //通道合并
    merge(groupMats, 2, mergeMat);
    //DFT变换
    dft(mergeMat, mergeMat);
    //分离通道
    split(mergeMat, groupMats);
    //计算幅值
    magnitude(groupMats[0], groupMats[1], groupMats[0]);
    Mat magnitudeMat = groupMats[0].clone();
    //归一化操作,幅值加1
    magnitudeMat += Scalar::all(1);
    //对数变换
    log(magnitudeMat, magnitudeMat);
    //归一化
    normalize(magnitudeMat, magnitudeMat, 0, 1, CV_MINMAX);
    //图像类型转换
    magnitudeMat.convertTo(magnitudeMat, CV_8UC1, 255, 0);
    /*************************频域中心移动*************************/
    int cx = magnitudeMat.cols / 2;
    int cy = magnitudeMat.rows / 2;
    Mat tmp;
    //Top-Left—为每一个象限创建ROI
    Mat q0(magnitudeMat, Rect(0, 0, cx, cy));
    Mat q1(magnitudeMat, Rect(cx, 0, cx, cy));
    Mat q2(magnitudeMat, Rect(0, cy, cx, cy));
    Mat q3(magnitudeMat, Rect(cx, cy, cx, cy));
    //交换象限(Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    //交换象限(Top-Right with Bottom-Left)
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    /*************************倾斜角检测*************************/
    //固定阈值二值化处理
    Mat binaryMagnMat;
    threshold(magnitudeMat, binaryMagnMat, 134, 255, CV_THRESH_BINARY);
    imshow("binaryMagnMat", binaryMagnMat);
    //霍夫变换
    vector<Vec2f> lines;
    binaryMagnMat.convertTo(binaryMagnMat, CV_8UC1, 255, 0);
    HoughLines(binaryMagnMat, lines, 1, CV_PI / 180, 100, 0, 0);
    //检测线个数
    Mat houghMat(binaryMagnMat.size(), CV_8UC3);
    for (size_t i = 0; i < lines.size(); ++i)
    {
        //绘制检测线
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        //坐标变换生成线表达式
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 + 1000 * (-b));
        pt2.y = cvRound(y0 + 1000 * (a));
        line(houghMat, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
    }


    float theta = 0;
    //检测线角度判断
    for (size_t i = 0; i < lines.size(); ++i)
    {
//        float thetaTemp = lines[i][1] * 180 / (float)CV_PI;
//        if (thetaTemp > 0 && thetaTemp < 90)
//        {
//            theta = thetaTemp;
//            break;
//        }
//        if (thetaTemp > 90 && thetaTemp < 180)
//        {
//            theta = thetaTemp - (float) CV_PI;
//            break;
//        }
        float piThresh = CV_PI/(float)90;
        float pi2 = CV_PI/(float)2;
        float thetaTemp = lines[i][1];
        if (abs(thetaTemp) < piThresh || abs(thetaTemp - pi2) < piThresh)
        {
            continue;
        } else
        {
            theta = thetaTemp;
            break;
        }
    }
    theta = theta * 180 / (float) CV_PI;

    //角度转换
    float angelT = nRows * tan(theta / 180 * (float)CV_PI) / nCols;
    theta = atan(angelT) * 180 / (float)CV_PI;
    cout << "theta:" << theta << endl;
    /*************************仿射变换矫正*************************/
    //取图像中心
    Point2f centerPoint = Point2f(nCols / 2, nRows / 2);
    double scale = 1;
    //计算旋转矩阵
    Mat warpMat = getRotationMatrix2D(centerPoint, theta, scale);
    //仿射变换
    Mat resultImage(srcImage.size(), srcImage.type());
    warpAffine(srcImage, resultImage, warpMat, resultImage.size());
    imshow("result", resultImage);
    waitKey(0);
    return resultImage;
}
