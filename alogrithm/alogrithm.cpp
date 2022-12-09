/**
 * ***************************@copyright (C) Copyright  2022  ZJU***************************
 * @file   alogrithm.cpp
 * @brief 此文件夹定义了一些自己实现的算法
 * @author Hao Lion(郝亮亮)    Email:(haolion_zju@foxmail.com)
 * @version 1.0
 * @date 2022-10-02
 * 
 * @verbatim:
 * ==============================================================================
 *                                                                               
 * ==============================================================================
 * @endverbatim
 * ***************************@copyright (C) Copyright  2022  ZJU***************************
 */

#include "alogrithm.hpp"
using namespace cv;

/**
 * @brief 
 * @param  src              原图像
 * @param  dst              目标图像
 * @param  gama            伽马矫正参数
 */
void GammaTransform(cv::Mat &gray_image,cv::Mat &dst,double gama)
{
    if (gray_image.type() == CV_8UC3) {
        std::cout << "Input must be Gray Image" << std::endl;
        cvtColor(gray_image, gray_image, COLOR_RGB2GRAY);
    }
	Mat imageGamma;
	//灰度归一化
	gray_image.convertTo(imageGamma, CV_64F, 1.0 / 255, 0);
	//伽马变换
	pow(imageGamma, gama, dst);//dst 要与imageGamma有相同的数据类型
	dst.convertTo(dst, CV_8U, 255, 0);
}

/**
* @brief 对输入图像进行细化,骨骼化
* @param src为输入图像,用cvThreshold函数处理过的8位灰度图像格式，元素中只有0与1,1代表有元素，0代表为空白
* @return 为对src细化后的输出图像,格式与src格式相同，元素中只有0与1,1代表有元素，0代表为空白
*/
cv::Mat thinImage(const cv::Mat & src)
{

	cv::Mat dst;
	assert(src.type() == CV_8UC1);
	//非原地操作时候，copy src到dst
	if (dst.data != src.data)
	{
		src.copyTo(dst);
	}
 
	int i, j, n;
	int width, height;
	//之所以减1，是方便处理8邻域，防止越界
	width = src.cols - 1;
	height = src.rows - 1;
	int step = src.step;
	int  p2, p3, p4, p5, p6, p7, p8, p9;
	uchar* img;
	bool ifEnd;
	Mat tmpimg;
	int dir[4] = { -step, step, 1, -1 };
	while (1)
	{
		//分四个子迭代过程，分别对应北，南，东，西四个边界点的情况
		ifEnd = false;
		for (n = 0; n < 4; n++)
		{
			dst.copyTo(tmpimg);
			img = tmpimg.data;
			for (i = 1; i < height; i++)
			{
				img += step;
				for (j = 1; j < width; j++)
				{
					uchar* p = img + j;
					//如果p点是背景点或者且为方向边界点，依次为北南东西，继续循环
					if (p[0] == 0 || p[dir[n]] > 0) continue;
					p2 = p[-step] > 0 ? 1 : 0;
					p3 = p[-step + 1] > 0 ? 1 : 0;
					p4 = p[1] > 0 ? 1 : 0;
					p5 = p[step + 1] > 0 ? 1 : 0;
					p6 = p[step] > 0 ? 1 : 0;
					p7 = p[step - 1] > 0 ? 1 : 0;
					p8 = p[-1] > 0 ? 1 : 0;
					p9 = p[-step - 1] > 0 ? 1 : 0;
					//8 simple判定
					int is8simple = 1;
					if (p2 == 0 && p6 == 0)
					{
						if ((p9 == 1 || p8 == 1 || p7 == 1) && (p3 == 1 || p4 == 1 || p5 == 1))
							is8simple = 0;
					}
					if (p4 == 0 && p8 == 0)
					{
						if ((p9 == 1 || p2 == 1 || p3 == 1) && (p5 == 1 || p6 == 1 || p7 == 1))
							is8simple = 0;
					}
					if (p8 == 0 && p2 == 0)
					{
						if (p9 == 1 && (p3 == 1 || p4 == 1 || p5 == 1 || p6 == 1 || p7 == 1))
							is8simple = 0;
					}
					if (p4 == 0 && p2 == 0)
					{
						if (p3 == 1 && (p5 == 1 || p6 == 1 || p7 == 1 || p8 == 1 || p9 == 1))
							is8simple = 0;
					}
					if (p8 == 0 && p6 == 0)
					{
						if (p7 == 1 && (p3 == 9 || p2 == 1 || p3 == 1 || p4 == 1 || p5 == 1))
							is8simple = 0;
					}
					if (p4 == 0 && p6 == 0)
					{
						if (p5 == 1 && (p7 == 1 || p8 == 1 || p9 == 1 || p2 == 1 || p3 == 1))
							is8simple = 0;
					}
					int adjsum;
					adjsum = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
					//判断是否是邻接点或孤立点,0,1分别对于那个孤立点和端点
					if (adjsum != 1 && adjsum != 0 && is8simple == 1)
					{
						dst.at<uchar>(i, j) = 0; //满足删除条件，设置当前像素为0
						ifEnd = true;
					}
 
				}
			}
		}
		if (!ifEnd) break;
	}
	return dst;

}

/**
* @brief 对骨骼化图数据进行过滤，实现两个点之间至少隔一个空白像素
* @param thinSrc为输入的骨骼化图像,8位灰度图像格式，元素中只有0与1,1代表有元素，0代表为空白
*/
void filterOver(cv::Mat & thinSrc)
{
	assert(thinSrc.type() == CV_8UC1);
	int width = thinSrc.cols;
	int height = thinSrc.rows;
	for (int i = 0; i < height; ++i)
	{
		uchar * p = thinSrc.ptr<uchar>(i);
		for (int j = 0; j < width; ++j)
		{
			// 实现两个点之间至少隔一个像素
			//  p9 p2 p3  
			//  p8 p1 p4  
			//  p7 p6 p5  
			uchar p1 = p[j];
			if (p1 == 0) continue;
			uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
			uchar p8 = (j == 0) ? 0 : *(p + j - 1);
			uchar p2 = (i == 0) ? 0 : *(p - thinSrc.step + j);
			uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - thinSrc.step + j + 1);
			uchar p9 = (i == 0 || j == 0) ? 0 : *(p - thinSrc.step + j - 1);
			uchar p6 = (i == height - 1) ? 0 : *(p + thinSrc.step + j);
			uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + thinSrc.step + j + 1);
			uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + thinSrc.step + j - 1);
			if (p2 + p3 + p8 + p9 >= 1)
			{
				p[j] = 0;
			}
		}
	}
}

/**
* @brief 从过滤后的骨骼化图像中寻找端点和交叉点
* @param thinSrc为输入的过滤后骨骼化图像,8位灰度图像格式，元素中只有0与1,1代表有元素，0代表为空白
* @param raudis卷积半径，以当前像素点位圆心，在圆范围内判断点是否为端点或交叉点
* @param thresholdMax交叉点阈值，大于这个值为交叉点
* @param thresholdMin端点阈值，小于这个值为端点
* @return 为对src细化后的输出图像,格式与src格式相同，元素中只有0与1,1代表有元素，0代表为空白
*/
std::vector<cv::Point> getPoints(const cv::Mat &thinSrc, unsigned int raudis, unsigned int thresholdMax, unsigned int thresholdMin)
{
	assert(thinSrc.type() == CV_8UC1);
	int width = thinSrc.cols;
	int height = thinSrc.rows;
	cv::Mat tmp;
	thinSrc.copyTo(tmp);
	std::vector<cv::Point> points;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			if (*(tmp.data + tmp.step * i + j) == 0)
			{
				continue;
			}
			int count=0;
			for (int k = i - raudis; k < i + raudis+1; k++)
			{
				for (int l = j - raudis; l < j + raudis+1; l++)
				{
					if (k < 0 || l < 0||k>height-1||l>width-1)
					{
						continue;
						
					}
					else if (*(tmp.data + tmp.step * k + l) == 1)
					{
						count++;
					}
				}
			}
 
			if (count > thresholdMax||count<thresholdMin)
			{
				Point point(j, i);
				points.push_back(point);
			}
		}
	}
	return points;
}
