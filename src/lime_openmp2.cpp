#include "lime.h"
#include <vector>
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>

namespace feature
{
    lime::lime(cv::Mat src)
    {
        channel = src.channels();
    }

    cv::Mat lime::lime_enhance(cv::Mat &src)
    {
        cv::Mat img_norm;
        src.convertTo(img_norm, CV_32F, 1 / 255.0, 0);

        cv::Size sz(img_norm.size());
        cv::Mat out(sz, CV_32F, cv::Scalar::all(0.0));

        auto gammT = out.clone();

        if (channel == 3)
        {

            Illumination(img_norm, out);
            Illumination_filter(out, gammT);

            //lime
            std::vector<cv::Mat> img_norm_rgb;
            cv::Mat img_norm_b, img_norm_g, img_norm_r;

            cv::split(img_norm, img_norm_rgb);

            img_norm_g = img_norm_rgb.at(0);
            img_norm_b = img_norm_rgb.at(1);
            img_norm_r = img_norm_rgb.at(2);

            cv::Mat one = cv::Mat::ones(sz, CV_32F);

            float nameta = 0.9;
            auto g = 1 - ((one - img_norm_g) - (nameta * (one - gammT))) / gammT;
            auto b = 1 - ((one - img_norm_b) - (nameta * (one - gammT))) / gammT;
            auto r = 1 - ((one - img_norm_r) - (nameta * (one - gammT))) / gammT;

            cv::Mat g1, b1, r1;

            //TODO <=1
            threshold(g, g1, 0.0, 0.0, 3);
            threshold(b, b1, 0.0, 0.0, 3);
            threshold(r, r1, 0.0, 0.0, 3);

            img_norm_rgb.clear();
            img_norm_rgb.push_back(g1);
            img_norm_rgb.push_back(b1);
            img_norm_rgb.push_back(r1);

            cv::merge(img_norm_rgb,out_lime);
            out_lime.convertTo(out_lime,CV_8U,255);

        }
        else if(channel == 1)
        {
            Illumination_filter(img_norm, gammT);
            cv::Mat one = cv::Mat::ones(sz, CV_32F);
            float nameta = 0.9;
            //std::cout<<img_norm.at<float>(1,1)<<std::endl;
            auto out = 1 - ((one - img_norm) - (nameta * (one - gammT))) / gammT;

            threshold(out, out_lime, 0.0, 0.0, 3);

            out_lime.convertTo(out_lime,CV_8UC1,255);

        }

        else
        {
            std::cout<<"There is a problem with the channels"<<std::endl;
            exit(-1);
        }
        return out_lime.clone();
    }

//	本段代码为使用OpenMP进行多核编程改写的blur函数 
	void omp_blur(const cv::Mat& img_in, cv::Mat& img_out, const cv::Size ksize)
	{
	    // 定义卷积核
	    cv::Mat kernel = cv::getGaussianKernel(ksize.width, -1, CV_32F);
	
	    // 对卷积核进行归一化
	    cv::Scalar sum = cv::sum(kernel);
	    kernel /= sum[0];
	
	    // 创建输出图像
	    img_out.create(img_in.size(), img_in.type());

	    // 并行执行模糊操作
	    #pragma omp parallel for
    	for (int y = 0; y < img_in.rows; ++y)
    	{
	        // 按行处理图像
	        cv::Mat row_in = img_in.row(y);
	        cv::Mat row_out = img_out.row(y);
	
	        // 对当前行进行模糊操作
	        cv::filter2D(row_in, row_out, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
   		}
	}
//
	
	void lime::Illumination_filter(cv::Mat& img_in,cv::Mat& img_out)
    {
        cv::Size ksize(5,5);
        //mean filter
        omp_blur(img_in, img_out, ksize);
        //GaussianBlur(img_in,img_mean_filter,Size(ksize,ksize),0,0);

        //gamma
        int row = img_out.rows;
        int col = img_out.cols;
        float tem;
        float gamma = 0.8;
        for(int i=0;i<row;i++)
        {

            for(int j=0;j<col;j++)
            {
                tem = pow(img_out.at<float>(i,j),gamma);
                tem = tem <= 0 ? 0.0001 : tem;  //  double epsolon = 0.0001;
                tem = tem > 1 ? 1 : tem;

                img_out.at<float>(i,j) = tem;

            }
        }

    }
    void lime::Illumination(cv::Mat& src,cv::Mat& out)
    {
        int row = src.rows, col = src.cols;

        for(int i=0;i<row;i++)
        {
            for(int j=0;j<col;j++)
            {
                out.at<float>(i,j) = lime::compare(src.at<cv::Vec3f>(i,j)[0],
                                                   src.at<cv::Vec3f>(i,j)[1],
                                                   src.at<cv::Vec3f>(i,j)[2]);
            }

        }

    }

}
