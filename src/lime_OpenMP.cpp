#include "lime.h"
#include <vector>
#include <iostream>
#include <omp.h>

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
	void blur_parallel(cv::Mat& img_in, cv::Mat& img_out, int ksize)
	{
	    cv::Mat kernel = cv::getGaussianKernel(ksize, -1, CV_32F);
	    cv::sepFilter2D(img_in, img_out, -1, kernel, kernel);
	
	    #pragma omp parallel for
	    for (int i = 0; i < img_out.rows; i++)
	    {
	        for (int j = 0; j < img_out.cols; j++)
	        {
	            img_out.at<cv::Vec3b>(i, j) = img_out.at<cv::Vec3b>(i, j) / (kernel.at<float>(0) * kernel.at<float>(0));
	        }
	    }
	}
	//
	
	void lime::Illumination_filter(cv::Mat& img_in,cv::Mat& img_out)
    {
        int ksize = 5;
        //mean filter
        blur_parallel(img_in, img_out, ksize);
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
