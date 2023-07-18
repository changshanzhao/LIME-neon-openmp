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
	void omp_blur(const cv::Mat& img_in, cv::Mat& img_out, cv::Size ksize) {
	    int w = img_in.cols;
	    int h = img_in.rows;
	    int ksize_x = ksize.width;
	    int ksize_y = ksize.height;
	    int n = w * h;
	    int len = ksize_x * ksize_y;
	    cv::Mat img_tmp(n, len, CV_32F);
	    cv::Mat img_out_tmp(n, len, CV_32F);
	
	#pragma omp parallel for shared(img_in, img_out, ksize) private(img_tmp, img_out_tmp) reduction(+:n)
	    for (int i = 0; i < n; i++) {
	        float* p_img_in = const_cast<float*>(img_in.ptr<float>(i));
	        float* p_img_out = const_cast<float*>(img_out.ptr<float>(i));
	        float* p_img_tmp = const_cast<float*>(img_tmp.ptr<float>(i));
	        float* p_img_out_tmp = const_cast<float*>(img_out_tmp.ptr<float>(i));
	
	        int x1 = i % ksize_x;
	        int y1 = i / ksize_x;
	        int x2 = (i + 1) % ksize_x;
	        int y2 = (i + 1) / ksize_x;
	
	        for (int j = y1; j < y2; j++) {
	            for (int x = x1; x < x2; x++) {
	                float sum = 0.0f;
	                for (int u = -ksize_x + x; u <= ksize_x + x; u++) {
	                    for (int v = -ksize_y + j; v <= ksize_y + j; v++) {
	                        if (u >= 0 && u < w && v >= 0 && v < h) {
	                            sum += p_img_in[((y1 - j) * w + (x1 - x))] * p_img_tmp[((y2 - j) * len + (x2 - x))];
	                        }
	                    }
	                }
	                p_img_out[i] = sum;
	                p_img_out_tmp[i] = sum;
	            }
	        }
	    }
	    img_out.copyTo(img_in);
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
