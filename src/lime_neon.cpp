#include "lime.h"
#include <vector>
#include <iostream>
#include <arm_neon.h>
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

	//	本段代码为使用neon指令改写的blur函数 
    void neon_blur(cv::Mat img_in, cv::Mat img_out, cv::Size ksize){
	    int kernel_size = ksize.width;
	    int kernel_half = kernel_size / 2;
	    int img_height = img_in.rows;
	    int img_width = img_in.cols;
	    int img_channels = img_in.channels();
	    int img_out_channels = img_out.channels();
	
	    float32x4_t kernel = vdupq_n_f32(1.0f / kernel_size);
	    float32x4_t accum;
	
	    uint8_t *pSrc = img_in.data + img_width * img_channels * kernel_half;
	    uint8_t *pDst = img_out.data;
	    uint8_t *pEnd = img_in.data + img_width * img_channels * (img_height - kernel_half);
	
	    for (; pSrc < pEnd; pSrc += img_width * img_channels, pDst += img_width * img_out_channels) {
	        for (int i = 0; i < img_width; i += 4) {
	            accum = vmovq_n_f32(0.0f);
	            for (int j = -kernel_half; j <= kernel_half; j++) {
	                uint8_t *p = pSrc + img_width * img_channels * j + i * img_channels;
	                float32x4_t data = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(p)))));
	                accum = vmlaq_f32(accum, kernel, data);
	            }
	            accum = vbslq_f32(vcltq_f32(accum, vmovq_n_f32(0.0f)), vmovq_n_f32(0.0f), accum);
	            accum = vbslq_f32(vcgtq_f32(accum, vmovq_n_f32(255.0f)), vmovq_n_f32(255.0f), accum);
	            uint16x4_t data_u16 = vqmovun_s32(vcvtq_s32_f32(accum));
	            uint8x8_t data_u8 = vqmovn_u16(vcombine_u16(data_u16, data_u16));
	            vst1_u8(pDst + i * img_out_channels, data_u8);
	        }
	    }
	}
	//
	
	void lime::Illumination_filter(cv::Mat& img_in,cv::Mat& img_out)
    {
        cv::Size ksize(5,5);
        //mean filter
        neon_blur(img_in, img_out, ksize);
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
