#include "lime.h"
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <arm_neon.h>
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

        else
        {
            std::cout<<"There is a problem with the channels"<<std::endl;
            exit(-1);
        }
        return out_lime.clone();
    }

	//	本段代码为使用neon指令和OpenMP进行多核编程加速改写的blur函数 
	void blur_neon_omp(cv::Mat& img_in, cv::Mat& img_out, cv::Size ksize)
	{
		int channel = 3;
	    // 获取图像尺寸
	    int rows = img_in.rows;
	    int cols = img_in.cols;
	
	    // 获取内核维度
	    int krows = ksize.height;
	    int kcols = ksize.width;

	    // 计算锚点
	    cv::Point anchor(-1, -1);
		
	    // 计算边界类型
	    int borderType = cv::BORDER_DEFAULT;

	    // 计算填充大小
	    int padx = kcols / 2;
	    int pady = krows / 2;

	    // 分配临时缓冲区
	    cv::Mat tmp(rows + pady * 2, cols + padx * 2, img_in.type());

	    // 将输入图像复制到临时缓冲区
	    cv::copyMakeBorder(img_in, tmp, pady, pady, padx, padx, borderType);

	    // 计算输出图像尺寸
	    int out_rows = rows;
	    int out_cols = cols;

	    // 计算线程数
	    int num_threads = omp_get_num_procs();
	    // 计算块大小
	    int block_size = rows / num_threads;
	    // 在行上并行
	    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
	    for (int i = 0; i < rows; i += block_size) {
	        // 计算当前块的开始和结束行索引
	        int start_row = i;
	        int end_row = std::min(i, rows);

	        // 处理当前块
	        for (int j = 0; j < cols; j++) {
	            // 计算当前块的开始和结束列索引
	            int start_col = j - padx;
	            int end_col = j + padx + 1;

	            // 初始化累加器
	            int accum[4] = {0, 0, 0, 0};

	            // 计算每个通道的卷积
	            for (int c = 0; c < channel; c++) {
	                // 初始化输入和输出图像中指向当前行的指针
	                const uchar* src_ptr = tmp.ptr<uchar>(start_row + pady, start_col + c);
	                uchar* dst_ptr = img_out.ptr<uchar>(start_row, j * channel + c);

	                // 计算当前块的卷积
	                for (int y = start_row; y < end_row; y++) {
	                    // 从输入图像加载8个像素
	                    uint8x8_t pixels = vld1_u8(src_ptr);

	                    // 将像素转换为16位整数
	                    uint16x8_t pixels16 = vmovl_u8(pixels);
	
	                    // 将像素添加到累加器
	                    accum[c] += vaddvq_u16(pixels16);

	                    // 将指针前移到下一行
	                    src_ptr += tmp.step;
	                    dst_ptr += img_out.step;

	                }

	                // 指向下一行的高级指针将累加器结果存储在输出图像中
	                *dst_ptr = accum[c] / (krows * kcols);
	            }
	        }
	    }
	}

	void lime::Illumination_filter(cv::Mat& img_in,cv::Mat& img_out)
    {
        int ksize = 5;
        //mean filter
        blur_neon_omp(img_in, img_out, cv::Size(ksize,ksize));
		//blur(img_in,img_out,cv::Size(ksize,ksize));
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
