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

	//	���δ���Ϊʹ��neonָ���OpenMP���ж�˱�̼��ٸ�д��blur���� 
	void blur_neon_omp(cv::Mat& img_in, cv::Mat& img_out, cv::Size ksize)
	{
		int channel = 3;
	    // ��ȡͼ��ߴ�
	    int rows = img_in.rows;
	    int cols = img_in.cols;
	
	    // ��ȡ�ں�ά��
	    int krows = ksize.height;
	    int kcols = ksize.width;

	    // ����ê��
	    cv::Point anchor(-1, -1);
		
	    // ����߽�����
	    int borderType = cv::BORDER_DEFAULT;

	    // ��������С
	    int padx = kcols / 2;
	    int pady = krows / 2;

	    // ������ʱ������
	    cv::Mat tmp(rows + pady * 2, cols + padx * 2, img_in.type());

	    // ������ͼ���Ƶ���ʱ������
	    cv::copyMakeBorder(img_in, tmp, pady, pady, padx, padx, borderType);

	    // �������ͼ��ߴ�
	    int out_rows = rows;
	    int out_cols = cols;

	    // �����߳���
	    int num_threads = omp_get_num_procs();
	    // ������С
	    int block_size = rows / num_threads;
	    // �����ϲ���
	    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
	    for (int i = 0; i < rows; i += block_size) {
	        // ���㵱ǰ��Ŀ�ʼ�ͽ���������
	        int start_row = i;
	        int end_row = std::min(i, rows);

	        // ����ǰ��
	        for (int j = 0; j < cols; j++) {
	            // ���㵱ǰ��Ŀ�ʼ�ͽ���������
	            int start_col = j - padx;
	            int end_col = j + padx + 1;

	            // ��ʼ���ۼ���
	            int accum[4] = {0, 0, 0, 0};

	            // ����ÿ��ͨ���ľ��
	            for (int c = 0; c < channel; c++) {
	                // ��ʼ����������ͼ����ָ��ǰ�е�ָ��
	                const uchar* src_ptr = tmp.ptr<uchar>(start_row + pady, start_col + c);
	                uchar* dst_ptr = img_out.ptr<uchar>(start_row, j * channel + c);

	                // ���㵱ǰ��ľ��
	                for (int y = start_row; y < end_row; y++) {
	                    // ������ͼ�����8������
	                    uint8x8_t pixels = vld1_u8(src_ptr);

	                    // ������ת��Ϊ16λ����
	                    uint16x8_t pixels16 = vmovl_u8(pixels);
	
	                    // ��������ӵ��ۼ���
	                    accum[c] += vaddvq_u16(pixels16);

	                    // ��ָ��ǰ�Ƶ���һ��
	                    src_ptr += tmp.step;
	                    dst_ptr += img_out.step;

	                }

	                // ָ����һ�еĸ߼�ָ�뽫�ۼ�������洢�����ͼ����
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
