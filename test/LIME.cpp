#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <lime.h>
using namespace std;
using namespace cv;
bool drawing_box;
Rect box;
int box_array_max_index=-1;
vector<Rect> box_array(100);
void draw_fix_box(Mat& img)
{
	if(box_array_max_index>=0&&box_array_max_index<100)
	{
		for(int i = 0;i<=box_array_max_index;i++){
		rectangle(
			img,
			Point(box_array[i].x, box_array[i].y),
			Point(box_array[i].x + box_array[i].width,
			box_array[i].y + box_array[i].height),
			Scalar(0xff,0x00,0x00)
			);
		}
	}
}
void draw_box(Mat& img){
		rectangle(
			img,
			Point(box.x, box.y),
			Point(box.x + box.width,
			box.y + box.height),
			Scalar(0xff,0x00,0x00)
			);
}

void my_mouse_callback(int event, int x, int y, int flags, void* param)
{
	Mat* image = (Mat*)param;
	switch(event)
	{
	case EVENT_LBUTTONDOWN:
	{
		drawing_box = true;
		box = Rect(x,y,0,0);
	}
	break;
	case EVENT_MOUSEMOVE:
		if(drawing_box)
		{
			box.width = x - box.x;
			box.height = y - box.y;		
		}
	break;
	case EVENT_LBUTTONUP:
	{
		if(box.width<0)
		{
			box.x += box.width;
			box.width*=-1;
		}
		if(box.height<0)
		{
			box.y += box.height;
			box.height*=-1;
		}
		box_array_max_index++;
		box_array[box_array_max_index].x=box.x;
		box_array[box_array_max_index].y=box.y;
		box_array[box_array_max_index].width=box.width;
		box_array[box_array_max_index].height=box.height;
		drawing_box = false;
	}
	break;
	default:
		break;
	}
}
int main()
{
	//读取视频
	cv::VideoCapture capture("../test/data/test.mp4");
	long totalFrameNumber = capture.get(CAP_PROP_FRAME_COUNT);
	cout<<"整个视频共"<<totalFrameNumber<<"帧"<<endl;
	capture.set( CAP_PROP_POS_FRAMES,0);
	
	//获取帧率
	double rate = capture.get(CAP_PROP_FPS);
	cout<<"帧率为:"<<rate<<endl;
	int delay = 1000/rate;
	cv::Mat frame;
	if(!capture.read(frame))
	    {
	    	cout<<"读取视频失败"<<endl;
	        return -1;
	    }
	cout<<"图片宽width为:"<<frame.rows<<endl;
	cout<<"图片高height为:"<<frame.cols<<endl;
	int isColor = 1;
	int fps = rate;
	int frameWidth = frame.rows;
	int frameHeight = frame.cols;
	printf("点击鼠标左键拖拉形成一个四边形区域，最后点击Esc开始检测\n");
	Mat currImage;
	capture >> currImage;
	feature::lime* l;
	l = new feature::lime(currImage);
	currImage = l->lime_enhance(currImage);
	delete l;
	namedWindow("画出检测区域",CV_WINDOW_AUTOSIZE);
	setMouseCallback(
		"画出检测区域",
		my_mouse_callback,
		(void*)(&currImage)
	);
	for (;;)
	{
		Mat temp;
		temp = currImage.clone();  //对图片进行处理
		draw_fix_box(temp);
		draw_box(temp);
		imshow("画出检测区域",temp);
		int r = waitKey(delay);
		if(r == 27)
			break;		
	}
	destroyWindow("画出检测区域");
    //帧差法检测框选区域	
	Mat preImage;
	currImage.copyTo(preImage);
	capture >> currImage;
	l = new feature::lime(currImage);
	currImage = l->lime_enhance(currImage);
	delete l;
	int CONTOUR_MAX_AERA = 200;
	while(true)
	{
		Mat preGrayImage, currGrayImage;
		cvtColor(preImage,preGrayImage,CV_BGR2GRAY);
		cvtColor(currImage,currGrayImage,CV_BGR2GRAY);
		Mat diffGray = preGrayImage - currGrayImage;
		threshold(diffGray,diffGray,30,255,THRESH_BINARY);
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(diffGray, contours, hierarchy,
		RETR_LIST,CHAIN_APPROX_SIMPLE,Point());
		//LIME增强提高对比度，突出信息，同时里面包含blur操作，降噪
		preImage = currImage.clone();
		draw_fix_box(currImage);
		for(int i = 0;i < contours.size();++i)
		{
			Rect r = boundingRect((Mat)contours[i]);
			for(int j = 0; j <= box_array_max_index; j++)
			{
				if(r.height * r.width>CONTOUR_MAX_AERA
					&& r.x>=box.x&&r.y>=box_array[j].y
					&& r.x + r.width <= box_array[j].x +box_array[j].width
					&& r.y + r.height < box_array[j].y +box_array[j].height){
					rectangle(currImage,Point(r.x,r.y),Point(r.x+r.width,r.y+r.height),Scalar(0xff,0xff,0x00));
					}
			}
		}
		imshow("CurrImg", currImage);
		capture >> currImage;
		l = new feature::lime(currImage);
		currImage = l->lime_enhance(currImage);
		delete l;
		if(currImage.empty()) break;
		if(waitKey(delay-30)==27) break;
	}
	capture.release();
	waitKey(0);
	return 0;
	

}


