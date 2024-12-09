// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include "Functions.h"

wchar_t* projectPath;


void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void processH()
{
	Mat H;
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat src, hsv, channels[3];
		src = imread(fname);

		GaussianBlur(src, src, Size(5, 5), 0, 0);
		namedWindow("Original", 1);

		cvtColor(src, hsv, COLOR_BGR2HSV);
		split(hsv, channels);

		// hue component of HSV model is channels[0]
		H = (255.0 / 180) * channels[0];
		imshow("Original", src);
	}

	waitKey(0);

}

void processS()
{
	Mat S;
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat src, hsv, channels[3];
		src = imread(fname);

		GaussianBlur(src, src, Size(5, 5), 0, 0);
		namedWindow("Original", 1);

		cvtColor(src, hsv, COLOR_BGR2HSV);
		split(hsv, channels);

		// saturation component of HSV model is channels[1]
		S = channels[1];
		imshow("Original", src);
	}

	waitKey(0);
}

void processHS()
{

}


//int main() 
//{
//	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
//    projectPath = _wgetcwd(0, 0);
//
//	int op;
//	do
//	{
//		system("cls");
//		destroyAllWindows();
//		
//		printf(" 0 - Exit\n\n");
//		printf("Option: ");
//		scanf("%d",&op);
//		switch (op)
//		{
//			
//		}
//	}
//	while (op!=0);
//	return 0;
//}