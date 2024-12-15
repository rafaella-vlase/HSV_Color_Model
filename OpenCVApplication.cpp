// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include "Functions.h"
#define MAX_HUE 256

using namespace cv;
using namespace std;
wchar_t* projectPath;

// variabile globale
int histG_hue[MAX_HUE]; // histograma globala / cumulativa
int histG_saturation[MAX_HUE]; // histograma globala / cumulativa
bool draw = false; // Variabila globala care arata starea actiune de mouse dragging:
// true - in derulare

Point Pstart, Pend; // Punctele / colturile aferente selectiei ROI curente


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

//Functia CallBack care se apeleaza la declansarea evenimentelor de mouse
void CallBackFuncL2(int event, int x, int y, int flags, void* userdata)
{
	vector<Mat>* SRCandHSChannels = (vector<Mat>*)userdata;
	Mat src = SRCandHSChannels->at(0);
	Mat H = SRCandHSChannels->at(1);
	Mat S = SRCandHSChannels->at(2);
	Rect roi; // regiunea de interes curenta (ROI)
	
	if (event == EVENT_LBUTTONDOWN)
	{
		// punctul de start al ROI
		Pstart.x = x;
		Pstart.y = y;
		draw = true;
		printf("Pstart: (%d, %d) ", Pstart.x, Pstart.y);
	}
	else if (event == EVENT_MOUSEMOVE)
	{
		if (draw == true) // actiune de mouse dragging in derulare (activ)
		{
			// desenarea se face intr-o copie a matricii sursa
			Mat temp = src.clone();
			rectangle(temp, Pstart, Point(x, y), Scalar(0, 255, 0), 1, 8, 0);
			imshow("src", temp);
		}
	}
	else if (event == EVENT_LBUTTONUP && draw)
	{
		// punctul de final (diametral opus) al ROI rectangulare
		draw = false; // actiune de mouse dragging s-a terminat (inactiva)
		Pend.x = x;
		Pend.y = y;
		printf("Pend: (%d, %d) ", Pend.x, Pend.y);

		// sortare crescatoare a celor doua puncta selectate dupa x si y
		roi.x = min(Pstart.x, Pend.x);
		roi.y = min(Pstart.y, Pend.y);
		roi.width = abs(Pstart.x - Pend.x);
		roi.height = abs(Pstart.y - Pend.y);

		printf("Local ROI: (%d, %d), (%d, %d)\n", roi.x, roi.y,
			roi.x + roi.width, roi.y + roi.height);

		rectangle(src, roi, Scalar(0, 255, 0), 1, 8, 0);
		// desenarea selectiei rectangulare se face peste imaginea sursa
		imshow("src", src);

		
		Mat Hroi = H(roi); // componenta H in regiunea ROI
		Mat Sroi = S(roi); // componenta S in regiunea ROI
		uchar hue;

		int hist_hue[MAX_HUE] = { 0 }; // histograma locala a lui Hue
		int hist_saturation[MAX_HUE] = { 0 }; // histograma locala a lui Saturation

		//construieste histograma locala aferente ROI
		for (int y = 0; y < roi.height; y++)
			for (int x = 0; x < roi.width; x++)
			{
				hue = Hroi.at<uchar>(y, x);
				hist_hue[hue]++;

				hue = Sroi.at<uchar>(y, x);
				hist_saturation[hue]++;
			}
		//acumuleaza histograma locala in cea globala
		for (int i = 0; i < MAX_HUE; i++) {
			histG_hue[i] += hist_hue[i];
			histG_saturation[i] += hist_saturation[i];
		}
		// afiseaza histohrama locala
		showHistogram("H local histogram", hist_hue, MAX_HUE, 200, true);
		// afiseaza histohrama globala
		showHistogram("H global histogram", histG_hue, MAX_HUE, 200, true);

		showHistogram("S local histogram", hist_saturation, MAX_HUE, 200, true);
		showHistogram("S global histogram", histG_saturation, MAX_HUE, 200, true);
	}
}


void ColorModel_Build()
{
	Mat src, hsv, channels[3];
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		src = imread(fname);
		// Aplicare FTJ gaussian pt. eliminare zgomote / netezire imagine
		GaussianBlur(src, src, Size(5, 5), 0, 0);
		namedWindow("Original", 1);

		cvtColor(src, hsv, COLOR_BGR2HSV);
		split(hsv, channels);

		// hue component of HSV model is channels[0]
		Mat H = channels[0] * (255.0 / 180);
		// saturation component of HSV model is channels[1]
		Mat S = channels[1];

		// complete after impelementing the call back function
		vector<Mat> SRCandHSChannels;
		SRCandHSChannels.push_back(src);
		SRCandHSChannels.push_back(H);
		SRCandHSChannels.push_back(S);

		setMouseCallback("Original", CallBackFuncL2, &SRCandHSChannels);
		imshow("Original", src);
		waitKey(0);
	}
}

void ColorModel_Filter()
{
	int hue, sat, i, j;
	int histF_hue[MAX_HUE]; // histograma filtrata cu FTJ
	int histF_saturation[MAX_HUE]; // histograma filtrata cu FTJ
	memset(histF_hue, 0, MAX_HUE * sizeof(unsigned int));
	memset(histF_saturation, 0, MAX_HUE * sizeof(unsigned int));

	// filtrare histograma cu filtru gaussian 1D de dimensiune w=7
	float gauss[7];
	float sqrt2pi = sqrtf(2 * PI);
	float sigma = 1.5;
	float e = 2.718;
	float sum = 0;

	// Construire gaussian
	for (i = 0; i < 7; i++) 
	{
		gauss[i] = 1.0 / (sqrt2pi * sigma) * powf(e, -(float)(i - 3) * (i - 3)
			/ (2 * sigma * sigma));
		sum += gauss[i];
	}

	int maxH = 0;
	int maxS = 0;
	// Filtrare cu gaussian
	for (j = 3; j < MAX_HUE - 3; j++)
		for (i = 0; i < 7; i++) 
		{
			// Filtrare H
			histF_hue[j] += (float)histG_hue[j + i - 3] * gauss[i];
			if (histF_hue[j] > maxH)
				maxH = histF_hue[j];
			// Filtrare S
			histF_saturation[j] += (float)histG_saturation[j + i - 3] * gauss[i];
			if (histF_saturation[j] > maxS)
				maxS = histF_saturation[j];
		}
	double threshold = 1 / 10;
	for (j = 0; j < MAX_HUE; j++)
	{
		// Threshold H
		if (histF_hue[j] < maxH * threshold)
			histF_hue[j] = 0;
		histG_hue[j] = histF_hue[j];
		// Threshold S
		if (histF_saturation[j] < maxS * threshold)
			histF_saturation[j] = 0;
		histG_saturation[j] = histF_saturation[j];
	}
	// Histograma pentru HUE
	showHistogram("H global filtered histogram", histG_hue, MAX_HUE, 200, true);
	// Histograma pentru HUE
	showHistogram("S global filtered histogram", histG_saturation, MAX_HUE, 200, true);
	waitKey(0);
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
		H = channels[0] * (255.0 / 180);
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
	Mat H, S, HS;
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat src, hsv, channels[3];
		src = imread(fname);

		GaussianBlur(src, src, Size(5, 5), 0, 0);
		namedWindow("Original", 1);

		cvtColor(src, hsv, COLOR_BGR2HSV);
		split(hsv, channels);
		H = channels[0] * (255.0 / 180);
		S = channels[1];

		imshow("Original", src);
	}

	// binarizing the H and S components

	waitKey(0);
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