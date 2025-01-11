// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
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

void save_file(const std::string& fileName, int array1[MAX_HUE], int array2[MAX_HUE]) {
	std::ofstream fout(fileName, std::ios::binary);

	if (fout.is_open()) {
		fout.write(reinterpret_cast<char*>(array1), sizeof(int) * 256);
		fout.write(reinterpret_cast<char*>(array2), sizeof(int) * 256);
		fout.close();
		std::cout << "Model was succesfully saved in file: " << fileName << std::endl;
	}
	else {
		std::cerr << "Error while opening file for writing." << std::endl;
	}
}

void read_file(const std::string& fileName, int array1[MAX_HUE], int array2[MAX_HUE]) {
	std::ifstream fin(fileName, std::ios::binary);

	if (fin.is_open()) {
		fin.read(reinterpret_cast<char*>(array1), sizeof(int) * 256);
		fin.read(reinterpret_cast<char*>(array2), sizeof(int) * 256);

		fin.close();
		std::cout << "Model was succesfully read from file: " << fileName << std::endl;
	}
	else {
		std::cerr << "Error while opening file for reading." << std::endl;
	}
}

void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

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


void meanStdDev(int* hist, double* returnMean, double* returnStdDev)
{
	double sum_intensity = 0, sum_frequency = 0;

	for (int i = 0; i < MAX_HUE; i++)
	{
		sum_intensity += i * hist[i];
		sum_frequency += hist[i];
	}

	double mean = sum_intensity / sum_frequency;
	double sumSquaredDifferences = 0;

	for (int i = 0; i < MAX_HUE; i++)
	{
		sumSquaredDifferences += hist[i] * pow(i - mean, 2);
	}

	double stdDev = sqrt(sumSquaredDifferences / sum_frequency);

	cout << "Mean: " << mean << endl;
	cout << "Standard Deviation: " << stdDev << endl;

	*returnMean = mean;
	*returnStdDev = stdDev;
}

Mat colorModel_binarization_HorS(Mat src, string windowName, bool modeH)
{
	double doubleMean, doubleStdDev;
	int* hist = modeH == true ? histG_hue : histG_saturation;
	meanStdDev(hist, &doubleMean, &doubleStdDev);

	int mean = (int)doubleMean;
	int stdDev = (int)doubleStdDev;
	double k = 3;

	cout << "Mean: " << mean << endl;
	cout << "Standard Deviation: " << stdDev << endl;

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) > mean - k * stdDev && src.at<uchar>(i, j) < mean + k * stdDev)
				src.at<uchar>(i, j) = 255;
			else
				src.at<uchar>(i, j) = 0;
		}
	imshow(windowName, src);
	return src;
}

Mat colorModel_binarization_HS(Mat H, Mat S, string windowName)
{
	double doubleMeanH, doubleStdDevH, doubleMeanS, doubleStdDevS;

	meanStdDev(histG_hue, &doubleMeanH, &doubleStdDevH);
	meanStdDev(histG_saturation, &doubleMeanS, &doubleStdDevS);

	double doubleSigma = (doubleStdDevH + doubleStdDevS) / 2;

	double k = 3;
	double Th = k * doubleSigma;
	int uX1 = (int)doubleMeanH;
	int uX2 = (int)doubleMeanS;
	Mat dst = H.clone();

	cout << "Coordonatele in spatiu de culoare ale mediilor H si S. X1:" << uX1 << " X2:" << uX2 << endl;
	cout << "Media deviatiilor standard: " << doubleSigma << endl;
	cout << "Threshold: " << Th << endl;

	for (int i = 0; i < H.rows; i++)
		for (int j = 0; j < H.cols; j++) 
		{
			double D = sqrt(
				(H.at<uchar>(i, j) - uX1) * (H.at<uchar>(i, j) - uX1)
				+
				(S.at<uchar>(i, j) - uX2) * (S.at<uchar>(i, j) - uX2));

			if (D > Th) {
				dst.at<uchar>(i, j) = 0;
			}
			else {
				dst.at<uchar>(i, j) = 255;
			}
		}
	imshow(windowName, dst);
	return dst;
}	

// Deseneaza o cruce de dimensiune size x size peste punctul p
void drawCross(Mat& img, Point p, int size, Scalar color, int thickness)
{
	line(img, Point(p.x - size / 2, p.y), Point(p.x + size / 2, p.y), color, thickness, 8);
	line(img, Point(p.x, p.y - size / 2), Point(p.x, p.y + size / 2), color, thickness, 8);
}

/* Etichetare / Detectie contur si calcul PGS folosind functii din OpenCV
Input:
	name - numele ferestrei in care s eva afisa rezultatul
	src - imagine binara rezultata in urma segmentarii (0/negru ~ fond; 255/alb ~ obiect)
	output_format = true : desenare obiecte pline ~ etichetare
					false : desenare conture ale obiectelor
Apel:
	Labeling("Contur - functii din OpenCV", dst, false);
	*/
void labeling(const string& name, const Mat& src, bool output_format)
{
	// dst - matrice RGB24 pt. afisarea rezultatului
	Mat dst = Mat::zeros(src.size(), CV_8UC3);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(src, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	Moments m;
	if (contours.size() > 0)
	{
		// iterate through all the top-level contours,
		// draw each connected component with its own random color
		int idx = 0;
		for (; idx >= 0; idx = hierarchy[idx][0])
		{
			const vector<Point>& c = contours[idx];

			m = moments(c); // calcul momente imagine
			double arie = m.m00; // aria componentei conexe idx

			if (arie > 100)
			{
				double xc = m.m10 / m.m00; // coordonata x a CM al componentei conexe idx
				double yc = m.m01 / m.m00; // coordonata y a CM al componentei conexe idx

				Scalar color(rand() & 255, rand() & 255, rand() & 255);

				if (output_format) // desenare obiecte pline ~ etichetare
					drawContours(dst, contours, idx, color, FILLED, 8, hierarchy);
				else  //desenare contur obiecte
					drawContours(dst, contours, idx, color, 1, 8, hierarchy);

				Point center(xc, yc);
				int radius = 5;

				// afisarea unor cercuri in jurul centrelor de masa
				//circle(final, center, radius,Scalar(255,255,355), 1, 8, 0);

				// afisarea unor cruci peste centrele de masa
				drawCross(dst, center, 9, Scalar(255, 255, 255), 1);

				//calcul axa de alungire folosind momentele centarte de ordin 2
				double mc20p = m.m20 / m.m00 - xc * xc; // double mc20p = m.mu20 / m.m00;
				double mc02p = m.m02 / m.m00 - yc * yc; // double mc02p = m.mu02 / m.m00;
				double mc11p = m.m11 / m.m00 - xc * yc; // double mc11p = m.mu11 / m.m00;
				float teta = 0.5 * atan2(2 * mc11p, mc20p - mc02p);
				float teta_deg = teta * 180 / PI;

				printf("ID=%d, arie=%.0f, xc=%0.f, yc=%0.f, teta=%.0f\n", idx, arie, xc, yc, teta_deg);

				// calcul puncte de intersectie & afisare axa de alungire
				Point* points = (Point*)malloc(4 * sizeof(Point));
				Point* goodPoints = (Point*)malloc(2 * sizeof(Point));
				points[0].y = yc + tan(teta) * (-1 * xc);
				points[0].x = 0;
				points[1].y = yc + tan(teta) * (dst.cols - 1 - xc);
				points[1].x = dst.cols - 1;
				points[2].x = xc + (-1 * yc) / tan(teta);
				points[2].y = 0;
				points[3].x = xc + (dst.rows - 1 - yc) / tan(teta);
				points[3].y = dst.rows - 1;

				int k = 0;
				for (int i = 0; i <= 3; i++)
					if (points[i].x >= 0 && points[i].x <= dst.cols - 1 && points[i].y >= 0 && points[i].y <= dst.rows - 1) 
					{
						printf("%d %d\n", points[i].x, points[i].y);
						goodPoints[k++] = points[i];
					}
				line(dst, goodPoints[0], goodPoints[1], Scalar(255, 255, 255));

			}
		}
	}

	imshow(name, dst);
}

Mat erodeDilatation(Mat src, string windowName)
{
	Mat element1 = getStructuringElement(MORPH_CROSS, Size(5, 5));
	erode(src, src, element1, Point(-1, -1), 2);
	dilate(src, src, element1, Point(-1, -1), 4);
	erode(src, src, element1, Point(-1, -1), 2);
	imshow(windowName, src);
	return src;
}

void processModel()
{
	cout << "Color model init" << endl;
	// Selectam ROI-uri si le adaugam la hisogramele globale
	ColorModel_Build();
	// Filtram si salvam histogramele (modelul antrenat)
	ColorModel_Filter();
	
	// Salvam modelul intr-un fisier
	string fileName;
	cout << "Give a name for the model: ";
	cin >> fileName;
	save_file(fileName, histG_hue, histG_saturation);
}

void importModel() 
{
	string fileName;
	cout << "Give the name of the model you want to import: ";
	cin >> fileName;
	read_file(fileName, histG_hue, histG_saturation);

	showHistogram("HUE after import", histG_hue, MAX_HUE, 200);
	showHistogram("SAT after import", histG_saturation, MAX_HUE, 200);

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

	// Binarizam imaginea in functie de modelul incarcat
	colorModel_binarization_HorS(H, "Imagine binarizata - H", 0);
	erodeDilatation(H, "Eroziune si Dilatare - H");
	labeling("Contur - H", H, false);
	labeling("Umplere - H", H, true);
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

	// Binarizam imaginea in functie de modelul incarcat
	colorModel_binarization_HorS(S, "Imagine binarizata - S", 1);
	erodeDilatation(S, "Eroziune si Dilatare - S");
	labeling("Contur - S", S, false);
	labeling("Umplere - S", S, true);
	waitKey(0);
}

void processHS()
{
	Mat H, S;
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

	// Binarizam imaginea in functie de modelul incarcat
	Mat HS = colorModel_binarization_HS(H, S, "Imagine binarizata - H&S");
	erodeDilatation(HS, "Eroziune si Dilatare - H&S");
	labeling("Contur - H&S", HS, false);
	labeling("Umplere - H&S", HS, true);

	waitKey(0);
}


int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);

	// se initializeaza histogramele globale la 0
	memset(histG_hue, 0, sizeof(unsigned int) * MAX_HUE);
	memset(histG_saturation, 0, sizeof(unsigned int) * MAX_HUE);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		
		printf("Alege un item:\n");
		printf(" 1 - Procesare model\n");
		printf(" 2 - Import model\n");
		printf(" 3 - Procesare H\n");
		printf(" 4 - Procesare S\n");
		printf(" 5 - Procesare H si S\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				processModel();
				break;
			case 2:
				importModel();
				break;
			case 3:
				processH();
				break;
			case 4:
				processS();
				break;
			case 5:
				processHS();
				break;
		}
	} while (op!=0);
	return 0;
}