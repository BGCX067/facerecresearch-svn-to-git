#include <cv.h>
#include <highgui.h>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

int main (){
	Mat img = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	
	Mat sum(img.rows, img.cols, CV_64F);

	//Calcula a imagem integral
	integral(img, sum);

	imshow ("Lena", img);
	cvWaitKey();
	destroyAllWindows();

	/*
	imshow ("Integral", sum);
	cvWaitKey();
	*/
	//system("pause");
	return EXIT_SUCCESS;
}