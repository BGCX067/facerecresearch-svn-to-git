/*
* Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
* Released to public domain under terms of the BSD Simplified license.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in the
*     documentation and/or other materials provided with the distribution.
*   * Neither the name of the organization nor the names of its contributors
*     may be used to endorse or promote products derived from this software
*     without specific prior written permission.
*
*   See <http://www.opensource.org/licenses/bsd-license>
*/

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <Windows.h>

using namespace cv;
using namespace std;

void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels) {
	std::ifstream file(filename.c_str(), ifstream::in);
	if(!file)
		throw std::exception();
	std::string line, path, classlabel;
	int i = 0;
	// for each line
	while (std::getline(file, line)) {
		// get current line
		std::stringstream liness(line);
		// split line
		std::getline(liness, path, ';');
		std::getline(liness, classlabel);
		// push pack the data if any
		if(!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
		i ++;
	}
}

void SetBack()
{
	COORD CursorPosition = {0, 0};
	SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), CursorPosition);
}

int main(int argc, const char *argv[]) {
	//// Example for a Linear Discriminant Analysis
	//// (example taken from: http://www.bytefish.de/wiki/pca_lda_with_gnu_octave)
	//double d[11][2] = {
	//		{2, 3},
	//		{3, 4},
	//		{4, 5},
	//		{5, 6},
	//		{5, 7},
	//		{2, 1},
	//		{3, 2},
	//		{4, 2},
	//		{4, 3},
	//		{6, 4},
	//		{7, 6}};
	//int c[11] = {0,0,0,0,0,1,1,1,1,1,1};
	//// convert into OpenCV representation
	//Mat _data = Mat(11, 2, CV_64FC1, d).clone();
	//vector<int> _classes(c, c + sizeof(c) / sizeof(int));
	//// perform the lda
	//subspace::LinearDiscriminantAnalysis lda(_data, _classes);
	//// GNU Octave finds the following Eigenvalue:
	////octave> d
	////d =
	////	 1.5195e+00
	////
	//// Eigen finds the following Eigenvalue:
	//// [1.519536390756363]
	////
	//// Since there's only 1 discriminant, this is correct.
	//cout << "Eigenvalues:" << endl << lda.eigenvalues() << endl;
	//// GNU Octave finds the following Eigenvectors:
	////	octave:13> V(:,1)
	////	V =
	////
	////	   0.71169  -0.96623
	////	  -0.70249  -0.25766
	////
	//// Eigen finds the following Eigenvector:
	//// [0.7116932742510111;
	////  -0.702490343980524 ]
	////
	//cout << "Eigenvectors:" << endl << lda.eigenvectors() << endl;
	//// project a data sample onto the subspace identified by LDA
	//Mat x = _data.row(0);
	//cout << "Projection of " << x << ": " << endl;
	//cout << lda.project(x) << endl;
	//// example for reading a face database from a CSV file
	////
	//// CSV -- https://github.com/bytefish/opencv/blob/master/lda/at.txt
	//// Database -- http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
	////
	//// Always make sure classes are given as {0, 1,..., n}!
	vector<Mat> testSamples;
	vector<Mat> trainImages;
	vector<int> trainLabels;
	vector<int> testLabels;
	// check for command line arguments
	if(argc != 3) {
		cout << "usage: " << argv[0] << " <trainImagesCSV.ext> <testImagesCSV.ext>" << endl;
		exit(1);
	}
	// path to your CSV
	string fn_csv = string(argv[1]);
	string fn2_csv = string(argv[2]);
	// read in the images
	try {
		read_csv(fn_csv, trainImages, trainLabels);
		read_csv(fn2_csv, testSamples, testLabels);
	} catch(exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\" or file \"" << fn2_csv << "\"." << endl;
		exit(1);
	}
	
	cout << "Modeling..." << endl;
	// build the Fisherfaces model
	Ptr<FaceRecognizer> lbph = createLBPHFaceRecognizer();
	lbph->train(trainImages, trainLabels);
		
	int predicted = -1;
	double result = 0.0;

	for (int i = 0; i < testSamples.size(); i ++){
		//Volta o cursor e sobrescreve
		system("cls");
		cout << "Predicting... \n" << ((double)(i+1)/(double)testSamples.size())*100 << "%" << endl;
		// test model
		predicted = lbph->predict(testSamples[i]);
		result += (predicted == testLabels[i]? 1 : 0);
	}

	result /= testSamples.size();

	system("cls");
	cout << "Operation Done.\n\nMatching rate = " << result*100.0 << "%" << endl;

		waitKey(0);
		return 0;
	}