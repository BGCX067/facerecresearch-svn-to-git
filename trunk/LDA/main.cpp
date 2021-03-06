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

#include "subspace.hpp"
#include "fisherfaces.hpp"
#include "helper.hpp"
#include "decomposition.hpp"

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

void blurImages(vector<Mat>& images, int width, int height)
{
	for (int i = 0; i < images.size(); i++)
	{
		blur(images[i], images[i], cvSize(width, height));
	}//end for	
}

void equalizeImages(vector<Mat>& images)
{
	for (int i = 0; i < images.size(); i++)
	{
		equalizeHist(images[i], images[i]);
	}//end for	
}

int main(int argc, const char *argv[]) {
	vector<Mat> testSamples;
	vector<Mat> trainImages;
	vector<int> trainLabels;
	vector<int> testLabels;
	// check for command line arguments
	argc = 3;
	if(argc != 3) {
		cout << "usage: " << argv[0] << " <trainImagesCSV.ext> <testImagesCSV.ext>" << endl;
		exit(1);
	}
	// path to your CSV
	/*string fn_csv = string(argv[1]);
	string fn2_csv = string(argv[2]);*/
	string fn_csv = string("treino_at_0-9.dat");
	string fn2_csv = string("teste_at.dat");
	// read in the images
	try {
		read_csv(fn_csv, trainImages, trainLabels);
		read_csv(fn2_csv, testSamples, testLabels);
	} catch(exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\" or file \"" << fn2_csv << "\"." << endl;
		exit(1);
	}
	// get width and height
	/*int width = trainImages[0].cols;
	int height = trainImages[0].rows;
	// get test instances
	Mat testSample = images[images.size()-1];
	int testLabel = labels[labels.size()-1];
	// ... and delete last element
	images.pop_back();
	labels.pop_back();*/
/*
	blurImages(trainImages, 3, 3);
	blurImages(testSamples, 3, 3);

	equalizeImages(trainImages);
	equalizeImages(testSamples);
	*/
	cout << "Modeling..." << endl;
	// build the Fisherfaces model
	subspace::Fisherfaces model(trainImages, trainLabels);

	int predicted = -1;
	double result = 0.0;
	for (int i = 0; i < testSamples.size(); i ++){
		//Volta o cursor e sobrescreve
		system("cls");
		cout << "Predicting... \n" << ((double)(i+1)/(double)testSamples.size())*100 << "%" << endl;
		// test model
		predicted = model.predict(testSamples[i]);
		
		/*imshow("Con.", testSamples[i]);
		imshow("Pre", testSamples[predicted - 1]);
		waitKey();
		destroyAllWindows();*/

		result += (predicted == testLabels[i]? 1 : 0);
	}

	result /= testSamples.size();

	system("cls");
	cout << "Operation Done.\n\nMatching rate = " << result*100.0 << "%" << endl;

		//cout << "predicted class = " << predicted << endl;
		//cout << "actual class = " << testLabel << endl;

		// get the eigenvectors
		//Mat W = model.eigenvectors();

		// show first 10 fisherfaces
		/*for(int i = 0; i < min(10,W.cols); i++) {
		Mat ev = W.col(i).clone();
		imshow(format("%d",i), toGrayscale(ev.reshape(1, height)));
		}*/
		waitKey(0);
		return 0;
	}
