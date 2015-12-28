#include <stdio.h>
#include <vector>
#include <fstream>
#include <string.h>
#include <opencv2/opencv.hpp>

using namespace cv;

void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, vector<string>& paths) {
	std::ifstream file(filename.c_str(), std::ifstream::in);
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
			paths.push_back(path);
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
		i ++;
	}
}

void gerarMedia(vector<Mat>&artistas, string nome_arquivo, bool equalize)
{
	Mat sum = Mat::zeros(cvSize(artistas[0].cols, artistas[0].rows), artistas[0].type());
	
	if (equalize)
	{
		for (int i = 0; i < artistas.size(); i++)
		{
			equalizeHist(artistas[i], artistas[i]);
			addWeighted(artistas[i], 1.0/(double)artistas.size(), sum, 1, 0, sum);
		}//end for
	}
	else
	{
		for (int i = 0; i < artistas.size(); i++)
		{
			addWeighted(artistas[i], 1.0/(double)artistas.size(), sum, 1, 0, sum);
		}//end for
	}
	
	imshow("Media Homens", sum);
	imwrite(nome_arquivo, sum);
	cvWaitKey();
	destroyAllWindows();
}

void blurImages(vector<Mat>& artistas, vector<string> caminhos, string nome_pasta)
{
	string nome_arquivo;
	unsigned int pos;
	string comando = "mkdir ";
	comando.append(nome_pasta);

	system(comando.c_str());

	for (int i = 0; i < artistas.size(); i++)
	{
		nome_arquivo = nome_pasta;
		pos = caminhos[i].find("\\");
		nome_arquivo += caminhos[i].substr(pos);
		blur(artistas[i], artistas[i], cvSize(3,3));
		imwrite(nome_arquivo, artistas[i]);
	}//end for	
}

void equalizeImages(vector<Mat>& artistas, vector<string> caminhos, string nome_pasta)
{
	string nome_arquivo;
	unsigned int pos;
	string comando = "mkdir ";
	comando.append(nome_pasta);

	system(comando.c_str());

	for (int i = 0; i < artistas.size(); i++)
	{
		nome_arquivo = nome_pasta;
		pos = caminhos[i].find("\\");
		nome_arquivo += caminhos[i].substr(pos);
		equalizeHist(artistas[i], artistas[i]);
		imwrite(nome_arquivo, artistas[i]);
	}//end for	
}

void projetarIntegrais(Mat rosto, Mat& integraisVerticais, Mat& integraisHorizontais){
	double Vx, Hy;
	for (int x = 0; x < rosto.rows; x ++)
	{
		Vx = 0.0;
		for (int j = 0; j < rosto.cols; j ++)
		{
			Vx += rosto.at<unsigned char>(x, j);
		}//end for
		integraisVerticais.at<double>(0, x) = Vx;
	}//end for

	for (int y = 0; y < rosto.cols; y ++)
	{
		Hy = 0.0;
		for (int j = 0; j < rosto.rows; j ++)
		{
			Hy += rosto.at<unsigned char>(j, y);
		}//end for
		integraisHorizontais.at<double>(0, y) = Hy;
	}//end for
}

void definirMapa(Mat rosto, Mat& mapa){
	Mat Vxs(1, rosto.rows, CV_64FC1), Hys(1, rosto.cols, CV_64FC1);

	double _P = mean(rosto)[0];

	projetarIntegrais(rosto, Vxs, Hys);

	for (int i = 0; i < rosto.rows; i ++)
	{
		for (int j = 0; j < rosto.cols; j ++)
		{
			mapa.at<double>(i, j) =
				(Vxs.at<double>(i)*Hys.at<double>(j))/(Vxs.total()*Hys.total()*_P);
		}//end for
	}//end for
}

int main(int argc, char* argv[])
{
	//Amostras de treino, respectivas classes e caminhos dos arquivos de imagens
	vector<Mat> treino;
	vector<int> classes;
	vector<string> caminhos;
	
	//Mapa de projeções verticais e horizontais
	Mat mapa;

	//Bordas
	Mat edges;

	//Parâmetro de combinação
	double combinacao = 0.9;

	read_csv("treino_at_0-9.dat", treino, classes, caminhos);

	Mat rosto = treino[169];
	mapa = Mat::zeros(rosto.rows, rosto.cols, CV_64FC1);
	
	imshow("Rosto", rosto);
	
	definirMapa(rosto, mapa);

	for (int x = 0; x < rosto.rows; x ++)
	{
		for (int y = 0; y < rosto.cols; y ++)
		{
			mapa.at<double>(x, y) = 
				(rosto.at<unsigned char>(x, y) +
				(combinacao * mapa.at<double>(x, y)))
				/
				(1 + combinacao);
		}//end for
	}//end for

	mapa.convertTo(mapa, CV_8UC1);

	Sobel(mapa, edges, 3, 3, 3);

	imshow("Mapa", mapa);

	imshow("Edges", edges);
	cvWaitKey();
	cvDestroyAllWindows();

	return 0;
}