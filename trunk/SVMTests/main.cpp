#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define TIPO CV_64FC1

FILE* arquivo;
vector<Mat> treino;

//Possuem as classes de treino e a quantidade de imagens por classe
vector<int> classes_treino, quant_por_classe;

int altura, largura, num_pixels, num_rostos_treino, num_classes, num_componentes;
Mat matriz, media, covar, icovar, autovalores, autovetores, projecoes;

void carregarTreino(const char* nome_do_arquivo);
void criarMatriz();
void obterMatrizMedia();
void mostrarImagemMedia();
void normalizarMatriz();
void calcularMatrizDeCovariancia();
void carregarAutoObjetos();
void mostrarImagensDosAutovetores();
void projetarTreino();
void gerarArquivoTreino(char * nome_arquivo_treino);
void gerarArquivoTeste(char * nome_arquivo_teste);

int main(int argc, char* argv[]){
	char* nome_arquivo_treino = argv[1];
	char* nome_arquivo_teste;
	Mat teste;
	int classe_pos = -1;
	double distancia = DBL_MAX, porcentagem_acerto = -1.0;

	system("cls");
	printf("--------------------------- SVM ------------------------------\n");
	printf("Loading training data...\n");
	carregarTreino(nome_arquivo_treino);

	printf("Creating row matrix...\n");
	criarMatriz();

	printf("Getting mean matrix...\n");
	obterMatrizMedia();

	printf("Normalizing matrix...\n");
	normalizarMatriz();

	printf("Calculating the covariance matrix...\n");
	calcularMatrizDeCovariancia();

	printf("Loading eigen-objects...\n");
	carregarAutoObjetos();

	//mostrarImagensDosAutovetores();

	printf("Projecting training samples...\n");
	projetarTreino();

	printf ("Generating trainning file for SVM...\n");
	gerarArquivoTreino(nome_arquivo_treino);

	printf ("Generating testing file for SVM...\n");
	nome_arquivo_teste = argv[2];
	gerarArquivoTeste(nome_arquivo_teste);

	printf("Operation Done!");
	return 0;
}

void criarMatriz(){
	matriz = cvCreateMat(num_rostos_treino, num_pixels, TIPO);

	for(int i = 0; i < num_rostos_treino; i ++){
		Mat Xi = matriz.row(i);

		if(treino[i].isContinuous()){
			treino[i].reshape(1, 1).convertTo(Xi, TIPO, 1/255.);
		}
		else{
			treino[i].clone().reshape(1, 1).convertTo(Xi, TIPO, 1/255.);
		}
	}
}

void obterMatrizMedia(){
	media = cvCreateMat(1, num_pixels, TIPO);
	for (int i = 0; i < num_pixels; i ++)
	{
		media.col(i) = mean(matriz.col(i)).val[0];
	}
}

void mostrarImagemMedia(){
	Mat to_show;

	media.reshape(1,altura).convertTo(to_show, CV_8UC1, 255);

	imshow("Media", to_show);
	waitKey();
	destroyAllWindows();
}

void normalizarMatriz(){
	for (int i = 0; i < num_rostos_treino; i ++)
	{
		subtract(matriz.row(i), media, matriz.row(i));
	}
}

void calcularMatrizDeCovariancia(){
	gemm(matriz, matriz, 1.0, Mat(), 0.0, covar, GEMM_2_T);
}

void carregarAutoObjetos(){
	eigen(covar, autovalores, autovetores);

	autovalores = autovalores.reshape(1,1);

	autovalores = Mat(autovalores, Range::all(), Range(0, num_componentes));
	autovetores = Mat(autovetores, Range::all(), Range(0, num_componentes));

	Mat _autovetores;
	gemm(autovetores, matriz, 1.0, Mat(), 0.0, _autovetores);//, CV_GEMM_A_T);
	autovetores = _autovetores;

	//Normalizar os autovetores
	for(int i = 0; i < num_componentes; i ++)
	{
		Mat vec = autovetores.row(i);
		normalize(vec, vec);
	}
}

void mostrarImagensDosAutovetores(){
	for (int i = 0; i < num_componentes; i ++)
	{
		Mat ev = autovetores.row(i).clone().reshape(1,altura);
		// Reshape to original size & normalize to [0...255] for imshow.
		Mat grayscale;
		normalize(ev, grayscale, 0, 255, NORM_MINMAX, CV_8UC1);
		// Show the image & apply a Jet colormap for better sensing.
		Mat cgrayscale;
		applyColorMap(grayscale, cgrayscale, COLORMAP_JET);

		imshow("Autovetor i", grayscale);
		imshow("Color i", cgrayscale);
		waitKey();
		destroyAllWindows();
	}	
}

void projetarTreino(){
	projecoes = cvCreateMat(num_rostos_treino, num_componentes, TIPO);
	for (int i = 0; i < num_rostos_treino; i ++)
	{
		gemm(matriz.row(i), autovetores, 1.0, Mat(), 0.0, projecoes.row(i), GEMM_2_T);
	}
}

void gerarArquivoTreino(char * nome_arquivo_treino){
	//Declaração de arquivo
	ofstream arquivo;
	string nome = nome_arquivo_treino;

	nome += "_SVM";
	//Abre o arquivo
	arquivo.open(nome);


	for (int i = 0; i < num_rostos_treino; i ++)
	{
		arquivo << classes_treino[i]+1 << " ";
		for (int j = 0; j < num_componentes; j ++)
		{
			arquivo << j+1 << ":" << projecoes.at<double>(i,j) << " ";
		}
		arquivo << endl;
	}

	//Fecha o arquivo
	arquivo.close();
}

void gerarArquivoTeste(char* nome_arquivo_teste){
	//Declaração de arquivo
	ofstream arquivo_saida;

	string nome = nome_arquivo_teste;

	nome += "_SVM";
	//Abre o arquivo
	arquivo_saida.open(nome);

	//Variáveis temporárias para obtenção do caminho e da classe
	char* caminho = (char*) malloc(sizeof(char)*64);
	char* classe;

	//Tratando o arquivo de caminho das imagens
	arquivo = fopen(nome_arquivo_teste,"r");

	//Testa a existência do arquivo
	if(!arquivo || feof(arquivo)){		
		printf("ERRO: O arquivo nao contem imagens de teste, ou nao existe!\n");
		exit(EXIT_FAILURE);
	}//end if

	double quant_acertos = 0.0, quant_testes = 0.0;
	int classe_pos = -1, classe_teste;
	double distancia;
	Mat a_projetar, projetado, rosto;

	//Enquanto o arquivo nao acabar, preencha o vetor com as imagens e as respectivas classes_teste
	while (!feof(arquivo)){

		//Carregando os caminhos e as classes_teste
		fscanf(arquivo,"%s", caminho);
		strtok(caminho, ";");
		classe = strtok(NULL, ";");

		rosto = imread(caminho, CV_LOAD_IMAGE_GRAYSCALE);
		classe_teste = atoi(classe);

		rosto.reshape(1,1).convertTo(a_projetar, TIPO, 1/255.);
		subtract(a_projetar, media, a_projetar);
		gemm(a_projetar, autovetores, 1.0, Mat(), 0.0, projetado, GEMM_2_T);

		arquivo_saida << classe_teste+1 << " ";
		for (int i = 0; i < num_componentes; i ++)
		{
			arquivo_saida << i+1 << ":" << projetado.at<double>(0,i) << " ";
		}
		arquivo_saida << endl;

	}//end while
	
	//Fechando o arquivo e memória alocada
	fclose(arquivo);
	//free(caminho);
}

void carregarTreino(const char* nome_do_arquivo){
	//Variáveis temporárias para obtenção do caminho e da classe
	char* caminho = (char*) malloc(sizeof(char)*64);
	char* classe;

	//Tratando o arquivo de caminho das imagens
	arquivo = fopen(nome_do_arquivo,"r");

	//Testa a existência do arquivo
	if(!arquivo || feof(arquivo)){		
		printf("ERRO: O arquivo nao contem imagens de treino, ou nao existe.\n");
		exit(EXIT_FAILURE);
	}//end if

	//Obtendo a primeira linha
	fscanf(arquivo,"%s", caminho);
	strtok(caminho, ";");
	classe = strtok(NULL, ";");//Obs.: É automaticamente atualizada a cada iteração

	//Alocando nos vetores a imagem carregada e sua classe
	treino.push_back(imread(caminho,CV_LOAD_IMAGE_GRAYSCALE));
	classes_treino.push_back(atoi(classe));
	num_classes = 1;
	int quant_classe_i = 1;

	//Enquanto o arquivo nao acabar, preencha o vetor com as imagens e as respectivas classes
	while (!feof(arquivo)){
		//Carregando os caminhos e as classes
		fscanf(arquivo,"%s", caminho);
		strtok(caminho, ";");
		classe = strtok(NULL, ";");

		//Alocando nos vetores a imagem carregada e sua classe
		treino.push_back(imread(caminho,CV_LOAD_IMAGE_GRAYSCALE));
		classes_treino.push_back(atoi(classe));

		//Verifica se as classes anteriores são compatíveis, senão, existe mais uma classe
		if (classes_treino.back() != classes_treino[classes_treino.size()-2]){
			quant_por_classe.push_back(quant_classe_i);
			num_classes ++;
			quant_classe_i = 0;
		}
		quant_classe_i ++;
	}//end while

	//Última adição
	quant_por_classe.push_back(quant_classe_i);

	//Guarda os dados necessários
	largura = treino[0].cols;
	altura = treino[0].rows;
	num_pixels = altura * largura;
	num_rostos_treino = treino.size();

	num_componentes = num_classes;// - 1;

	//Fechando o arquivo e memória alocada
	fclose(arquivo);
	//free(caminho);
}