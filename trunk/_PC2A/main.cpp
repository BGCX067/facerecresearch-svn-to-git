#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define TIPO CV_64FC1

FILE* arquivo;
vector<Mat> treino;
vector<int> classes_treino;
int altura, largura, num_pixels, num_rostos_treino, num_classes, num_componentes;
Mat matriz, media, covar, icovar, autovalores, autovetores, projecoes;

vector<Mat> mapas_de_projecoes, projecoes_combinadas;
double combinacao;

void carregarTreino(const char* nome_do_arquivo);
void criarMatriz();
void obterMatrizMedia();
void mostrarImagemMedia();
void normalizarMatriz();
void calcularMatrizDeCovariancia();
void carregarAutoObjetos();
void mostrarImagensDosAutovetores();
void projetarTreino();
void classificar(Mat rosto, int& classe_pos, double& distancia);
void classificar(char* nome_do_arquivo, double& porcentagem_acerto);

void definirMapa(int treino_pos);
void projetarIntegrais(int treino_pos, Mat& integraisVerticais, Mat& integraisHorizontais);
void definirProjecoesCombinadas(double combinacao);
void definirMapa(Mat rosto, Mat& mapa);
void projetarIntegrais(Mat rosto, Mat& integraisVerticais, Mat& integraisHorizontais);

int main(int argc, char* argv[]){
	char* nome_arquivo_treino = argv[1];
	char* nome_arquivo_teste;
	Mat teste;
	int classe_pos = -1;
	double distancia = DBL_MAX, porcentagem_acerto = -1.0;

	/*argc = 4;
	nome_arquivo_treino = "masculino_unica_amostra.dat";
	argv[2] = "teste_masculino_one_sample.dat";
	argv[3] = "0.91";
		*/
	if (argc < 4)
	{
		printf("Uso: %s <nome_arquivo_treino.ext> <nome_arquivo_teste.ext | nome_imagem_teste.ext> <alpha>");
		exit(EXIT_FAILURE);
	}//end if

	system("cls");
	printf("--------------------------- (PC)2A ------------------------------\n");
	printf("Loading training data...\n");
	carregarTreino(nome_arquivo_treino);

	printf("Generating the projection-combined versions...\n");
	definirProjecoesCombinadas(atof(argv[3]));
	
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

	printf("Classifying...\n");
	if (strstr(argv[2], ".dat")){
		nome_arquivo_teste = argv[2];
		classificar(nome_arquivo_teste, porcentagem_acerto);
		printf("O metodo obteve %3.4f%% de acerto!\n", porcentagem_acerto);
	}else{
		teste = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
		classificar(teste, classe_pos, distancia);
		imshow ("Resultado", treino[classe_pos]);
		waitKey(0);
		destroyAllWindows();
	}

	printf("Operation Done!");
	return 0;
}

void criarMatriz(){
	matriz = cvCreateMat(num_rostos_treino, num_pixels, TIPO);

	for(int i = 0; i < num_rostos_treino; i ++){
		Mat Xi = matriz.row(i);

		if(projecoes_combinadas[i].isContinuous()){
			projecoes_combinadas[i].reshape(1, 1).convertTo(Xi, TIPO, 1/255.);
		}//end if
		else{
			projecoes_combinadas[i].clone().reshape(1, 1).convertTo(Xi, TIPO, 1/255.);
		}//end else
	}
}

void definirMapa(int treino_pos){
	Mat Vxs(1, altura, CV_64FC1), Hys(1, largura, CV_64FC1);

	double _P = mean(treino[treino_pos])[0];

	projetarIntegrais(treino_pos, Vxs, Hys);
	
	for (int i = 0; i < altura; i ++)
	{
		for (int j = 0; j < largura; j ++)
		{
			mapas_de_projecoes[treino_pos].at<double>(i, j) =
				(Vxs.at<double>(i)*Hys.at<double>(j))/(Vxs.total()*Hys.total()*_P);
		}//end for
	}//end for
}

void projetarIntegrais(int treino_pos, Mat& Vxs, Mat& Hys){
	double Vx, Hy;
	for (int x = 0; x < altura; x ++)
	{
		Vx = 0.0;
		for (int j = 0; j < largura; j ++)
		{
			Vx += treino[treino_pos].at<unsigned char>(x, j);
		}//end for
		Vxs.at<double>(0, x) = Vx;
	}//end for

	for (int y = 0; y < largura; y ++)
	{
		Hy = 0.0;
		for (int j = 0; j < altura; j ++)
		{
			Hy += treino[treino_pos].at<unsigned char>(j, y);
		}//end for
		Hys.at<double>(0, y) = Hy;
	}//end for
}

void definirProjecoesCombinadas(double combinacao){
	for (size_t i = 0; i < num_rostos_treino; i ++)
	{
		mapas_de_projecoes.push_back(Mat(altura, largura, CV_64FC1));
		projecoes_combinadas.push_back(Mat(altura, largura, CV_64FC1));

		definirMapa(i);

		for (int x = 0; x < altura; x ++)
		{
			for (int y = 0; y < largura; y ++)
			{
				projecoes_combinadas[i].at<double>(x, y) = 
					(treino[i].at<unsigned char>(x, y) +
					(combinacao * mapas_de_projecoes[i].at<double>(x, y)))
					/
					(1 + combinacao);
			}//end for
		}//end for
		
		projecoes_combinadas[i].convertTo(projecoes_combinadas[i], CV_8UC1);
		mapas_de_projecoes[i].convertTo(mapas_de_projecoes[i], CV_8UC1);
		imshow("Rosto1", treino[i]);
		imshow("Mapa", mapas_de_projecoes[i]);
		imshow("Proj1", projecoes_combinadas[i]);
		waitKey();
		destroyAllWindows();
	}//end for

}

void definirMapa(Mat rosto, Mat& mapa){
	Mat Vxs(1, altura, CV_64FC1), Hys(1, largura, CV_64FC1);

	double _P = mean(rosto)[0];

	projetarIntegrais(rosto, Vxs, Hys);

	for (int i = 0; i < altura; i ++)
	{
		for (int j = 0; j < largura; j ++)
		{
			mapa.at<double>(i, j) =
				(Vxs.at<double>(i)*Hys.at<double>(j))/(Vxs.total()*Hys.total()*_P);
		}//end for
	}//end for
}

void projetarIntegrais(Mat rosto, Mat& integraisVerticais, Mat& integraisHorizontais){
	double Vx, Hy;
	for (int x = 0; x < altura; x ++)
	{
		Vx = 0.0;
		for (int j = 0; j < largura; j ++)
		{
			Vx += rosto.at<unsigned char>(x, j);
		}//end for
		integraisVerticais.at<double>(0, x) = Vx;
	}//end for

	for (int y = 0; y < largura; y ++)
	{
		Hy = 0.0;
		for (int j = 0; j < altura; j ++)
		{
			Hy += rosto.at<unsigned char>(j, y);
		}//end for
		integraisHorizontais.at<double>(0, y) = Hy;
	}//end for
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

	Mat _autovetores(num_componentes, num_pixels, TIPO);
	gemm(autovetores, matriz, 1.0, Mat(), 0.0, _autovetores);
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

void classificar(Mat rosto, int& classe_pos, double& distancia){
	double distancia_atual;
	Mat a_projetar, projetado;
	Mat rosto_combinado(altura, largura, CV_64FC1), mapa(altura, largura, CV_64FC1);
	
	definirMapa(rosto, mapa);

	for (int x = 0; x < altura; x ++)
	{
		for (int y = 0; y < largura; y ++)
		{
			rosto_combinado.at<double>(x, y) = 
				(rosto.at<unsigned char>(x, y) +
				(combinacao * mapa.at<double>(x, y)))
				/
				(1 + combinacao);
		}
	}
	
	rosto_combinado.reshape(1,1).convertTo(a_projetar, TIPO, 1/255.);
	subtract(a_projetar, media, a_projetar);
	gemm(a_projetar, autovetores, 1.0, Mat(), 0.0, projetado, GEMM_2_T);

	for (int i = 0; i < num_rostos_treino; i ++)
	{
		distancia_atual = norm(projecoes.row(i), projetado, NORM_L2);

		if (distancia_atual < distancia)
		{
			distancia = distancia_atual;
			classe_pos = i;
		}
	}
	a_projetar.release();
	projetado.release();
}

void classificar(char* nome_do_arquivo, double& porcentagem_acerto){
	//Variáveis temporárias para obtenção do caminho e da classe
	char* caminho = (char*) malloc(sizeof(char)*64);
	char* classe;

	//Tratando o arquivo de caminho das imagens
	arquivo = fopen(nome_do_arquivo,"r");

	//Testa a existência do arquivo
	if(!arquivo || feof(arquivo)){		
		printf("ERRO: O arquivo nao contem imagens de teste, ou nao existe!\n");
		exit(EXIT_FAILURE);
	}//end if

	double quant_acertos = 0.0, quant_testes = 0.0;
	int classe_pos = -1, classe_teste;
	double distancia;
	Mat rosto;

	//Enquanto o arquivo nao acabar, preencha o vetor com as imagens e as respectivas classes_teste
	while (!feof(arquivo)){
		classe_pos = -1;
		distancia = DBL_MAX;

		//Carregando os caminhos e as classes_teste
		fscanf(arquivo,"%s", caminho);
		strtok(caminho, ";");
		classe = strtok(NULL, ";");

		rosto = imread(caminho, CV_LOAD_IMAGE_GRAYSCALE);
		classe_teste = atoi(classe);

		//Retorna a posição no vetor de faces da classe identificada
		classificar(rosto, classe_pos, distancia);

		if (classes_treino[classe_pos] == classe_teste)
		{
			quant_acertos ++;
		}
		quant_testes ++;
	}//end while

	porcentagem_acerto = (quant_acertos/quant_testes)*100.0;

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

	//Enquanto o arquivo nao acabar, preencha o vetor com as imagens e as respectivas classes
	while (!feof(arquivo)){
		//Carregando os caminhos e as classes
		fscanf(arquivo,"%s", caminho);
		strtok(caminho, ";");
		classe = strtok(NULL, ";");

		//Alocando nos vetores a imagem carregada e sua classe
		treino.push_back(imread(caminho,CV_LOAD_IMAGE_GRAYSCALE));
		classes_treino.push_back(atoi(classe));
	}//end while

	//Guarda os dados necessários
	largura = treino[0].cols;
	altura = treino[0].rows;
	num_pixels = altura * largura;
	num_rostos_treino = treino.size();

	num_componentes = num_rostos_treino;

	//Fechando o arquivo e memória alocada
	fclose(arquivo);
	//free(caminho);
}