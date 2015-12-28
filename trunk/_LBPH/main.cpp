#include <stdio.h>
#include <vector>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

FILE* arquivo;
vector<Mat> treino, histogramas;
vector<int> classes_treino;
int altura, largura, num_pixels, num_rostos_treino, num_classes;

bool valido(Mat imagem, int indice_i, int indice_j, int acesso_i, int acesso_j);
Mat obterImagemLBP(Mat imagem);
void carregarTreino(const char* nome_do_arquivo);
void criarHistogramas();
Mat obterHistogramaLBP(Mat rosto);
void classificar(Mat rosto, int& classe_pos, double& distancia);
void classificar(char* nome_do_arquivo, double& porcentagem_acerto);

int main(int argc, char* argv[]){
	char* nome_arquivo_treino = argv[1];
	char* nome_arquivo_teste;
	Mat teste;
	int classe_pos = -1;
	double distancia = DBL_MAX, porcentagem_acerto = -1.0;

	system("cls");
	printf("--------------------------- LBPH ------------------------------\n");
	printf("Loading training data...\n");
	carregarTreino(nome_arquivo_treino);

	printf("Creating training histograms...\n");
	criarHistogramas();
	
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
	return EXIT_SUCCESS;
}

bool valido(Mat imagem, int indice_i, int indice_j, int acesso_i, int acesso_j){
	return !(((indice_i + acesso_i) < 0 || (indice_j + acesso_j) < 0) ||
		((indice_i + acesso_i) >= imagem.rows || (indice_j + acesso_j) >= imagem.cols));
}

Mat obterImagemLBP(Mat imagem){
	unsigned char temp;
	Mat imagem_retorno = cvCreateMat(imagem.rows, imagem.cols, imagem.type());
	for (int i = 0; i < imagem.rows; i ++)
	{
		for (int j = 0; j < imagem.cols; j ++)
		{
			//Zerando o temporário
			temp = 0;
			temp = temp << 1 | (valido(imagem, i, j, -1, -1) && (imagem.at<unsigned char>(i-1, j-1) >= imagem.at<unsigned char>(i, j)));
			temp = temp << 1 | (valido(imagem, i, j, -1, 0) && (imagem.at<unsigned char>(i-1, j) >= imagem.at<unsigned char>(i, j)));
			temp = temp << 1 | (valido(imagem, i, j, -1, 1) && (imagem.at<unsigned char>(i-1, j+1) >= imagem.at<unsigned char>(i, j)));
			temp = temp << 1 | (valido(imagem, i, j, 0, 1) && (imagem.at<unsigned char>(i, j+1) >= imagem.at<unsigned char>(i, j)));
			temp = temp << 1 | (valido(imagem, i, j, 1, 1) && (imagem.at<unsigned char>(i+1, j+1) >= imagem.at<unsigned char>(i, j)));
			temp = temp << 1 | (valido(imagem, i, j, 1, 0) && (imagem.at<unsigned char>(i+1, j) >= imagem.at<unsigned char>(i, j)));
			temp = temp << 1 | (valido(imagem, i, j, 1, -1) && (imagem.at<unsigned char>(i+1, j-1) >= imagem.at<unsigned char>(i, j)));
			temp = temp << 1 | (valido(imagem, i, j, 0, -1) && (imagem.at<unsigned char>(i, j-1) >= imagem.at<unsigned char>(i, j)));

			imagem_retorno.at<unsigned char>(i, j) = temp;
		}
	}
	return imagem_retorno;
}

void criarHistogramas(){
	//Estabelecer o tamanho do histograma
	int histSize = 256;

	//Define os limites
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	histogramas.resize(treino.size());

	for (int i = 0; i < num_rostos_treino; i++)
	{
		//Calcula os Histogramas
		calcHist(&treino[i], 1, 0, Mat(), histogramas[i], 1, &histSize, &histRange);
	}
}

Mat obterHistogramaLBP(Mat rosto){
	Mat histograma, rosto_lbp;

	//Estabelecer o tamanho do histograma
	int histSize = 256;

	//Define os limites
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	
	rosto_lbp = obterImagemLBP(rosto);

	//Calcula o histograma
	calcHist(&rosto_lbp, 1, 0, Mat(), histograma, 1, &histSize, &histRange);

	return histograma;
}

void classificar(Mat rosto, int& classe_pos, double& distancia){
	double distancia_atual;
	Mat histograma_lbp;

	histograma_lbp = obterHistogramaLBP(rosto);

	for (int i = 0; i < num_rostos_treino; i ++){
		distancia_atual = compareHist(histograma_lbp, histogramas[i], CV_COMP_CHISQR);
		
		cout << distancia_atual << endl;

		if (distancia_atual < distancia)
		{
			distancia = distancia_atual;
			classe_pos = i;
		}
	}
	histograma_lbp.release();
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
	free(caminho);
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
	treino.push_back(obterImagemLBP(imread(caminho,CV_LOAD_IMAGE_GRAYSCALE)));
	classes_treino.push_back(atoi(classe));

	//Enquanto o arquivo nao acabar, preencha o vetor com as imagens e as respectivas classes
	while (!feof(arquivo)){
		//Carregando os caminhos e as classes
		fscanf(arquivo,"%s", caminho);
		strtok(caminho, ";");
		classe = strtok(NULL, ";");

		//Alocando nos vetores a imagem carregada e sua classe
		treino.push_back(obterImagemLBP(imread(caminho,CV_LOAD_IMAGE_GRAYSCALE)));
		classes_treino.push_back(atoi(classe));
	}//end while

	//Guarda os dados necessários
	largura = treino[0].cols;
	altura = treino[0].rows;
	num_pixels = altura * largura;
	num_rostos_treino = treino.size();
	
	//Fechando o arquivo e memória alocada
	fclose(arquivo);
	free(caminho);
}