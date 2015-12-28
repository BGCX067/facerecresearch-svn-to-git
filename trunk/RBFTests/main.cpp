#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <annie.h>

using namespace cv;
using namespace std;
using namespace annie;

#define TIPO CV_64FC1

FILE* arquivo;
vector<Mat> treino;

//Possuem as classes de treino e a quantidade de imagens por classe
vector<int> classes_treino, quant_por_classe;

int altura, largura, num_pixels, num_rostos_treino, num_classes, num_componentes;
Mat matriz, media, covar, icovar, autovalores, autovetores, projecoes;

//Ponteiro para a rede RBF
RadialBasisNetwork *rbf;
//Ponteiro para o vetor de saidas da rede
annie::real ** saidasRBF;

void carregarTreino(const char* nome_do_arquivo);
void criarMatriz();
void obterMatrizMedia();
void mostrarImagemMedia();
void normalizarMatriz();
void calcularMatrizDeCovariancia();
void carregarAutoObjetos();
void mostrarImagensDosAutovetores();
void projetarTreino();
void treinarRede();
void classificar(Mat rosto, int& classe_pos, double& distancia);
void classificar(char* nome_do_arquivo, double& porcentagem_acerto);

int main(int argc, char* argv[]){
	//argc = 3;
	//argv[1] = "treino_at.dat";
	char* nome_arquivo_treino = argv[1];
	char* nome_arquivo_teste;
	Mat teste;
	int classe_pos = -1;
	double distancia = DBL_MAX, porcentagem_acerto = -1.0;

	system("cls");
	printf("--------------------------- RBF Network ------------------------------\n");
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

	printf("Training network...\n");
	treinarRede();

	//argv[2] = "at\\s1\\2.pgm";
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
	gemm(autovetores, matriz, 1.0, Mat(), 0.0, _autovetores, CV_GEMM_A_T);
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

void treinarRede(){
	try{
		rbf = new RadialBasisNetwork(num_componentes, num_classes, 1);

		TrainingSet treino (num_componentes, 1);

		annie::real ** treino_RBF;

		treino_RBF = (annie::real **) malloc (sizeof(annie::real *) * num_classes);

		for (int i = 0; i < num_classes; i ++)
		{
			treino_RBF[i] = (annie::real *) malloc (sizeof(annie::real) * num_componentes);
			for (int j = 0; j < num_componentes; j ++)
			{
				treino_RBF[i][j] = projecoes.at<double>(i,j);
			}//end for
		}//end for

		annie::real ** saidas = (annie::real **) malloc (sizeof(annie::real *) * num_classes);
		
		//Utilizadas após o treino
		saidasRBF = (annie::real **) malloc (sizeof(annie::real *) * num_classes);

		for (int i = 0; i < num_classes; i ++)
		{
			saidas[i] = (annie::real *) malloc (sizeof(annie::real));
			saidas[i][0] = i;
			treino.addIOpair(treino_RBF[i], saidas[i]);//Adiciona os pares I/O
			rbf->setCenter(i, treino_RBF[i]);//Seta os centros
			//cout << "Weight " << i << "  " << rbf->getWeight(i,0) << endl;
		}//end for

		//Remove os Bias
		for (int i = 0; i < treino.getOutputSize(); i ++)
		{
			rbf->removeBias(i);
		}

		/*cout << "Before training results" << endl;
		for (int i = 0; i < num_classes; i ++)
		{
			cout << rbf->getOutput(treino_RBF[i])[0] << endl;
		}//end for*/
		
		//Treinando a rede
		rbf->trainWeights(treino);

		//cout << "\nAfter training results" << endl;
		for (int i = 0; i < num_classes; i ++)
		{
			saidasRBF[i] = (annie::real *) malloc (sizeof(annie::real));
			saidasRBF[i][0] = rbf->getOutput(treino_RBF[i])[0];
			//cout << saidasRBF[i][0] << endl;
		}//end for

	}//end try
	catch (annie::Exception &e)
	{
		cerr << "ERRO: " << e.what() << endl;
	}
}

void classificar(Mat rosto, int& classe_pos, double& distancia){
	double distancia_atual;
	Mat a_projetar, projetado;

	rosto.reshape(1,1).convertTo(a_projetar, TIPO, 1/255.);
	subtract(a_projetar, media, a_projetar);
	gemm(a_projetar, autovetores, 1.0, Mat(), 0.0, projetado, GEMM_2_T);

	//////////////////////////////////////////RBF Test//////////////////////////////////////
	try{
		annie::real * entrada;

		entrada = (annie::real *) malloc(sizeof(annie::real) * num_componentes);
		for (int j = 0; j < num_componentes; j ++)
		{
			entrada[j] = projetado.at<double>(0,j);
		}//end for

		//cout << rbf->getOutput(entrada)[0] << endl;

		for (int i = 0; i < num_rostos_treino; i ++)
		{
			distancia_atual = abs(saidasRBF[i][0] - rbf->getOutput(entrada)[0]);

			if (distancia_atual < distancia)
			{
				distancia = distancia_atual;
				classe_pos = i;
			}//end if
		}//end for
		//cout << "\nDistancia encontrada: " << distancia << endl;
	}//end try
	catch (annie::Exception &e)
	{
		cerr << "ERRO: " << e.what() << endl;
	}
	//////////////////////////////////////////////////////////////////////////////////
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

	num_componentes = num_classes - 1;

	//Fechando o arquivo e memória alocada
	fclose(arquivo);
	free(caminho);
}