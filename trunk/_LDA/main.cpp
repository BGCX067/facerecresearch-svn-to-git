#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define TIPO CV_64FC1

FILE* arquivo;
vector<Mat> treino;

//Possuem as classes de treino e a quantidade de imagens por classe
vector<int> classes_treino, quant_por_classe;

int altura, largura, num_pixels, num_rostos_treino;
int num_classes, num_componentes, nova_dimensao;

//Possui médias de cada classe
vector<Mat> medias;

//Possui a média total
Mat media;

//Ponteiro para a matriz
Mat* matriz;

//Possui as projeções em suas linhas
Mat projecoes;

//Possui em cada posição as imagens de cada classe nas linhas
vector<Mat> matrizes;

//Possui as matrizes de covariância de cada classe
vector<Mat> covars;

//Utilizado para a redução de dimensionalidade
PCA pca;

//Guarda as projeções das imagens reduzidas dimensionalmente
Mat projetadasPorPCA;

//Possui a matriz de espalhamento interno (Sw) e a de espalhamento entre as classes (Sb)
Mat Sw, Sb;

//Auto-objetos
Mat autovalores, autovetores;

void carregarTreino(const char* nome_do_arquivo);
void criarMatrizes();
void obterMedias();
void calcularMatrizesDeCovariancia();
void calcularEspalhamento();
void fazerPCA();
void normalizarMatrizes();
void calcularAutoObjetos();
void mostrarImagensDosAutovetores();
void projetarTreino();
void classificar(Mat rosto, int& classe_pos, double& distancia);
void classificar(char* nome_do_arquivo, double& porcentagem_acerto);

int main(int argc, char* argv[]){
	char* nome_arquivo_treino = "treino_at_0-9.dat"; //argv[1];
	char* nome_arquivo_teste;
	Mat teste;
	int classe_pos = -1;
	double distancia = DBL_MAX, porcentagem_acerto = -1.0;

	system("cls");
	printf("--------------------------- LDA ------------------------------\n");
	printf("Loading training data...\n");
	carregarTreino(nome_arquivo_treino);

	printf("Reducing dimensionality...\n");
	fazerPCA();

	printf("Creating row matrices...\n");
	criarMatrizes();

	printf("Getting mean matrix...\n");
	obterMedias();

	printf("Normalizing matrices...\n");
	normalizarMatrizes();
	
	printf("Calculating within-class and between-class scatter matrices...\n");
	calcularEspalhamento();

	printf("Calculating eigen-objects...\n");
	calcularAutoObjetos();

	//mostrarImagensDosAutovetores();

	printf("Projecting training samples...\n");
	projetarTreino();

	printf("Classifying...\n");
	//if (strstr(argv[2], ".dat")){
		nome_arquivo_teste = "teste_at.dat"; //argv[2];
		classificar(nome_arquivo_teste, porcentagem_acerto);
		printf("O metodo obteve %3.4f%% de acerto!\n", porcentagem_acerto);
/*	}else{
		teste = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
		classificar(teste, classe_pos, distancia);
		imshow ("Resultado", treino[classe_pos]);
		waitKey(0);
		destroyAllWindows();
	}
	*/
	printf("Operation Done!");
	system("pause");
	return 0;
}

void fazerPCA(){
	matriz = new Mat(num_rostos_treino, num_pixels, TIPO);

	for(int i = 0; i < num_rostos_treino; i ++){
		Mat Xi = matriz->row(i);

		if(treino[i].isContinuous()){
			treino[i].reshape(1, 1).convertTo(Xi, TIPO);
		}
		else{
			treino[i].clone().reshape(1, 1).convertTo(Xi, TIPO);
		}
	}
	
	pca((*matriz), Mat(), CV_PCA_DATA_AS_ROW, (num_rostos_treino - num_classes));
	
	projetadasPorPCA = pca.project((*matriz));

	nova_dimensao = pca.eigenvectors.rows;
}

void criarMatrizes(){
	//Iterador do vetor
	int k = 0;

	//Para criação do vetor de grupo de matrizes agrupado por classes
	for(int i = 0; i < num_classes; i ++){
		int quant = quant_por_classe[i];
		Mat matriz_atual = cvCreateMat(quant, nova_dimensao, TIPO);
		
		for(int j = 0; j < quant; j ++){
			projetadasPorPCA.row(k).copyTo(matriz_atual.row(j));
			k ++;
		}
		matrizes.push_back(matriz_atual);
	}
}

void obterMedias(){
	media = Mat::zeros(1, nova_dimensao, TIPO);
	for(int i = 0; i < num_classes; i ++)
	{
		Mat media_atual = Mat::zeros(1, nova_dimensao, TIPO);
		for(int j = 0; j < quant_por_classe[i]; j ++)
		{
			add(media_atual, matrizes[i].row(j), media_atual);
			add(media, matrizes[i].row(j), media);
		}
		media_atual.convertTo(media_atual, TIPO, 1.0/static_cast<double>(quant_por_classe[i]));
		medias.push_back(media_atual);
	}
	media.convertTo(media, TIPO, 1.0/static_cast<double>(num_rostos_treino));
}

void normalizarMatrizes(){
	for (int i = 0; i < num_classes; i ++)
	{
		for (int j = 0; j < quant_por_classe[i]; j ++)
		{
			subtract(matrizes[i].row(j), medias[i], matrizes[i].row(j));
		}//end for
	}//end for

	//Liberando memória
	matrizes.clear();
}

void calcularEspalhamento(){	
	//Dá dimensão para o cálculo intra-classe
	Sw = Mat::zeros(nova_dimensao, nova_dimensao, TIPO);

	//Calcula o espalhamento dentro das classes
	mulTransposed(projetadasPorPCA, Sw, true);
	
	Sb = Mat::zeros(nova_dimensao, nova_dimensao, TIPO);

	//Calcula o espalhamento entre as classes
	for (int i = 0; i < num_classes; i++) {
		Mat tmp;
		subtract(medias[i], media, tmp);
		mulTransposed(tmp, tmp, true);
		add(Sb, tmp, Sb);
	}
}

void calcularAutoObjetos(){
	//Inversão da Matriz Sw
	Mat Swi = Sw.inv();
	
	//M = inv(Sw)*Sb
	Mat M;
	gemm(Swi, Sb, 1.0, Mat(), 0.0, M);
	
	//Obtendo os autovalores e os autovetores correspondentes
	eigen(M, autovalores, autovetores);
	
	autovalores = autovalores.reshape(1,1);
	
	//cout << autovalores << endl;

	//TODO: Arrumar erro de criação da matriz de projeção. Olhar FisherFaces após termino do LDA
	autovalores = Mat(autovalores, Range::all(), Range(0, num_componentes));
	autovetores = Mat(autovetores, Range::all(), Range(0, num_componentes));
		
	Mat _autovetores;
	gemm(pca.eigenvectors, autovetores, 1.0, Mat(), 0.0, _autovetores, GEMM_1_T);
	autovetores = _autovetores;
}

void mostrarImagensDosAutovetores(){
	for (int i = 0; i < num_componentes; i ++)
	{
		Mat ev = autovetores.col(i).clone().reshape(1,altura);
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
	Mat X;
	matriz->convertTo(X, TIPO);

	projecoes = cvCreateMat(num_rostos_treino, num_componentes, TIPO);
	
	//Subtrai treino da média do PCA
	for(int i = 0; i < X.rows; i++) {
		Mat linha_i = X.row(i);
		subtract(linha_i, pca.mean.reshape(1,1), linha_i);
		
		//Calcula a projeção como projecoes[i] = (X-média)*W
		gemm(X.row(i), autovetores, 1.0, Mat(), 0.0, projecoes.row(i));
	}

	//Liberando a memória
	delete matriz;
}

void classificar(Mat rosto, int& classe_pos, double& distancia){
	double distancia_atual;
	Mat a_projetar, projetado;

	rosto.reshape(1,1).convertTo(a_projetar, TIPO);
	subtract(a_projetar, pca.mean.reshape(1,1), a_projetar);

	//Calcula a projeção como projecao = (X-média)*W
	gemm(a_projetar, autovetores, 1.0, Mat(), 0.0, projetado);

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
	char caminho[64];
	char* classe;
	char* contexto;

	//Tratando o arquivo de caminho das imagens
	fopen_s(&arquivo, nome_do_arquivo,"r");

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
		fscanf_s(arquivo,"%s", caminho, _countof(caminho));
		strtok_s(caminho, ";", &contexto);
		classe = strtok_s(NULL, ";", &contexto);

		rosto = imread(caminho, CV_LOAD_IMAGE_GRAYSCALE);
		classe_teste = atoi(classe);

		//Retorna a posição no vetor de faces da classe identificada
		classificar(rosto, classe_pos, distancia);

		/*imshow("Esperada", rosto);
		imshow("Predita", treino[classe_pos]);

		waitKey();
		destroyAllWindows();
		*/
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
	char caminho[64];
	char* classe;
	char* contexto;

	//Tratando o arquivo de caminho das imagens
	fopen_s(&arquivo, nome_do_arquivo,"r");

	//Testa a existência do arquivo
	if(!arquivo || feof(arquivo)){		
		printf("ERRO: O arquivo nao contem imagens de treino, ou nao existe.\n");
		exit(EXIT_FAILURE);
	}//end if

	//Obtendo a primeira linha
	fscanf_s(arquivo,"%s", caminho, _countof(caminho));
	strtok_s(caminho, ";", &contexto);
	classe = strtok_s(NULL, ";", &contexto);//Obs.: É automaticamente atualizada a cada iteração

	//Alocando nos vetores a imagem carregada e sua classe
	treino.push_back(imread(caminho, CV_LOAD_IMAGE_GRAYSCALE));
	classes_treino.push_back(atoi(classe));
	num_classes = 1;
	int quant_classe_i = 1;

	//Enquanto o arquivo nao acabar, preencha o vetor com as imagens e as respectivas classes
	while (!feof(arquivo)){
		//Carregando os caminhos e as classes
		fscanf_s(arquivo,"%s", caminho, _countof(caminho));
		strtok_s(caminho, ";", &contexto);
		classe = strtok_s(NULL, ";", &contexto);

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
	//free(caminho);
}