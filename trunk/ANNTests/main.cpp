#include <annie.h>

using namespace annie;
using namespace std;

int main(){
	//Define as entradas e saídas do treino
	real entrada1[]={0,0};
	real saida1[]={0};
	real entrada2[]={1,0};
	real saida2[]={1};
	real entrada3[]={0,1}; 
	real saida3[]={1};
	real entrada4[]={1,1}; 
	real saida4[]={0};


	try{
		//Treina duas entradas e uma saída
		TrainingSet treino(2,1);

		//Adiciona as entradas ao treino
		treino.addIOpair(entrada1,saida1);
		treino.addIOpair(entrada2,saida2);
		treino.addIOpair(entrada3,saida3);
		treino.addIOpair(entrada4,saida4);

		//RBF com 2 entradas, 2 centros e 1 saída
		RadialBasisNetwork rbf(2,2,1);

		real centro1[2]={0,0};
		real centro2[2]={1,1};
		rbf.setCenter(0,centro1);
		rbf.setCenter(1,centro2);

		//Vetor que irá conter as saídas
		VECTOR saidas;
		
		cout<<"Resultados antes do treino:"<<endl;

		saidas = rbf.getOutput(entrada1);
		cout<<"0 XOR 0 = "<<saidas[0]<<endl;

		saidas = rbf.getOutput(entrada2);
		cout<<"1 XOR 0 = "<<saidas[0]<<endl;

		saidas = rbf.getOutput(entrada3);
		cout<<"0 XOR 1 = "<<saidas[0]<<endl;

		saidas = rbf.getOutput(entrada4);
		cout<<"1 XOR 1 = "<<saidas[0]<<endl;

		//Treinando a rede
		rbf.trainWeights(treino);

		cout<<"\nResultados depois do treino:"<<endl;

		saidas = rbf.getOutput(entrada1);
		cout<<"0 XOR 0 = "<<saidas[0]<<endl;

		saidas = rbf.getOutput(entrada2);
		cout<<"1 XOR 0 = "<<saidas[0]<<endl;

		saidas = rbf.getOutput(entrada3);
		cout<<"0 XOR 1 = "<<saidas[0]<<endl;

		saidas = rbf.getOutput(entrada4);
		cout<<"1 XOR 1 = "<<saidas[0]<<endl;

		cout << "\nCentros" << endl;
		cout << rbf.getWeight(0,0) << endl;
		cout << rbf.getWeight(1,0) << endl;

		real entrada[2] = {2,2};
		cout << "Saida para " << entrada[0] << ", " << entrada[1] << " = " << rbf.getOutput(entrada)[0] << endl;
	}
	catch (Exception &e)
	{
		cout << e.what();
	}

	system("pause");
	return 0;
}