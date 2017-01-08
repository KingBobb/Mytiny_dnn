
#include<iostream>
#include<string>
#include"util/nn_error.h"
using namespace mytiny_dnn;

using namespace std;

void sample1_convnet(const string& data_dir = "D:/learning/Graduating/tiny-dnn-master/data");

int main(int argc, char** argv)
{
	try
	{
		if (argc == 2)
		{
			sample1_convnet(argv[1]);
		}
		else
		{
			sample1_convnet();
		}
	}
	catch (const nn_error&e)
	{
		std::cout << e.what() << std::endl;
	}

}

void sample1_convnet(const string&data_dir)
{

	cout << "hello world!" << endl;
	system("pause");
}