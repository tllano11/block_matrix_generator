#include<iostream>
#include<fstream>
#include<vector>

using namespace std;

int main(int argc, char** argv) {
    ifstream ifs(argv[1], ios::binary|ios::ate);
    ifstream::pos_type pos = ifs.tellg();

    char* result = new char[pos];

    ifs.seekg(0, ios::beg);
    ifs.read(result, pos);

    long double* output = (long double*) result;

    int row = atoi(argv[2]);
    int col = atoi(argv[3]);

    for(int i = 0; i < row; ++i){
      for(int j=0; j < col; ++j){
	cout << output[i * col + j] << " ";
      }
      cout << endl;
    }
}
