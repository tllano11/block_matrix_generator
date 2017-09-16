#include<iostream>
#include<fstream>
#include<vector>

using namespace std;

int main() {
    ifstream ifs("./mini.out", ios::binary|ios::ate);
    ifstream::pos_type pos = ifs.tellg();

    cout << "Size: " << pos << endl;

    char* result = new char[pos];

    ifs.seekg(0, ios::beg);
    ifs.read(result, pos);

    long double* output = (long double*) result;

    int size = 10;

    for(int i = 0; i < size; ++i){
      for(int j=0; j < size; ++j){
	cout << output[i*size+j] << " ";
      }
      cout << endl;
    }
}
