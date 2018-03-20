#include<iostream>  //cout
#include<fstream>   //ifstream

// -----------------------VECTOR HERE-----------------
using namespace std;

int main(int argc, char** argv) {
  // Open a file with ifstream, select the mode (binary) and put the output position at the end of the file (ate).
  // argv[1] -> The name of the file.
  ifstream ifs(argv[1], ios::binary|ios::ate);

  // tellg -> Returns the position of the current character in the input stream. "pos" will store the length of the file.
  ifstream::pos_type pos = ifs.tellg();

  char* result = new char[pos];

  //seekg -> Sets the position of the next character, in this case to the beginning of the stream (ios::beg).
  ifs.seekg(0, ios::beg);
  // Extracts "pos" characteres from the stream "result"
  ifs.read(result, pos);

  double* output = (double*) result;

  int row = atoi(argv[2]);
  int col = atoi(argv[3]);

  for(int i = 0; i < row; ++i){
    for(int j=0; j < col; ++j){
    	cout << output[i * col + j] << " ";
    }
    cout << endl;
  }
}
