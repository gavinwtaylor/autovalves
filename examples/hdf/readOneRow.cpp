#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <string>
#include "H5Cpp.h"
using namespace H5;

int main() {

  H5File file("test.h5",H5F_ACC_RDONLY);//TRUNC means overwrite existing
                                       //EXCL fails if exists already
                                       //RDONLY read only
                                       //RDWR read/write if already exists
  DataSet dataset=file.openDataSet("testDataset");
  DataSpace dataspace=dataset.getSpace();

  H5T_class_t type_class=dataset.getTypeClass();
  if (type_class != H5T_FLOAT)
    std::cerr << "NOT A DOUBLE" << std::endl;

  int rank=dataspace.getSimpleExtentNdims();

  hsize_t* setSize = new hsize_t[rank];//datasetDimension
  hsize_t* readSize = new hsize_t[rank];//chunk to be read size

  dataspace.getSimpleExtentDims(setSize,NULL);

  readSize[0]=2;
  readSize[1]=setSize[1];

  cerr << setSize[0] << ',' << setSize[1] << endl;
  cerr << readSize[0] << ',' << readSize[1] << endl;

  double* forReading=new double[readSize[0]*readSize[1]]; //MUST BE 1D

  hsize_t offset[2]={1,0};
  hsize_t count[2]={readSize[0],readSize[1]};

  DataSpace memspace(rank,count,NULL);

  dataspace.selectHyperslab(H5S_SELECT_SET,count,offset);
  
  dataset.read(forReading,PredType::NATIVE_DOUBLE,memspace,dataspace);

  for (int row = 0; row < readSize[0]; row++) {
    for (int col = 0; col < readSize[1]; col++){
      cerr << '(' << row << ',' << col << ',' << (row*readSize[1]+col) << ')';
      cerr << forReading[row*readSize[1]+col] << ' ';
    }
    cerr << endl;
  }
  delete [] forReading;
}
