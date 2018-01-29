#include <iostream>
#include <string>
#include "H5Cpp.h"
using namespace H5;

int main() {

  H5File file("test.h5",H5F_ACC_RDONLY);//TRUNC means overwrite existing
                                       //EXCL fails if exists already
                                       //RDONLY read only
                                       //RDWR read/write if already exists
  DataSet dataset=file.openDataSet("testDataset");
  H5T_class_t type_class=dataset.getTypeClass();
  if (type_class != H5T_FLOAT)
    std::cerr << "NOT A DOUBLE" << std::endl;

  DataSpace dataspace=dataset.getSpace();

  int rank=dataspace.getSimpleExtentNdims();

  hsize_t* setSize = new hsize_t[rank];//datasetDimension
  hsize_t* readSize = new hsize_t[rank];//chunk to be read size

  dataspace.getSimpleExtentDims(setSize,NULL);

  std::cout << setSize[0] << ',' << setSize[1] << std::endl;

  double* forReading=new double[setSize[0]*setSize[1]]; //MUST BE 1D
  
  dataset.read(forReading,PredType::NATIVE_DOUBLE);

  for (int row = 0; row < setSize[0]; row++) {
    for (int col = 0; col < setSize[1]; col++)
      std::cout << forReading[row*setSize[1]+col] << ' ';
  }
  delete [] forReading;
}
