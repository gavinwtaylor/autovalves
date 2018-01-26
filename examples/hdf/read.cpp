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
  std::cout << rank << std::endl;
}
