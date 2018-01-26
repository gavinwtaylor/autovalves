#include <iostream>
#include <string>
#include "H5Cpp.h"
using namespace H5;

int main() {

  double data[5][3];
  double val=0;
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 3; j++) {
      data[i][j]=val;
      val+=.1;
    }
  H5File file("test.h5",H5F_ACC_TRUNC);//TRUNC means overwrite existing
                                       //EXCL fails if exists already
                                       //RDONLY read only
                                       //RDWR read/write if already exists
  hsize_t dimensions[2];
  dimensions[0]=5;
  dimensions[1]=3;
  DataSpace dataspace(2,dimensions); //2D, 5x3
  DataSet
    dataset=file.createDataSet("testDataset",PredType::NATIVE_DOUBLE,dataspace);
  dataset.write(data,PredType::NATIVE_DOUBLE);
}
