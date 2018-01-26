#include <hdf5.h>

int main() {
  /* create the file */
  file_id = H5Fcreate("myfile.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  /* create attribute0 */
  space_id = H5Screate(H5S_SCALAR);
  attr_id = H5Acreate(file_id, "attribute0", H5T_NATIVE_INT32, space_id, H5P_DEFAULT);
  H5Awrite(attr_id, H5T_NATIVE_INT32, 42);
  H5Aclose(attr_id);
  H5Sclose(space_id);

  /* create dataset0 */
  space_id = H5Screate_simple(rank, dims, maxdims);
  dset_id = H5Dcreate(file_id, "dataset0", H5T_NATIVE_FLOAT, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, somedata0);
  H5Dclose(dset_id);
  H5Sclose(space_id);

  /* create group0 */
  group_id = H5Gcreate(file_id, "/group0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  /* and dataset1 */
  space_id = H5Screate_simple(rank, dims, maxdims);
  dset_id = H5Dcreate(group_id, "dataset1", H5T_NATIVE_FLOAT, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, somedata1);
  H5Dclose(dset_id);
  H5Sclose(space_id);
  H5Gclose(group_id);

  /* finished! */
  H5Fclose(file_id);

  return 0;
}
