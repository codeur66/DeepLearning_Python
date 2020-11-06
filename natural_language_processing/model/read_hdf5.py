from h5py import File
INTERNAL_FILE_HDF5 = "../data/internal_dataset.h5py"
EXTERNAL_FILE_HDF5 = "../data/external_dataset.h5py"


def get_internal_hdf():
    hdf_in = File(INTERNAL_FILE_HDF5, "r")
    x_trnset = hdf_in.get("/dataset/train/x_train/x_trainset")
    x_valset = hdf_in.get("/dataset/train/x_train/x_valset")
    y_trnset = hdf_in.get("/dataset/train/y_train/y_trainset")
    y_valset = hdf_in.get("/dataset/train/y_train/y_valset")
    return x_trnset, x_valset, y_trnset, y_valset, hdf_in


def get_external_hdf():
    hdf_ext =File(EXTERNAL_FILE_HDF5, "r")
    return hdf_ext.get("hdf_embedding_matrix"), hdf_ext


# import numpy as np
# n1 = np.zeros(shape=x_trn_set.shape)
# print(x_trn_set.read_direct(np))