from h5py import File
from natural_language_processing.configurations.configuration_infrastructure import Config
from natural_language_processing.configurations.configurations import CFG

config = Config.from_json(CFG)


def get_internal_hdf():
    try:
        hdf_in = File(config.data.HDFS_INTERNAL_DATA_FILENAME, "r")
    except IOError:
        print("The internal file failed to open for read.")
    else:
        x_trnset = hdf_in.get("/dataset/train/x_train/x_trainset")
        x_valset = hdf_in.get("/dataset/train/x_train/x_valset")
        y_trnset = hdf_in.get("/dataset/train/y_train/y_trainset")
        y_valset = hdf_in.get("/dataset/train/y_train/y_valset")
        return x_trnset, x_valset, y_trnset, y_valset, hdf_in


def get_external_hdf():
    try:
        hdf_ext = File(config.external_data_sources.HDFS_EXTERNAL_DATA_FILENAME, "r")
        return hdf_ext.get("external_dataset"), hdf_ext
    except IOError:
        print("The external file failed to open for read.")