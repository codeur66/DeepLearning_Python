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
        x_trnset = hdf_in.get("/dataset/train/x_trainset")
        y_trnset = hdf_in.get("/dataset/train/y_trainset")
        x_valset = hdf_in.get("/dataset/val/x_valset")
        y_valset = hdf_in.get("/dataset/val/y_valset")
        x_testset = hdf_in.get("/dataset/test/x_testset")
        y_testset = hdf_in.get("/dataset/test/y_testset")
        return x_trnset, y_trnset, x_valset, y_valset, x_testset, y_testset, hdf_in


def get_external_hdf():
    try:
        hdf_ext = File(config.external_data_sources.HDFS_EXTERNAL_DATA_FILENAME, "r")
        return hdf_ext.get("external_dataset"), hdf_ext
    except IOError:
        print("The external file failed to open for read.")
