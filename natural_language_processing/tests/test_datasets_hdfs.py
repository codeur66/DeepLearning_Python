import unittest
import natural_language_processing.model.read_hdf5 as rd


class TestDatasets(unittest.TestCase):
    print("Tests if every dataset type is not empty and is ndarray.")

    def setUp(self) -> None:
        self.x_trnset, self.y_trnset, self.x_valset, self.y_valset, self.hdf_in = rd.get_internal_hdf()
        self.embeddings, self.hdf_ext = rd.get_external_hdf()

    def test_noneType_internal(self):
        if type(self.x_trnset) is type(None) or \
                type(self.y_trnset) is type(None) or \
                type(self.x_valset) is type(None) or \
                type(self.y_valset) is type(None):
            raise Exception("NoneType in internal dataset is not permitted.Every test in datasets stops.")

    def test_noneType_external(self):
        if type(self.embeddings) is type(None):
            raise Exception("NoneType in external dataset is not permitted.Every test in datasets stops.")

    def test_datatype_get_internal_hdf(self):
        self.assertEqual(type(self.x_trnset[:]).__name__, 'ndarray')
        self.assertEqual(type(self.y_trnset[:]).__name__, 'ndarray')
        self.assertEqual(type(self.x_valset[:]).__name__, 'ndarray')
        self.assertEqual(type(self.y_valset[:]).__name__, 'ndarray')
        self.assertEqual(str(self.hdf_in), '<HDF5 file "internal_dataset.h5py" (mode r)>')

    def test_datatype_get_external_hdf(self):
        self.assertEqual(type(self.embeddings[:]).__name__, 'ndarray')
        self.assertEqual(str(self.hdf_ext), '<HDF5 file "external_dataset.h5py" (mode r)>', "HEY")


    def tearDown(self) -> None:
        self.hdf_in.close()
        self.hdf_ext.close()

if __name__ == '__main__':
    unittest.main()
