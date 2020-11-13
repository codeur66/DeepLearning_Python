import unittest
import natural_language_processing.model.read_hdf5 as rd


class MyTestCase(unittest.TestCase):
    print("Tests if every dataset is not empty.")
    x_trnset, y_trnset, x_valset, y_valset, hdf_in = rd.get_internal_hdf()
    embeddings, hdf_ext = rd.get_external_hdf()

    def test_noneType_internal(self):
        if type(MyTestCase.x_trnset) is type(None) or \
                type(MyTestCase.y_trnset) is type(None) or \
                type(MyTestCase.x_valset) is type(None) or \
                type(MyTestCase.y_valset) is type(None):
            raise Exception("NoneType in internal dataset is not permitted.Every test in datasets stops.")

    def test_noneType_external(self):
        if type(MyTestCase.embeddings) is type(None):
            raise Exception("NoneType in external dataset is not permitted.Every test in datasets stops.")

    def test_datatype_get_internal_hdf(self):
        self.assertEqual(type(MyTestCase.x_trnset[:]).__name__, 'ndarray')
        self.assertEqual(type(MyTestCase.y_trnset[:]).__name__, 'ndarray')
        self.assertEqual(type(MyTestCase.x_valset[:]).__name__, 'ndarray')
        self.assertEqual(type(MyTestCase.y_valset[:]).__name__, 'ndarray')
        self.assertEqual(str(MyTestCase.hdf_in), '<HDF5 file "internal_dataset.h5py" (mode r)>')

    def test_datatype_get_external_hdf(self):
        self.assertEqual(type(MyTestCase.embeddings[:]).__name__, 'ndarray')
        self.assertEqual(str(MyTestCase.hdf_ext), '<HDF5 file "external_dataset.h5py" (mode r)>', "HEY")


if __name__ == '__main__':
    unittest.main()
