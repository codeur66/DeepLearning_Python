import unittest
import mock
# jenkins exposes the workspace directory through env.
# with this code we can do import and set workspace.
# to run it outside the jenkins workspace comment it.
# comment the three below lines if you run it outside of Jenkins Server.

# import sys
# import os
# sys.path.append(os.environ['WORKSPACE'])
import natural_language_processing.model.read_hdf5 as rd

class TestParserWordEmbeddings(unittest.TestCase):

    @mock.patch()
    def embeddings_vectors(cls):
        pass




if __name__ == '__main__':
    unittest.main()
