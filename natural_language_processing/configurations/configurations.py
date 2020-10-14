CFG = {
    "external_data_sources": {
        "word_embeddings": "/home/nikoscf/PycharmProjects/DeepLearningWithPython/natural_language_processing/data/glove.6B"
        , "embeddings_file_name": "glove.6B.100d.txt"
        , "embeddings_dimension": 100
    },
    "data": {
        "path_train_data": '/home/nikoscf/PycharmProjects/DeepLearningWithPython/natural_language_processing/data/aclImdb/aclImdb'
        , "maxlen": 100
        , "training_samples": 200
        , "validation_samples": 10000
        ,  "max_words": 10000
    },
    "train": {

    },
    "model":{

    }
}


""" Choice of training_samples":200 
Pretrained word embeddings are meant to
be particularly useful on problems where little training data is available (otherwise,
task-specific embeddings are likely to outperform them), so we restrict the learning
to only 200 unique records."""