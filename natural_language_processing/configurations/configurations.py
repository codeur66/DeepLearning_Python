CFG = {
    "train": {
        "path_train_data": '/home/nikoscf/PycharmProjects/DeepLearningWithPython/natural_language_processing/data/aclImdb/aclImdb'
        , "maxlen": 100
        , "training_samples":200
        , "validation_samples":10000
        ,  "max_words": 10000

    }
}


""" Choice of training_samples":200 
Pretrained word embeddings are meant to
be particularly useful on problems where little training data is available (otherwise,
task-specific embeddings are likely to outperform them), so we restrict the learning
to only 200 unique records."""