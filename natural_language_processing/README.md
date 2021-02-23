
# Supervised NLP on movies reviews


* Reviews are as text and are either positive or negative
  Through text processing the NN will learn to do itself the binary
  supervised classification
  

## Data Pipeline
_________________________________________________________________

* Parameters for the Data pipeline in one file using 
  object serialization.
* The processing of texts from the movies' reviews named
"internal dataset"
* Use of pretrained world embeddings vocabulary, named "external".
* Use of HDFS
* Precise logging

## Model
__________________________________________________________________

* Parameters for the Model in one file using 
  object serialization.
* Build in Keras using OOP
* Read HDFS files
* Trainning giving the choice of using of not the word embeddings
* Plot, Evaluation
* Precise logging

## DevOps
__________________________________________________________________

* Jenkins CI/CD installed and configured, connected to Github,
  with job named DeepLearning_job_,
* Jenkins Pipeline_job for the build and tests.
* Monitoring for Test Coverage

##Future Goals
________________________________________________________________
* Mock Tests
* MLFlow