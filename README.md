# Google Tensorflow developer certificate

Repo containing material and practice for the Tensorflow developer certificate exam.
I have used Coursera's DeepLearning.ai  Tensorflow in Practice specialisation for preparing for the exam and adapting
the notebooks into .py scripts to work through the skills checklist for the four themes:


* Foundational principles of machine learning (ML) and deep learning (DL)
using TensorFlow
* Convolutional Neural Network in Tensorflow
* Natural Language Processing in TensorFlow
* Sequence, Time Series and Prediction


## For installing requirements


install pyenv from https://github.com/pyenv/pyenv?tab=readme-ov-file#getting-pyenv

```
pyenv install 3.11.4
pip install poetry
```

### activate env

```
cd <root of repo>
pyenv local 3.11.4
poetry install
poetry shell
```

## Download data 

For accessing cats and dogs data used for the scripts, in the root of the repo, download the
zip file from https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip and place in 
the `data` directory in root of repo.

for inception weights download https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 
and place in data folder.
