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

`pyenv install 3.11.4`
`pip install poetry`

### activate env

```
cd <root of repo>
pyenv local 3.11.4
poetry install
poetry shell
```


################################################################

For accessing data used for the scripts, ive stored them and made it available in AWS Simple Storage.  
Dowload AWS cli from here https://aws.amazon.com/cli/  and then in command line, cd to root of repo and run: 

`aws s3 cp s3://tensorflow-data-certification data --recursive`

