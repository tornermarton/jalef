# JALEF - Just Another Language Engineering Framework

This deep learning framework is created for my summer internship at Avaya. The repo includes also the files I used
to solve my task which was intent classification with neural networks.

Student: Torner Márton \
Supervisor: Szaszák György, PhD \
Supervisor: Gyires-Tóth Bálint, PhD

- [JALEF - Just Another Language Engineering Framework](#jalef-just-another-language-engineering-framework)
  - [Token](#token)
  - [Docker](#docker)
  - [GPU](#gpu)
  - [Task](#task)
  - [Data](#data)
    - [Coursera](#coursera)
    - [Avaya](#avaya)
  - [Framework](#framework)
    - [Layers](#layers)
    - [Models](#models)
    - [Preprocessing](#preprocessing)
    - [Other files](#other-files)
  - [My work](#my-work)
    - [Notebooks](#notebooks)
    - [Neural Network](#neural-network)
      - [Word embeddings](#word-embeddings)
        - [Word2Vec](#word2-vec)
        - [BERT (Bidirectional Encoder Representations from Transformers)](#bert-bidirectional-encoder-representations-from-transformers)
      - [Classification](#classification)
    - [Logs](#logs)
    - [Weights](#weights)
    - [Results](#results)
    - [Demo](#demo)


## Token

The notebooks also support using them in colab but since this repository is private it can only be cloned with a token. 
For security reasons please do not send this token to anyone and clear the output field of the notebook cell where you 
fill it in before giving the notebook to someone else.

```
ba7aebdb4a34e89bd1906f1f845e1025787df7f1
```

## Docker

The instructions and infos about the docker container can be found in [./docker/DOCKER.md](./docker/DOCKER.md)

## GPU

For my experiments I used an `NVIDIA GeForce GTX TITAN X 12GB` GPU.

## Task

The task was to test a conventional and a new approach to classify texts based on their content a.k.a. 
intent classification. The 2 chosen word embeddings were Word2Vec which is a static model (it gives the same vector 
for the same word every time) and BERT which is context based (the word embedding vector can vary based on the context 
it appears). For further information please see the Word embeddings section where I provided some useful resources.

## Data

### Coursera

Coursera raw lecture text data.

Server: `flower.tmit.bme.hu`

Port: `6504`

Download: 

```bash
scp -r -P 6504 [USER]@flower.tmit.bme.hu:/home/marton/lecture_text_data.zip ./coursera.zip
```

The repo also includes this zip file under the data folder.

### Avaya

Later.

## Framework

I created this framework to help myself organize the tools for my work and also included some models just to practice.
It is structured in a way that it can be installed as a python package if you wish.

### Layers

The implementation of layers used by the models which are not part of the tf.keras library.

### Models

The networks I implemented based on the articles I read. In engine.py you can find the core classes which can help to 
implement further models without writing the basic things all over again.

### Preprocessing

The preprocessors I implemented to create the network inputs from sequences of words for each word embedding I tried.

### Other files

Here you can find other tools/functions I used in the notebooks for visualization or evaluation.

## My work

### Notebooks

This project involved some Jupyter Notebooks to visualize the data and provide some examples on using the framework. 
Some of them also contains notes from me to give a better understanding what I did and why.

### Neural Network

#### Word embeddings

In this project I used and compared 

##### Word2Vec

Tutorial on word2vec by TensorFlow: [TF tutorial](https://www.tensorflow.org/tutorials/representation/word2vec)

I used pretrained word vectors by Google. 

[Download link](https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz)

It includes word vectors for a vocabulary of 3 million words and phrases that they trained on roughly 100 billion words 
from a Google News dataset. The vector length is 300 features.

##### BERT (Bidirectional Encoder Representations from Transformers)

Paper: [BERT](https://arxiv.org/pdf/1810.04805.pdf)

Useful article: [BERT Explained](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)

Repo (with links to pretrained models): [Google BERT repo](https://github.com/google-research/bert)

Keras implementation used as a base for my BERT layer: [BERT in keras](https://towardsdatascience.com/bert-in-keras-with-tensorflow-hub-76bcbc9417b)

TensorflowHub (downloadable pretrained models - search for bert): [TF Hub](https://tfhub.dev)

#### Classification

For the classification task I used a simple network on top of the embedding layers with Bidirectional LSTM followed by 
two fully-connected layers and an output layer with softmax activation.

### Logs

To see the training logs please use tensorboard (root directory is `logs/tensorboard`). 
For further instructions see docker.

### Weights

Before the following sections I have to mention that unfortunately the network weights are too big to upload to this 
repo so you have to run the training to get them.

### Results

The results (predictions for the test set) of my experiments are saved under the data folder and visualized in the 
evaluation_[dataset] notebook files.

### Demo

For the coursera dataset I also created a small demo where the pretrained model can be loaded and can be used to 
classify a random out-of-sample sentence/text.