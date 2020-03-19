# serving-nlp

This repository contains the code for the my blog post on [setting up a server for NLP models in production](http://anbasile.github.io/posts/serving-ml-models-production/).

# Instructions

1. `conda env create -f environment.yml`
2. `python train_model.py`
3. `mkdir serving-nlp/model_repository/`
4. `mv sentiment-model/ model_repository/`
5. `cp config.pbtxt model_repository/sentiment-model/`
6. `touch model_repository/sentiment-model/labels.txt`
7. `printf '%s\n%s\n' 'negative' 'positive' >> model_repository/sentiment-model/labels.txt`
4. `docker-compose up`
5. `python query_model.py`
