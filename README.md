# 6.806-project
Final Project for 6.806 NLP - Question Retrieval, Transfer Learning
by Janice Chui and Katherine Young

Android dataset: https://github.com/jiangfeng1124/Android
AskUbuntu dataset: https://github.com/taolei87/askubuntu

Code Structure
da.py: Implementation and models for Domain Adaptation
direct_transfer.py: Direct transfer using models trained on AskUbuntu and evaluated on Android dataset
evaluation.py: Evaluation code taken from Tao Lei's rcnn (https://github.com/taolei87/rcnn/blob/master/code/qa/evaluation.py)
inout.py: Data pre-processing
meter.py: AUC code taken from PyTorchNet (https://github.com/pytorch/tnt)
part1.py: Implementation and models for Question Retrieval
tfidf.py: TF-IDF evaluation for unsupervised methods

Please run each file (da.py, direct_transfer.py, part1.py) directly. The best hyperparameters are included in the models. 
