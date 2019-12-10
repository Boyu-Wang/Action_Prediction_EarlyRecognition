
This repository contains code for "Predicting Body Movement and Recognizing Actions: an Integrated Framework for Mutual Benefits": [PDF](https://www3.cs.stonybrook.edu/~minhhoai/papers/movePred_FG18.pdf)

The code is written in torch7. 

# Usage:

1. train classification model use ```lstm_classifier_init.lua```
   The classification loss is computed at every time step, and perform weighted average to get final loss
2. train prediction model use ```lstm_deterministic_prediction.lua```
   The prediction model also takes one hot class vector as input.
3. At test time, use ```lstm_classifier_test_prediction_jointly.lua``` to perform joint classification and prediction.



Please consider cite the paper using the following BibTeX entry:

```
@inproceedings{m_Wang-Hoai-FG18, 
  author = {Boyu Wang and Minh Hoai}, 
  title = {Predicting Body Movement and Recognizing Actions: an Integrated Framework for Mutual Benefits}, 
  year = {2018}, 
  booktitle = {Proceedings of the International Conference on Automatic Face and Gesture Recognition}, 
} 
```