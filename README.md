# Isolation Forests

This is my implementation of the Isolation Forest anomaly detection algorithm from 
[a paper](https://www.researchgate.net/publication/224384174_Isolation_Forest) by Fei Tony Liu, Kai Ming Ting, 
and Zhi-Hua Zhou. The code is adapted from a project I did for a course in Algorithms and Data Structures taught by [Terence Parr](https://github.com/parrt) as part of the University of San Francisco Master of Science in Data Science program.

## Improving on the original algorithm for efficiency

The original algorithm is pretty slow in Python. Instead of randomly splitting the data as described by Liu, Ting, and Zhou, I randomly partitioned the data 5 times and then chose the partition that created the biggest difference in size between the two parts. This improved the efficiency of the algorithm by causing anomalies to be isolated sooner.

## Data

I used a breast cancer dataset in Scikit-learn and [this](https://www.kaggle.com/mlg-ulb/creditcardfraud) credit card fraud dataset from Kaggle.
