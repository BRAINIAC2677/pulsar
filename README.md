# `PULSAR`: Graph based Positive Unlabeled Learning with Multi Stream Adaptive Convolutions for Parkinson’s Disease Recognition


## Overview


![Alt Text](/assets/PULSAR_full_overview.png)


Timely diagnosis of movement disorders like Parkinson’s disease (PD) improves quality of life. However, access to clinical diagnosis is limited in low-income countries. Here, we present PULSAR, a novel method for classifying individuals with or without PD from webcam-recorded videos of the finger-tapping task used in the Movement Disorder Society - Unified Parkinson's Disease Rating Scale (MDS-UPDRS). PULSAR was trained and evaluated on data from 382 participants, including 183 self-reported PD patients. We used an adaptive graph convolutional neural network to dynamically learn task-specific spatio-temporal edges and enhanced it with a multi-stream convolution model to capture critical features like finger joint locations, tapping velocity, and acceleration of tapping. As video labels are self-reported, some non-PD labels may be undiagnosed cases. To address this, we used Positive Unlabeled (PU) Learning, which outperformed traditional supervised learning. PULSAR achieved 80.95% accuracy on the validation set and 71.29% mean accuracy (2.49% standard deviation) on an independent test set. We hope PULSAR can aid in accessible PD screening and that these techniques may extend to assessing disorders like ataxia and Huntington’s disease.

[`Link To Paper`](https://zarif98sjs.github.io/PULSAR/)

## Datasets
- [uspark_finger_tapping_train_val](/datasets/uspark_finger_tapping/train_val_data/)
- [uspark_finger_tapping_test](/datasets/uspark_finger_tapping/test_data/)
- [banglapark_finger_tapping_test](/datasets/banglapark_finger_tapping/test_data/)

## Reproducing Test Scores on BanglaPark/USPark
- Clone the repo
- Create a virtual environment
- Install the requirements from requirements.txt

        pip install -r requirements.txt
- Run test_banglapark.py for BanglaPark test set 

        python test_banglapark.py
- Run test_uspark.py for USPark test set

        python test_uspark.py

- A csv file named `banglapark_test_set_scores.csv` or `uspark_test_set_scores.csv` will be created in the root directory
- As mentioned in paper, for each model variant 120 patients will be sampled from the test set and evaluated 20 times. The mean of the 20 scores will be reported in the csv file.


## Training on USPark
- clone the repo
- create a virtual environment
- install the requirements from requirements.txt

        pip install -r requirements.txt
- run train_uspark.py

        python train_uspark.py

- models will be saved in a newly created trained_models folder
- you can change the hyperparameters from the [config file](/config.py)
