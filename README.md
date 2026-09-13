# Stroke prediction

Comparing multiple ML models on the problem of stroke prediction

### Dataset

[prosperchuks - Diabetes, Hypertension and Stroke Prediction](https://www.kaggle.com/datasets/prosperchuks/health-dataset/data)

```bash
kaggle datasets download prosperchuks/health-dataset -p <path_to_this_folder>/data
```

### Reference work

[prasadshingare - Diabetes, Hypertension and Stroke Prediction ](https://www.kaggle.com/code/prasadshingare/diabetes-hypertension-and-stroke-prediction)

### Project structure

* Data preprocessing - [preprocessing.ipynb](preprocessing.ipynb)
* Reference project results recreation - [reference_recreation.ipynb](reference_recreation.ipynb)
* Model creation and results - [main_model.ipynb](main_model.ipynb)
* Package requirements - [requirements.txt](requirements.txt)

## Theoretical basis

## Dataset analysis

The dataset consists of 11 collumns and 4910 rows, with 9 predictors and one target vector.
Sex, Hypertension, Heart disease, Ever married, Residence type, Smoking status, Stroke - Binary 0/1 values
BMI, Average glucose level - Numerical values
Work type - Categorical 0-4 values

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th></th>
      <th>sum</th>
      <th>unique</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sex</th>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0</td>
      <td>111</td>
    </tr>
    <tr>
      <th>hypertension</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>heart_disease</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ever_married</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>work_type</th>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Residence_type</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>avg_glucose_level</th>
      <td>0</td>
      <td>2903</td>
    </tr>
    <tr>
      <th>bmi</th>
      <td>0</td>
      <td>370</td>
    </tr>
    <tr>
      <th>smoking_status</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>stroke</th>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>

The target Stroke vector already had a mostly equal distribution, so no further processing was done.
1  -  20460
0   - 20450

![1789231484293](image/README/1789231484293.png)\

Correlation matrix showed no collumns with over a 0.3 correlation outside of the diagonal, and 12 collumns with an absolute correlation over 0.2.
![1789231612676](image/README/1789231612676.png)

## Reference results reproduction

### Data preprocessing

The reference project removed all predictor collumns with an absolute correlation with the stroke collumn under 0.2. Which kept the predictors as: Heart disease, Hypertension and Average glucose level.

![1789231886950](image/README/1789231886950.png)

Dataset mean and standard deviation:
![1789231926474](image/README/1789231926474.png)![1789231937500](image/README/1789231937500.png)

Histograms of predictors with target values shown:
![1789232000421](image/README/1789232000421.png)

The dataset was split into a train and test set, as in the reference project, in a stratified 80-20 split:
	Total dataset size: 40910

	Training set size: 32728 (80.00%)

	Test set size: 8182 (20.00%)

The data was standardized using a standard scaler to make the model better at predicting and not skew the results.

![1789232194225](image/README/1789232194225.png)

### Reference model creation and comparison

#### Logistic regression

A grid search was done with a stratified 10 fold, standardization and maximum iteration of 5000. The scoring was done with ROC-AUC:

| Parameter / Metric                  | Configuration / Search Space | Best Result           |
| :---------------------------------- | :--------------------------- | :-------------------- |
| **`log_reg__C`**            | `np.logspace(-3, 3, 15)`   | `7.196856730011514` |
| **`log_reg__solver`**       | `['lbfgs', 'liblinear']`   | `'liblinear'`       |
| **`log_reg__penalty`**      | `['l1', 'l2']`             | `'l2'`              |
| **`log_reg__class_weight`** | `[None, 'balanced']`       | `None`              |
| Best CV ROC-AUC                     | —                           | **`0.704`**   |

Results of the model with optimal hyperparameters:

* Test ROC-AUC: 0.698
* Test accuracy: 0.676

Reference results:

- Accuracy: 0.682
- Fl score: 0.681

![1789232753305](image/README/1789232753305.png)

#### Decision tree

Results of the decision tree model:

* Test ROC-AUC: 0.999
* Test accuracy: 0.997

Reference results:

- Accuracy: 0.999
- Fl score: 0.999

![1789232741776](image/README/1789232741776.png)

#### Random forrest

Results of the random forrest model:

* Test ROC-AUC: 0.999
* Test accuracy: 0.997

Reference results:

- Accuracy: 0.997
- Fl score: 0.997

![1789232731642](image/README/1789232731642.png)

#### KNN

A grid search was done with a stratified 5 fold. The scoring was done with ROC-AUC:

| Parameter / Metric        | Configuration / Search Space        | Best Result     |
| :------------------------ | :---------------------------------- | :-------------- |
| **`n_neighbors`** | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` | `7`           |
| **`weights`**     | `['uniform', 'distance']`         | `'distance'`  |
| **`metric`**      | `['euclidean', 'manhattan']`      | `'euclidean'` |

Results of the KNN model with optimal hyperparameters:

* Test ROC-AUC: 0.998
* Test accuracy: 0.997

Reference results:

- Accuracy: 0.851
- Fl score: 0.849

![1789232714559](image/README/1789232714559.png)

### Discussion and result analysis

#### Results comparison

The results have similar values for all models except the K-nearest neighbors models. The slight differences in other model might come from the standardization of data when data preprocessing, or a slight difference in hyperparameters of the model.
The biggest difference occurs in the results of the KNN model, where the reference paper get a test accuracy of 0.85. The model from this paper manages to get the same results as both the Random forrest and the Decision tree model, and an accuracy of almost 0.997 on the test set. These differences are expected to be the result of a grid search for optimal parameters being done in this model creation, and being absent from the reference work.

#### Duplicates analysis

Three of the four models managed to get near perfect results, which leads to an analysis of wether the model is really predicting well or there is an issue in the dataset, preprocessing ot train-test splitting.
A possible contender for the reason of the perfect predictions is a test and train set sharing many duplicate predictor value combinations. When doing an analysis this was seen to be the case:

* Number of rows in test: 8182
* Number of test rows duplicated in train: 8177
* Percentage of test set: 99.94%

After the preprocessing was done, and collumns with low correlation values were removed from the data, the refernce work is left with a train set having 99.94% of rows that are already present in the training set. When removing the duplicate rows, the data is left with as little as 3033 rows, and a completely skewed target vector. This shows that this preprocessing is not adequate for working with this dataset, since removing duplicates ends up with a non-trainable dataset.

* Shape of the dataframe before removing duplicates: (40910, 4)
* Shape of the dataframe after removing duplicates: (3033, 4)

![1789233072162](image/README/1789233072162.png)

#### Analyzing the 23 outliers

The three models with the near-perfect predictions were all incorrectly predicting 23 rows with a stroke value of 1. An analysis is done to make sense of the bad predictions.
Inability to predict some values, may come as a result of the duplicates in the test and train set. If two rows would have the same predictor values but a different target value it would make it impossible for the model to correctly learn to predict these values. The results showed there was sunch an issue in the data with 6,14% of rows being involved in conflicts:

* Number of distinct feature-sets with conflicting stroke labels: 21
* Number of rows involved in a conflict: 2512
* Percentage of dataset: 6.14%

The skewed values seemed to acctually come mostly from the stroke values with a value of 1:

Stroke value counts among conflicting rows:
1    2378
0     134

All 23 incorrectly predicted values were seen to be a result of conflicting target values, as expected:

* Number of badly classified rows: 23
* Number of badly classified rows duplicated in train: 23
* Percentage of badly classified set: 100.00%

The histograms show that the incorrectly classified rows usually had no heart disease, no hypertension and a low average glucose value, which is expected to make it harder to predict since all predictors are shown to drive the target value when higher:

![1789233189873](image/README/1789233189873.png)

## New preprocessing and model creation

### Data preprocessing

To fix the issues of the reference data preprocessing, no predictors were left out.  Because of this the data contained no fully duplicate rows:

Shape of the dataframe before removing duplicates: (40910, 11)
Shape of the dataframe after removing duplicates: (40910, 11)

The three rows with a missing value for the sex collumn were removed. The histograms showed values were present for age that were below 0, which also needed to be removed:

![1789290425221](image/README/1789290425221.png)

Number of rows with negative age removed: 58
Number of rows left in the dataframe: 40849

A pair-plot was created but no obvious patterns for the stroke value were determined:

![1789290500302](image/README/1789290500302.png)

After the analysis the data was again split in an 80-20 split into a train and test set, with stratification enabled. The predictors were then standardized using a standard scaler, and the work type collumn was processed as a categorical collumn with values from 0 to 4:

Processed training shape: (32681, 14)
Processed test shape: (8171, 14)

#### Logistic regression

A grid search was done with a stratified 10 fold, standardization and maximum iteration of 5000. The scoring was done with ROC-AUC:

| Parameter / Metric                      | Configuration / Search Space | Best Result            |
| :-------------------------------------- | :--------------------------- | :--------------------- |
| **`log_reg__C`**                | `np.logspace(-3, 3, 15)`   | `0.3727593720314938` |
| **`log_reg__solver`**           | `['lbfgs', 'liblinear']`   | `'lbfgs'`            |
| **`log_reg__penalty`**          | `['l1', 'l2']`             | `'l2'`               |
| **`log_reg__class_weight`**     | `[None, 'balanced']`       | `'balanced'`         |
| **Best Cross-Validation ROC-AUC** | /                            | **`0.750`**    |

Results of the model with optimal hyperparameters:

* Test ROC-AUC: 0.765
* Test accuracy: 0.701

Earlier results:

* Test ROC-AUC: 0.698
* Test accuracy: 0.676

![1789291356216](image/README/1789291356216.png)

#### Decision tree

Results of the model with optimal hyperparameters:

* Test ROC-AUC: 1.0
* Test accuracy: 1.0

Earlier results:

* Test ROC-AUC: 0.999
* Test accuracy: 0.997

![1789291491227](image/README/1789291491227.png)

#### Random Forrest

Results of the model with optimal hyperparameters:

* Test ROC-AUC: 0.999
* Test accuracy: 0.998

Earlier results:

* Test ROC-AUC: 0.999
* Test accuracy: 0.997

![1789291566701](image/README/1789291566701.png)

#### KNN

A grid search was done with a stratified 10 fold and standardization. The scoring was done with ROC-AUC:

| Parameter / Metric                      | Configuration / Search Space             | Best Result         |
| :-------------------------------------- | :--------------------------------------- | :------------------ |
| **`model__n_neighbors`**        | `[3, 5, 7, 9, 11, 13, 15, 17, 19, 21]` | `11`              |
| **`model__weights`**            | `['uniform', 'distance']`              | `'distance'`      |
| **`model__metric`**             | `['euclidean', 'manhattan']`           | `'manhattan'`     |
| **Best Cross-Validation ROC-AUC** | /                                        | **`0.980`** |

Results of the model with optimal hyperparameters:

* Test ROC-AUC: 0.983
* Test accuracy: 0.898

Earlier results:

* Test ROC-AUC: 0.998
* Test accuracy: 0.997

![1789291748560](image/README/1789291748560.png)

#### Naive Bayes

A grid search was done with a stratified 10 fold and standardization. The scoring was done with ROC-AUC:

| Parameter / Metric                      | Configuration / Search Space | Best Result            |
| :-------------------------------------- | :--------------------------- | :--------------------- |
| **`model__var_smoothing`**      | `np.logspace(-12, -6, 13)` | `1e-06`              |
| **Best Cross-Validation ROC-AUC** | /                            | **`0.747311`** |

Results of the model with optimal hyperparameters:

* Test ROC-AUC: 0.762
* Test accuracy: 0.684

![1789291875126](image/README/1789291875126.png)

#### SVM

A grid search was done with a stratified 10 fold and standardization. The scoring was done with ROC-AUC:

| Parameter / Metric                      | Configuration / Search Space       | Best Result            |
| :-------------------------------------- | :--------------------------------- | :--------------------- |
| **`model__C`**                  | `[0.1, 1, 10, 100]`              | `100`                |
| **`model__gamma`**              | `['scale', 'auto', 0.001, 0.01]` | `'scale'`            |
| **`model__kernel`**             | `['rbf', 'linear']`              | `'rbf'`              |
| **`model__class_weight`**       | `[None, 'balanced']`             | `None`               |
| **Best Cross-Validation ROC-AUC** | /                                  | **`0.910350`** |

Results of the model with optimal hyperparameters:

* Test ROC-AUC: 0.912
* Test accuracy: 0.845

#### XGBoost

A grid search was done with a stratified 10 fold and standardization. The scoring was done with ROC-AUC:

| Parameter / Metric                      | Configuration / Search Space                         | Best Result            |
| :-------------------------------------- | :--------------------------------------------------- | :--------------------- |
| **`model__n_estimators`**       | `[100, 200, 400]`                                  | `400`                |
| **`model__max_depth`**          | `[3, 4, 6]`                                        | `6`                  |
| **`model__learning_rate`**      | `[0.01, 0.05, 0.1]`                                | `0.1`                |
| **`model__subsample`**          | `[0.8, 1.0]`                                       | `0.8`                |
| **`model__scale_pos_weight`**   | `[1, (y_train == 0).sum() / (y_train == 1).sum()]` | `1.0023895594632681` |
| **Best Cross-Validation ROC-AUC** | /                                                    | **`0.999999`** |

Results of the model with optimal hyperparameters:

* Test ROC-AUC: 1.0
* Test accuracy: 0.999

![1789292007991](image/README/1789292007991.png)

#### AdaBoost

A grid search was done with a stratified 10 fold and standardization. The scoring was done with ROC-AUC:

| Parameter / Metric                      | Configuration / Search Space         | Best Result            |
| :-------------------------------------- | :----------------------------------- | :--------------------- |
| **`model__n_estimators`**       | `[50, 100, 200, 400, 600]`         | `600`                |
| **`model__learning_rate`**      | `[0.01, 0.05, 0.1, 0.5, 0.7, 1.0]` | `1.0`                |
| **Best Cross-Validation ROC-AUC** | /                                    | **`0.781603`** |

Results of the model with optimal hyperparameters:

* Test ROC-AUC: 0.794
* Test accuracy: 0.708

![1789292044284](image/README/1789292044284.png)

### Discussion and result analysis

#### Comparison of different models and earlier results

| Model               | Earlier Test ROC-AUC | Earlier Test Accuracy | New Test ROC-AUC | New Test Accuracy |
| ------------------- | -------------------: | --------------------: | ---------------: | ----------------: |
| Logistic Regression |                0.698 |                 0.676 |            0.765 |             0.701 |
| Decision Tree       |                0.999 |                 0.997 |            1.000 |             1.000 |
| Random Forest       |                0.999 |                 0.997 |            0.999 |             0.998 |
| KNN                 |                0.998 |                 0.997 |            0.983 |             0.898 |
| Naive Bayes         |                    / |                     / |            0.762 |             0.684 |
| SVM                 |                    / |                     / |            0.912 |             0.845 |
| XGBoost             |                    / |                     / |            1.000 |             0.999 |
| AdaBoost            |                    / |                     / |            0.794 |             0.708 |

#### Future work
