# Assignment 1

## Reading Dataset

We use the `pandas` library for that. **Pandas is a Python library used for working with data sets.** To read CSV data we use the `read_csv` method which returns a dataframe.

> **Data Science:** is a branch of computer science where we study how to store, use and analyze data for deriving information from it. 

To get the rows and columns of a dataframe we use the `.shape` attribute.

## Cleaning Dataset

To remove the rows that have missing values use the `dropna()` method. Now, if only specific columns need to be considered then use pass the column names in the `subset` param of the `dropna()` method.

## Splitting Dataset

[Read More](https://realpython.com/train-test-split-python-data/)

The dataset we have is split to:

- The **training set** is applied to train, or fit, your model.
- The **test set** is needed for an unbiased evaluation of the final model.

Splitting datasets also helps in detecting:

- **Underfitting** is when the model unable to find relations between the data. This happens when the dataset has less features.
- **Overfitting** usually takes place when a model has an excessively complex structure and learns both the existing relations among data and noise.

To split we use the `train_test_split()` method of the `sklearn` library.

## Regression

[Read More](https://realpython.com/linear-regression-in-python/)

Regression is the search for relationship between various variables.


