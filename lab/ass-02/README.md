# Assignment 2

## Logistic Regression

Logistic Regression is used when the dependent variable(target) is categorical.

M -> Malignant
B -> Benign

### `solver`

Algorithm to use in the optimization problem.

- For small datasets, `liblinear` is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones;

- For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss;

- ‘liblinear’ is limited to one-versus-rest schemes.

### Penalty

Penalty parameter is a form of regularisation.

**Regularization** is a technique used for tuning the function by adding an additional penalty term in the error function

Penalized logistic regression imposes a penalty to the logistic model for having too many variables. This results in shrinking the coefficients of the less contributive variables toward zero. This is also known as regularization.
