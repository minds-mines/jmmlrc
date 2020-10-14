# jmmlrc
Joint Multi-Modal Longitudinal Regression and Classification (JMMLRC) Estimator

This repository contains the code for performing a classification and a regression task at the same time on longitudinal data. This work has been applied to the ADNI dataset and has shown state-of-the-art performance in predicting patients with and without Alzheimer's Disease.

## Getting Started
In order to set up your environment you need to install [pipenv](https://pipenv.readthedocs.io/en/latest)

```bash
cd jmmlrc
pipenv install
pipenv shell
```

Once you have started the pipenv environment you can run all associated tests with the following:

```bash
python setup.py test
```

The code located in `tests/test_estimator.py` illustrates how the `JMMLRC` estimator can be used in practice. 

## Experiment Hyperparameters
In our recent Transactions on Medical Imaging submission we compare our method to the an array of machine learning algorithms with the following settings:

### Regression
```python
linear = LinearRegression(normalize=True, fit_intercept=False)
ridge = Ridge(alpha=1000)
lasso = Lasso(alpha=0.001)
mlp = MLPRegressor(alpha=1, activation='logistic', hidden_layer_sizes=(10,))
elm = ELM(neurons=100, func="rbf_l2")

jmmlrc_l21 = JMMLRC(gamma1 = 100, gamma2 = 0, gamma3 = 0)
jmmlrc_group = JMMLRC(gamma1 = 0, gamma2 = 100, gamma3 = 0)
jmmlrc_trace = JMMLRC(gamma1 = 0, gamma2 = 0, gamma3 = 100)

jmmlrc = JMMLRC(gamma1 = 1e-5, gamma2 = 0.01, gamma3 = 100)
```

### Classification
```python
log = LogisticRegression(C=0.1, penalty='l1')
tree = DecisionTreeClassifier(criterion='entropy')
svc = SVC(C=10, kernel='sigmoid')
knn = KNeighborsClassifier(weights='distance', algorithm='kd_tree', n_neighbors=20, p=1)
mlp = MLPClassifier(alpha=1, hidden_layer_sizes=(10,), activation='logistic')
sgd = SGDClassifier(alpha=0.01, loss='log', penalty='elasticnet', l1_ratio=0.5)
linearsvc = LinearSVC(C=0.001, loss='hinge')
elm = ELM(neurons=1000, func="tanh", classification="c")

jmmlrc_l21 = JMMLRC(gamma1 = 100, gamma2 = 0, gamma3 = 0)
jmmlrc_group = JMMLRC(gamma1 = 0, gamma2 = 1000, gamma3 = 0)
jmmlrc_trace = JMMLRC(gamma1 = 0, gamma2 = 0, gamma3 = 100)

jmmlrc = JMMLRC(gamma1 = 1e-5, gamma2 = .01, gamma3 = 100)
```

## Citation
If you find this code useful in your research, please consider citing:
```
@article{brand2019joint,
  title={Joint Multi-Modal Longitudinal Regression and Classification for Alzheimer’s Disease Prediction},
  author={Brand, Lodewijk and Nichols, Kai and Wang, Hua and Shen, Li and Huang, Heng},
  journal={IEEE Transactions on Medical Imaging},
  volume={39},
  number={6},
  pages={1845--1855},
  year={2019},
  publisher={IEEE}
}
```
