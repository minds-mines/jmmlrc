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
