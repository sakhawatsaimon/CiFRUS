# Class-invariant Feature Range Uniform Sampling (CiFRUS)

## Introduction

This is the official implementation of the tabular data augmentation method described in the following KDD 2024 paper: [A Novel Feature Space Augmentation Method to Improve Classification Performance and Evaluation Reliability](https://doi.org/10.1145/3637528.3671736). The core augmentation method is class-invariant and supports majority-based prediction of unlabeled instances. Each unlabeled instance can be expanded into a set of augmented instances, followed by classifier prediction and aggregation of the predicted labels or class probabilities.

## Installation

### Installing with Pip

`CiFRUS` can be installed from [PyPI](https://pypi.org/project/cifrus/) with `pip`:

```
python -m pip install cifrus
```

The PyPI package only contains the core augmentation module. To acquire the datasets and code for conducting experiments, please fork this repository.

### From source
`cifrus.py` can be directly copied to the project repository.

## Usage

`CiFRUS` is compatible with [scikit-learn](https://scikit-learn.org/stable/) and [imbalanced-learn](https://imbalanced-learn.org/stable/). Given feature matrix `X` where each row is a sample and corresponding class labels `y`, augmentation can be performed as follows:

```
from cifrus.cifrus import CiFRUS
cfrs = CiFRUS()
X_resampled, y_resampled = cfrs.fit_resample(X, y)
```

By default, `CiFRUS` augmentation increases the number of samples in the majority class five-fold, and balances the minority classes.

The majority-based prediction of `CiFRUS` can be used by passing the classifier's `predict_proba()` function as a parameter to the `resample_predict_proba` function of the `CiFRUS` object, as demonstrated below:

```
# any sklearn compatible classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier() 
clf.fit(X_resampled, y_resampled)

# X_test is resampled, and the probabilities are aggregated internally
y_pred = cfrs.resample_predict_proba(clf.predict_proba, X_test)
```

## License

This work is licensed under the [Creative Commons Attribution 4.0 International License](https://github.com/sakhawathossain/CiFRUS/blob/main/LICENSE).

## Citation

If you use `CiFRUS` in your scientific research, please consider citing us:

```
@inproceedings{saimon2024novel,
    author = {Saimon, Sakhawat Hossain and Najnin, Tanzira and Ruan, Jianhua},
    title = {A Novel Feature Space Augmentation Method to Improve Classification Performance and Evaluation Reliability},
    doi = {10.1145/3637528.3671736},
    booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    pages = {2512–2523},
    year = {2024},
}
```