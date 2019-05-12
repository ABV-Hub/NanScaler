# NanScaler

## Introduction
**Use `sklearn.preprocessing` scaler classes with nan values!** It lets you to handle `nan` values in a dataset by *ignoring* them. The reason this class was written is that existing `sklearn.preprocessing` scaler and transformer classes do not work with `nan` values at the moment (which is sad). One does not have to lose data during scaling or impute any possibly misleading data to `nan`'s, especially for *machine learning* purposes. `NanScaler` helps a lot.

Check [sklearn.preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) classes for further information. I actually use them quite a lot and I am a huge fan of complete [scikit learn](https://scikit-learn.org/stable/index.html) library.


## Installation
Clone this repository with following:
```
git clone
```

## Requirements
There are total 3 python packages that this project require:
- `scikit-learn`
- `numpy`
- `pandas`

Install packages with this fashion, (check their websites for further instructions)
```
pip install scikit-learn
```

## Verification
After installation, head to the directory and run:
```
$ python demo.py
```

## Basic Guideline
Create a `NanScaler` object using one of the scikit learn scaler or transformer classes (source is mentioned above). **Use it like any other scikit learn object!**

```
sc = NanScaler(StandardScaler)
arr_scaled = sc.fit_transform(arr)
print(arr_scaled)
```

Check the demo for further information, [demo.py](https://github.com/mcandar/NanScaler/blob/master/demo.py).


## License
[MIT License](https://github.com/mcandar/NanScaler/blob/master/LICENSE)
