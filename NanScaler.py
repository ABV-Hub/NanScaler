import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NanScaler(BaseEstimator,TransformerMixin):
    """
    Just a CustomScaler to handle nan values since sklearn scaler
    classes cannot at the time.
    
    Parameters
    ----------
    scaler : callable
        Scaler classes from sklearn.preprocessing (e.g. RobustScaler,
        StandardScaler, QuantileTransformer, etc.)
        
    See also
    --------
    https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
    
    Notes
    -----
    Nans are treated as missing values and eliminated in all cases. This class
    does not work on sparse matrices.
    
    Examples
    --------
    >>> from sklearn.preprocessing import RobustScaler
    >>> arr = np.random.randint(0,50,(5,3))
    >>> custom = NanScaler(RobustScaler)
    >>> custom.fit(arr)
    >>> tfmd = custom.transform(arr)
    >>> print(tfmd)
    [[ 1.6        -0.375       0.85714286]
     [ 0.         -0.75       -0.78571429]
     [-0.4         0.625       0.        ]
     [-1.1         0.79166667 -0.14285714]
     [ 0.6         0.          1.14285714]]
    """
    def __init__(self, scaler):
        self.scaler = scaler

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        if hasattr(self, 'scalers'):
            del self.scalers, self.n_samples, self.n_features

    def __fun(self,x,method):
        self.n = self.n + 1
        attr = getattr(self.scalers[self.n],method)
        return attr(x.reshape((-1,1))).flatten()
    
    def __apply(self,X,method):
        output = X.copy() * 1.0
        index = ~pd.isnull(output) # non-null values
        self.n = -1
        output[~index] = 1
        output = np.apply_along_axis(self.__fun,0,output,method)
        output[~index] = np.nan
        return output

    def fit(self, X, y=None):
        """Compute necessary parameters to be used for later scaling.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored
        """
        self._reset()
        self.n_samples  = X.shape[0]
        self.n_features = X.shape[1] if len(X.shape) > 1 else 1
        self.scalers = np.apply_along_axis(lambda x: self.scaler().fit(x[~pd.isnull(x)].reshape((-1,1))),0,X).flatten()
        return self
    
    def transform(self, X):
        """Perform standardization by centering and scaling
        
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
            
        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Transformed array.
        """
        return self.__apply(X,'transform')

    def inverse_transform(self, X):
        """Scale back the data to the original representation
        
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
            
        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Transformed array.
        """
        return self.__apply(X,'inverse_transform')
