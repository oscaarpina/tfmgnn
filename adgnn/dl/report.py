from sklearn.utils import Bunch
from ..dl.metrics import BinaryMetric, BasicMetric, MultipleMetric

class Report(Bunch):
    def __init__(self, **kwargs):
        super(Report, self).__init__(**kwargs)

    def add(self, m : BinaryMetric, **kwargs):
        if issubclass(m, BasicMetric):
            assert "y_true" in kwargs and "y_score" in kwargs and "y_pred" in kwargs
            setattr(self, m.key(), m.compute(**{**kwargs, **self}) )
        if issubclass(m, MultipleMetric):
            assert "y_true" in kwargs and "y_score" in kwargs and "y_pred" in kwargs
            results = m.compute(**{**kwargs, **self})
            for k,v in results.items():
                setattr(self, k, v)