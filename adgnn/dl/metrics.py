from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, precision_recall_curve, accuracy_score, recall_score, precision_score, precision_recall_fscore_support

class BinaryMetric(ABC):
    pass

class BasicMetric(BinaryMetric):
    @classmethod
    @abstractmethod
    def key(cls) -> str:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def compute(cls, **kwargs):
        raise NotImplementedError()


class Precision(BasicMetric):
    @classmethod
    def key(cls):
        return "precision"

    @classmethod
    def compute(cls, **kwargs):
        assert "y_true" in kwargs and "y_pred" in kwargs, "y_true and y_pred needed"
        y_true, y_pred = kwargs["y_true"], kwargs["y_pred"]
        return precision_score(y_true, y_pred)


class Recall(BasicMetric):
    @classmethod
    def key(cls):
        return "recall"
    @classmethod
    def compute(cls, **kwargs):
        assert "y_true" in kwargs and "y_pred" in kwargs, "y_true and y_pred needed"
        y_true, y_pred = kwargs["y_true"], kwargs["y_pred"]
        return recall_score(y_true, y_pred)


class Accuracy(BasicMetric):
    @classmethod
    def key(cls):
        return "accuracy"
    @classmethod
    def compute(cls, **kwargs):
        assert "y_true" in kwargs and "y_pred" in kwargs, "y_true and y_pred needed"
        y_true, y_pred = kwargs["y_true"], kwargs["y_pred"]
        return accuracy_score(y_true, y_pred)

class PRCurve(BasicMetric):
    @classmethod
    def key(cls) -> str:
        return "pr_curve"

    @classmethod
    def compute(cls, **kwargs):
        assert "y_true" in kwargs and "y_score" in kwargs, "y_true and y_score needed"
        y_true, y_score = kwargs["y_true"], kwargs["y_score"]
        p, r, th = precision_recall_curve(y_true, y_score)
        return (p, r, th)

class PRAUC(BasicMetric):
    @classmethod
    def key(cls):
        return "pr_auc"

    @classmethod
    def compute(cls, **kwargs):
        assert "y_true" in kwargs and "y_score" in kwargs, "y_true and y_score needed"
        y_true, y_score = kwargs["y_true"], kwargs["y_score"]
        p, r, th = precision_recall_curve(y_true, y_score)
        return auc(r, p)


class ROCAUC(BasicMetric):
    @classmethod
    def key(cls):
        return "roc_auc"

    @classmethod
    def compute(cls, **kwargs):
        assert "y_true" in kwargs and "y_score" in kwargs, "y_true and y_score needed"
        y_true, y_score = kwargs["y_true"], kwargs["y_score"]
        return roc_auc_score(y_true, y_score)


class ConfusionMatrix(BasicMetric):
    @classmethod
    def key(cls):
        return "cm"

    @classmethod
    def compute(cls, **kwargs):
        assert "y_true" in kwargs and "y_pred" in kwargs, "y_true and y_pred needed"
        y_true, y_pred = kwargs["y_true"], kwargs["y_pred"]
        return confusion_matrix(y_true, y_pred)


class SavingsCost(BasicMetric):
    @classmethod
    def key(cls) -> str:
        return "savings_cost"

    @classmethod
    def compute(cls, **kwargs):
        assert Precision.key() in kwargs and Recall.key() in kwargs, " \"precision\" and \"recall\" needed to compute this metric"
        precision, recall = kwargs[Precision.key()], kwargs[Recall.key()]
        c_pet, c_mri, c_avg, ro = 3000, 700, 1847.1517, 0.2
        return 1 - (1 / (2 * c_avg)) * \
                       (ro * c_pet / precision + c_mri / recall) \
            if precision != 0 and recall != 0 else 0


class SavingsBurden(BasicMetric):

    @classmethod
    def key(cls) -> str:
        return "savings_burden"

    @classmethod
    def compute(cls, **kwargs):
        assert Precision.key() in kwargs, " \"precision\"  needed to compute this metric"
        precision, recall = kwargs[Precision.key()], kwargs[Recall.key()]
        c_pet, c_mri, c_avg, ro = 3000, 700, 1847.1517, 0.2
        return 1 - ro / precision if precision != 0 else 0

class MultipleMetric(BinaryMetric):
    @classmethod
    @abstractmethod
    def compute(cls, **kwargs) -> dict:
        raise NotImplementedError()


class PRFS(MultipleMetric):
    @classmethod
    def compute(cls, **kwargs) -> dict:
        assert "y_true" in kwargs and "y_pred" in kwargs, "y_true and y_pred needed"
        y_true, y_pred = kwargs["y_true"], kwargs["y_pred"]
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average='binary')
        return {Precision.key():p, Recall.key():r, FScore.key():f, "support":s}


class FScore(BasicMetric):
    @classmethod
    def key(cls) -> str:
        return "fscore"

    @classmethod
    def compute(cls, **kwargs):
        assert "precision" in kwargs and "recall" in kwargs, " \"precision\" and \"recall\" needed to compute this metric"
        precision, recall = kwargs[Precision.key()], kwargs[Recall.key()]
        return 2*precision*recall/(precision+recall)