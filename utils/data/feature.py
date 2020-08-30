from abc import ABC, abstractmethod
from sklearn.utils import Bunch
from ..data.pipeline import DataLayer, MODE_IND, MODE_TRAIN, MODE_TEST
from ..data.storage import GDatabase
import pandas as pd
import numpy as np

class FeaturePipeline(DataLayer):
    def __init__(self, layers, feature_code):
        self.layers = layers
        self.feature_code = feature_code

    def run(self, gdb: GDatabase, mode):
        for l in self.layers:
            if type(l) is tuple:
                l, l_mode = l[0], l[1]
            else:
                l_mode = mode
            gdb = l.run(gdb, self.feature_code, mode)
        return gdb

class FeatureLayer(ABC):
    def __init__(self):
        self.params = None

    def parse_feature_code(self, gdb: GDatabase, feature_code):
        entity, feature = feature_code.split(".")[0], feature_code.split(".")[1]
        return entity, feature

    @abstractmethod
    def compute_params(self, gdb: GDatabase, feature_code):
        raise NotImplementedError()

    def fit_params(self, gdb: GDatabase, feature_code):
        self.params = self.compute_params(gdb, feature_code)

    @abstractmethod
    def transform_feature(self, gdb: GDatabase, feature_code, params):
        raise NotImplementedError()

    def run(self, gdb: GDatabase, feature_code, mode):
        if mode is MODE_IND:
            params = self.compute_params(gdb, feature_code)
            return self.transform_feature(gdb, feature_code, params)
        elif mode is MODE_TRAIN:
            self.fit_params(gdb, feature_code)
            return self.transform_feature(gdb, feature_code, self.params)
        elif mode is MODE_TEST:
            return self.transform_feature(gdb, feature_code, self.params)


class FeatureGeneric(FeatureLayer):
    def __init__(self):
        super(FeatureGeneric, self).__init__()

    def compute_params(self, gdb: GDatabase, feature_code):
        raise NotImplementedError()

    def transform_feature(self, gdb: GDatabase, feature_code, params):
        raise NotImplementedError()


class FeatureGroupby(FeatureLayer):
    def __init__(self, gb_features):
        super(FeatureGroupby, self).__init__()
        self.gb_features = gb_features
        self.df_metrics = None

    def compute_params(self, gdb: GDatabase, feature_code):
        raise NotImplementedError()

    def transform_feature(self, gdb: GDatabase, feature_code, params):
        raise NotImplementedError()


class FeatureGraph(FeatureLayer):
    def __init__(self):
        super(FeatureGraph, self).__init__()

    def compute_params(self, gdb: GDatabase, feature_code):
        pass

    def transform_feature(self, gdb: GDatabase, feature_code, params):
        raise NotImplementedError()


class FeatureWrt(FeatureLayer):
    def __init__(self, constraint_features, wrt_features):
        super(FeatureWrt, self).__init__()
        self.constraint_features = constraint_features
        self.wrt_features = wrt_features

    def compute_params(self, gdb: GDatabase, feature_code):
        pass

    def transform_feature(self, gdb: GDatabase, feature_code, params):
        raise NotImplementedError()

class Scale(FeatureGeneric):
    def __init__(self, factor):
        super(Scale, self).__init__()
        self.factor = factor

    def compute_params(self, gdb: GDatabase, feature_code):
        return None

    def transform_feature(self, gdb: GDatabase, feature_code, params):
        e, f = self.parse_feature_code(gdb, feature_code)
        gdb[e][f] = gdb[e][f]/self.factor
        return gdb

class FScale(FeatureGeneric):
    def __init__(self, factor_feature):
        super(FScale, self).__init__()
        self.factor_feature = factor_feature

    def compute_params(self, gdb: GDatabase, feature_code):
        return None

    def transform_feature(self, gdb: GDatabase, feature_code, params):
        e, f = self.parse_feature_code(gdb, feature_code)
        gdb[e][f] = gdb[e][f]/gdb[e][self.factor_feature]
        return gdb


class ZScore(FeatureGeneric):
    def __init__(self):
        super(ZScore, self).__init__()

    def compute_params(self, gdb: GDatabase, feature_code):
        e, f = self.parse_feature_code(gdb, feature_code)
        mean, std = gdb[e][f].mean(), gdb[e][f].std()
        return Bunch(mean=mean, std=std)

    def transform_feature(self, gdb: GDatabase, feature_code, params):
        e, f = self.parse_feature_code(gdb, feature_code)
        gdb[e][f] = (gdb[e][f] - params.mean) / params.std
        return gdb

class MinMax(FeatureGeneric):
    def __init__(self):
        super(MinMax, self).__init__()

    def compute_params(self, gdb: GDatabase, feature_code):
        e, f = self.parse_feature_code(gdb, feature_code)
        min, max = gdb[e][f].min(), gdb[e][f].max()
        return Bunch(min=min, max=max)

    def transform_feature(self, gdb: GDatabase, feature_code, params):
        e, f = self.parse_feature_code(gdb, feature_code)
        gdb[e][f] = (gdb[e][f] - params.min) / (params.max - params.min)
        return gdb


class OneHot(FeatureGeneric):
    def __init__(self):
        super(OneHot, self).__init__()

    def compute_params(self, gdb: GDatabase, feature_code):
        e, f = self.parse_feature_code(gdb, feature_code)
        labels = gdb[e][f].apply(lambda x: str(int(x)) if type(x) is float else str(x)).unique()
        return Bunch(labels=labels)

    def transform_feature(self, gdb: GDatabase, feature_code, params):
        e, f = self.parse_feature_code(gdb, feature_code)
        gdb[e][f] = gdb[e][f].apply(lambda x: str(int(x)) if type(x) is float else str(x))
        gdb[e][f] = gdb[e][f].apply(lambda x: x if x in params.labels else np.nan)
        gdb[e] = pd.get_dummies(data=gdb[e], columns=[f])
        return gdb


class ZScoreGroupby(FeatureGroupby):
    def __init__(self, gb_features):
        super(ZScoreGroupby, self).__init__(gb_features)

    def compute_params(self, gdb: GDatabase, feature_code):
        e, f = self.parse_feature_code(gdb, feature_code)
        df_metrics = gdb[e].groupby(self.gb_features).agg({f: ["mean", "std"]})
        df_metrics.columns = [c[0] + "_" + c[1] for c in df_metrics.columns]
        return Bunch(df_metrics=df_metrics)

    def transform_feature(self, gdb: GDatabase, feature_code, params):
        e, f = self.parse_feature_code(gdb, feature_code)
        columns = gdb[e].columns
        gdb[e] = gdb[e].merge(params.df_metrics, how="left", left_on=self.gb_features, right_on=self.gb_features)
        gdb[e][f] = (gdb[e][f] - gdb[e][f + "_mean"]) / gdb[e][f + "_std"]
        gdb[e] = gdb[e][columns]
        return gdb


class MinMaxGroupby(FeatureGroupby):
    def __init__(self, gb_features):
        super(MinMaxGroupby, self).__init__(gb_features)

    def compute_params(self, gdb: GDatabase, feature_code):
        e, f = self.parse_feature_code(gdb, feature_code)
        df_metrics = gdb[e].groupby(self.gb_features).agg({f: ["min", "max"]})
        df_metrics.columns = [c[0] + "_" + c[1] for c in df_metrics.columns]
        return Bunch(df_metrics=df_metrics)

    def transform_feature(self, gdb: GDatabase, feature_code, params):
        e, f = self.parse_feature_code(gdb, feature_code)
        columns = gdb[e].columns
        gdb[e] = gdb[e].merge(params.df_metrics, how="left", left_on=self.gb_features, right_on=self.gb_features)
        gdb[e][f] = (gdb[e][f] - gdb[e][f + "_min"]) / (gdb[e][f + "_max"] - gdb[e][f + "_min"])
        gdb[e] = gdb[e][columns]
        return gdb

class ScaleGroupby(FeatureGroupby):
    def __init__(self, gb_features, agg="sum"):
        super(ScaleGroupby, self).__init__(gb_features)
        self.agg = agg

    def compute_params(self, gdb: GDatabase, feature_code):
        e, f = self.parse_feature_code(gdb, feature_code)
        df_metrics = gdb[e].groupby(self.gb_features).agg({f : self.agg})
        df_metrics.columns = [f + "_agg"]
        return Bunch(df_metrics=df_metrics)

    def transform_feature(self, gdb: GDatabase, feature_code, params):
        e, f = self.parse_feature_code(gdb, feature_code)
        columns = gdb[e].columns
        gdb[e] = gdb[e].merge(params.df_metrics, how="left", left_on=self.gb_features, right_on=self.gb_features)[f]
        gdb[e][f] = gdb[e][f] / gdb[e][f + "_agg"]
        gdb[e] = gdb[e][columns]
        return gdb

class ScaleGraph(FeatureGraph):
    def __init__(self, graph_feature):
        super(ScaleGraph, self).__init__()
        self.graph_feature = graph_feature

    def transform_feature(self, gdb: GDatabase, feature_code, params):
        e, f = self.parse_feature_code(gdb, feature_code)
        columns = gdb[e].columns
        gdb[e] = gdb[e].merge(gdb.graph[[self.graph_feature, gdb.id_feature]], left_on=gdb.id_feature, right_on=gdb.id_feature, how="left")
        gdb[e][f] = gdb[e][f] / gdb[e][self.graph_feature]
        gdb[e] = gdb[e][columns]
        return gdb

class SymWrt(FeatureWrt):
    def __init__(self, constraint_features, wrt_features):
        super(SymWrt, self).__init__(constraint_features, wrt_features)

    def transform_feature(self, gdb: GDatabase, feature_code, params):
        e, f = self.parse_feature_code(gdb, feature_code)
        columns = list(gdb[e].columns)
        gdb[e] = gdb[e].merge(gdb[e], how="left", left_on=self.constraint_features + self.wrt_features,
                      right_on=self.constraint_features + [self.wrt_features[1], self.wrt_features[0]],
                      suffixes=("", "_T")).fillna(0)
        gdb[e][f] = (gdb[e][f] + gdb[e][f + "_T"]) / 2
        gdb[e] = gdb[e][columns]
        return gdb