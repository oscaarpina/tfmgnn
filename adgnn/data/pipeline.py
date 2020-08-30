from ..data.storage import GDatabase
from abc import ABC, abstractmethod

MODE_IND   = 0
MODE_TRAIN = 1
MODE_TEST  = 2

class DataLayer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self, gdb: GDatabase, mode):
        raise NotImplementedError()

class DataPipeline(DataLayer):
    def __init__(self, layers):
        super(DataPipeline, self).__init__()
        self.layers = layers

    def run(self, gdb, mode="train"):
        for l in self.layers:
            assert issubclass(type(l), DataLayer), "NON-VALID LAYER"
            gdb = l.run(gdb, mode)
            assert type(gdb) is GDatabase, "NON-VALID OUTPUT"
        return gdb

class DataLayerFunction(DataLayer):
    def __init__(self, f, **kwargs):
        super(DataLayerFunction, self).__init__()
        self.f = f
        self.kwargs = kwargs

    def run(self, gdb: GDatabase, mode):
        return self.f(gdb, mode, **self.kwargs)