from torch_geometric.data import InMemoryDataset

class PyGDataset(InMemoryDataset):
    def __init__(self, root, data_list):
        super(PyGDataset, self).__init__(root, None, None)
        self.data, self.slices = self.collate(data_list)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return ["data.pt"]
