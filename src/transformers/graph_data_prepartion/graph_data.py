import io
import os
import pickle


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class GraphDataLoader():
    def __init__(self, path, max_node_num):
        self.max_node_num = max_node_num
        self.graph_instance = self.load(path)

