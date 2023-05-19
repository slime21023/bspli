from . import partitioning

class Indexing:
    def __init__(self, gl_size, ll_size):
        """
        gl_size: the leaf size of global model that used by partitioning
        ll_size: the leaf size of local model that used by partitioning
        """
        self._gl_size = gl_size
        self._ll_size = ll_size
        self._g_model = None
        self._m_model = []

    def train(data):
        
        pass