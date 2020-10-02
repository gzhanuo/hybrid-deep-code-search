from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
import dataset.my_ast

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
