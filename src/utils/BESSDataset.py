from math import isclose
from numpy import array, concatenate, load, savez_compressed
from typing import Optional
from os import makedirs, path

from src.utils import BESSSchedulingOptModel
from timexer.data_provider.data_loader import Dataset_Custom

class BESSDataset():
    def __init__(
            self,
            dataset: Dataset_Custom,
            cache_dir: Optional[str] = None,
            **kwargs
    ):
        self.dataset = dataset
        self.pred_len = self.dataset.pred_len

        types = ['train', 'val', 'test']
        self.flag = types[self.dataset.set_type]
        self.cache_name = self.dataset.data_path+'_'+self.flag+'.npz'

        kwargs.update({'H': self.pred_len})
        self.sols, self.obj_values = self._load_bess_solutions(cache_dir, **kwargs)
        return


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        true_sols, true_objs = self.sols[index], self.obj_values[index]
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.dataset.__getitem__(index)
        item = tuple(sub_item.astype('float32') for sub_item in [seq_x, seq_y, seq_x_mark, seq_y_mark, true_sols, true_objs])
        return item


    def __getitem_prices__(self, index):
        item = self.dataset.__getitem__(index)
        seq_y = item[1]
        prices = self.dataset.inverse_transform(seq_y)
        prices = prices[-self.pred_len:, -1]
        return prices


    def _load_bess_solutions(self, cache_dir: Optional[str] = None, **kwargs):
        if cache_dir is None:
            return self._compute_bess_solutions(cache_dir, **kwargs)
        else:
            try:
                cache_path = path.join(cache_dir, self.cache_name)
                cache = load(cache_path, allow_pickle=True)
                self._validate_cached_solutions(cache['sols'], cache['obj_values'], **kwargs)

                print(f"Cached BESS solutions found at {cache_path}. Skipping pre-computation...")
                return cache['sols'], cache['obj_values']
            except Exception:
                return self._compute_bess_solutions(cache_dir, **kwargs)


    def _compute_bess_solutions(self, cache_dir: Optional[str], **kwargs):
        print('Pre-computing BESS solutions...')
        optModel = BESSSchedulingOptModel(**kwargs)

        sols, obj_values = [], []
        for i in range(self.__len__()):
            prices = self.__getitem_prices__(i)
            sol, obj_value = self._compute_bess_solution(prices, optModel)
            sols.append(sol)
            obj_values.append(obj_value)

        sols = array(sols)
        obj_values = array(obj_values).astype('float32')

        if cache_dir is not None:
            self._cache_bess_solutions(cache_dir, sols, obj_values)

        return sols, obj_values


    def _compute_bess_solution(self, prices, optModel: BESSSchedulingOptModel):
        cost_vector = concatenate([prices + optModel.C, -prices + optModel.C])
        optModel.setObj(cost_vector)
        return optModel.solve()


    def _cache_bess_solutions(self, cache_dir: str, sols: array, obj_values: array):
        makedirs(cache_dir, exist_ok=True)
        cache_path = path.join(cache_dir, self.cache_name)
        savez_compressed(cache_path, sols=sols, obj_values=obj_values)
        return


    def _validate_cached_solutions(self, sols, obj_values, **kwargs):
        if sols.shape[0] != self.__len__():
            raise Exception('Cache size does not match dataset size.')
        optModel = BESSSchedulingOptModel(**kwargs)
        prices = self.__getitem_prices__(0)
        sol, obj_value = self._compute_bess_solution(prices, optModel)
        if (not isclose(obj_value, obj_values[0], abs_tol = 0.01 )) or ((sol - sols[0]).sum() > 0.1):
            raise Exception('Cached solutions do not match dataset.')
        return
