from multiprocessing import Pool
from tools.data_io import DataFolder


class AbstractParallelRoutine:
    def __init__(self, config, in_folder, file_list_txt=None):
        self._config = config
        self._in_data_folder = DataFolder(in_folder, file_list_txt)
        self._num_processes = config['num_processes']

    def run_parallel(self):
        pool = Pool(processes=self._num_processes)
        chunk_list = self._in_data_folder.get_chunks_list(self._num_processes)
        result_obj_list = [pool.apply_async(self._run_chunk,
                                          (file_idx_chunk,))
                           for file_idx_chunk in chunk_list]
        result_list = []
        for thread_idx in range(len(result_obj_list)):
            result_obj = result_obj_list[thread_idx]
            result_obj.wait()
            print(f'Thread with idx {thread_idx} / {len(result_obj_list)} is completed', flush=True)
            result_list = result_list + result_obj.get()

        return result_list

    def run_non_parallel(self, idx_list=None):
        run_idx_list = idx_list if idx_list else range(self.num_files())
        for idx in run_idx_list:
            self._in_data_folder.print_idx(idx)
            self._run_single_scan(idx)
            # print('Done', flush=True)

    def _run_single_scan(self, idx):
        raise NotImplementedError

    def _run_chunk(self, chunk_list):
        result_list = []
        for idx in chunk_list:
            self._in_data_folder.print_idx(idx)
            result = self._run_single_scan(idx)
            result_list.append(result)
        return result_list

    def num_files(self):
        return self._in_data_folder.num_files()
