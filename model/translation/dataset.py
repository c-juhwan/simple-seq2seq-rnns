# Standard Library Modules
import pickle
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
# Pytorch Modules
from torch.utils.data.dataset import Dataset

class TranslationDataset(Dataset):
    def __init__(self, args: argparse.Namespace, data_path: str) -> None:
        super(TranslationDataset, self).__init__()
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)

        self.data_list = []
        self.src_vocabulary = data_['src_vocabulary']
        self.tgt_vocabulary = data_['tgt_vocabulary']
        self.src_vocab_size = data_['src_vocab_size']
        self.tgt_vocab_size = data_['tgt_vocab_size']
        self.pad_token_id = data_['pad_token_id']
        self.unk_token_id = data_['unk_token_id']
        self.bos_token_id = data_['bos_token_id']
        self.eos_token_id = data_['eos_token_id']

        for idx in tqdm(range(len(data_['source_ids'])), desc=f'Loading data from {data_path}'):
            source_ids = data_['source_ids'][idx]
            rev_source_ids = data_['reverse_source_ids'][idx]
            target_ids = data_['target_ids'][idx]
            self.data_list.append({
                'source_ids': source_ids if not args.reverse_input else rev_source_ids, # Reverse source ids if reverse_input is True
                'target_ids': target_ids,
                'index': idx,
            })

        del data_

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        return self.data_list[idx]
