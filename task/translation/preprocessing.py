# Standard Library Modules
import os
import gc
import sys
import pickle
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore FutureWarning for pandas
from collections import Counter, OrderedDict
# 3rd-party Modules
import pandas as pd
from tqdm.auto import tqdm
from nltk.tokenize import WordPunctTokenizer
# Pytorch Modules
import torch
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
# Huggingface Modules
from datasets import load_dataset
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path

def preprocessing(args: argparse.Namespace) -> None:
    """
    Main function for preprocessing.

    Args:
        args (argparse.Namespace): Arguments.

    """

    # Load the data
    train_data, valid_data, test_data = load_huggingface_data(args)
    assert len(train_data['source_text']) == len(train_data['target_text'])
    print(f'Number of train data: {len(train_data)}')
    print(f'Number of valid data: {len(valid_data)}')
    print(f'Number of test data: {len(test_data)}')

    # Define tokenizer
    tokenizer = get_tokenizer(WordPunctTokenizer().tokenize)

    # Build vocabulary - Source
    counter = Counter()
    for idx in tqdm(range(len(train_data['source_text'])), desc='Building vocabulary'):
        counter.update(tokenizer(train_data['source_text'][idx]))
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    src_vocabulary = vocab(ordered_dict, specials=['<pad>', '<unk>', '<bos>', '<eos>'], min_freq=500)
    src_vocabulary.set_default_index(src_vocabulary['<unk>'])
    src_vocab_size = len(src_vocabulary)
    args.src_vocab_size = src_vocab_size

    # Build vocabulary - Target
    counter = Counter()
    for idx in tqdm(range(len(train_data['target_text'])), desc='Building vocabulary'):
        counter.update(tokenizer(train_data['target_text'][idx]))
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    tgt_vocabulary = vocab(ordered_dict, specials=['<pad>', '<unk>', '<bos>', '<eos>'], min_freq=500)
    tgt_vocabulary.set_default_index(tgt_vocabulary['<unk>'])
    tgt_vocab_size = len(tgt_vocabulary)
    args.tgt_vocab_size = tgt_vocab_size
    print(f"Source vocabulary size: {src_vocab_size} / Target vocabulary size: {tgt_vocab_size}")

    del counter, sorted_by_freq_tuples, ordered_dict
    gc.collect() # Garbage Collection

    # Assertion - Check if special token indices are same
    assert src_vocabulary['<pad>'] == tgt_vocabulary['<pad>']
    assert src_vocabulary['<unk>'] == tgt_vocabulary['<unk>']
    assert src_vocabulary['<bos>'] == tgt_vocabulary['<bos>']
    assert src_vocabulary['<eos>'] == tgt_vocabulary['<eos>']

    # Preprocessing - Define data_dict
    data_dict = {
        'train': {
            'source_ids': [],
            'reverse_source_ids': [],
            'target_ids': [],
            'src_vocabulary': src_vocabulary,
            'tgt_vocabulary': tgt_vocabulary,
            'src_vocab_size': args.src_vocab_size,
            'tgt_vocab_size': args.tgt_vocab_size,
            'pad_token_id': src_vocabulary['<pad>'],
            'unk_token_id': src_vocabulary['<unk>'],
            'bos_token_id': src_vocabulary['<bos>'],
            'eos_token_id': src_vocabulary['<eos>'],
        },
        'valid': {
            'source_ids': [],
            'reverse_source_ids': [],
            'target_ids': [],
            'src_vocabulary': src_vocabulary,
            'tgt_vocabulary': tgt_vocabulary,
            'src_vocab_size': args.src_vocab_size,
            'tgt_vocab_size': args.tgt_vocab_size,
            'pad_token_id': src_vocabulary['<pad>'],
            'unk_token_id': src_vocabulary['<unk>'],
            'bos_token_id': src_vocabulary['<bos>'],
            'eos_token_id': src_vocabulary['<eos>'],
        },
        'test': {
            'source_ids': [],
            'reverse_source_ids': [],
            'target_ids': [],
            'src_vocabulary': src_vocabulary,
            'tgt_vocabulary': tgt_vocabulary,
            'src_vocab_size': args.src_vocab_size,
            'tgt_vocab_size': args.tgt_vocab_size,
            'pad_token_id': src_vocabulary['<pad>'],
            'unk_token_id': src_vocabulary['<unk>'],
            'bos_token_id': src_vocabulary['<bos>'],
            'eos_token_id': src_vocabulary['<eos>'],
        }
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset)
    check_path(preprocessed_path)

    for idx in tqdm(range(len(valid_data['source_text'])), desc=f'Preprocessing valid data'):
        # Get source and target text
        source_text = valid_data['source_text'][idx]
        target_text = valid_data['target_text'][idx]

        # Tokenize source and target text using tokenizer
        source_tokenized = tokenizer(source_text)
        target_tokenized = tokenizer(target_text)
        # Convert each token to index using vocabulary
        source_tokenized = [src_vocabulary[token] for token in source_tokenized]
        target_tokenized = [tgt_vocabulary[token] for token in target_tokenized]
        # Add <bos> and <eos> token
        source_tokenized = [src_vocabulary['<bos>']] + source_tokenized + [src_vocabulary['<eos>']]
        target_tokenized = [tgt_vocabulary['<bos>']] + target_tokenized + [tgt_vocabulary['<eos>']]

        # Add padding to source
        if len(source_tokenized) < args.max_seq_len:
            source_tokenized += [src_vocabulary['<pad>']] * (args.max_seq_len - len(source_tokenized))
        else: # Truncate
            source_tokenized = source_tokenized[:args.max_seq_len]
            source_tokenized[-1] = src_vocabulary['<eos>'] # Replace last token with <eos>
        # Add padding to target
        if len(target_tokenized) < args.max_seq_len:
            target_tokenized += [tgt_vocabulary['<pad>']] * (args.max_seq_len - len(target_tokenized))
        else: # Truncate
            target_tokenized = target_tokenized[:args.max_seq_len]
            target_tokenized[-1] = tgt_vocabulary['<eos>'] # Replace last token with <eos>

        # Convert each tokenized text to tensor
        source_tokenized = torch.tensor(source_tokenized, dtype=torch.long)
        target_tokenized = torch.tensor(target_tokenized, dtype=torch.long)

        # Append data to data_dict
        data_dict['valid']['source_ids'].append(source_tokenized)
        data_dict['valid']['reverse_source_ids'].append(source_tokenized.flip(0)) # Reverse source for paper implementation
        data_dict['valid']['target_ids'].append(target_tokenized)
        del source_text, target_text, source_tokenized, target_tokenized # Delete variables after saving
    # Save data_dict as pickle file
    with open(os.path.join(preprocessed_path, f'valid_processed.pkl'), 'wb') as f:
        pickle.dump(data_dict['valid'], f)
    del valid_data, data_dict['valid'] # Delete variables after saving

    for idx in tqdm(range(len(test_data['source_text'])), desc=f'Preprocessing test data'):
        # Get source and target text
        source_text = test_data['source_text'][idx]
        target_text = test_data['target_text'][idx]

        # Tokenize source and target text using tokenizer
        source_tokenized = tokenizer(source_text)
        target_tokenized = tokenizer(target_text)
        # Convert each token to index using vocabulary
        source_tokenized = [src_vocabulary[token] for token in source_tokenized]
        target_tokenized = [tgt_vocabulary[token] for token in target_tokenized]
        # Add <bos> and <eos> token
        source_tokenized = [src_vocabulary['<bos>']] + source_tokenized + [src_vocabulary['<eos>']]
        target_tokenized = [tgt_vocabulary['<bos>']] + target_tokenized + [tgt_vocabulary['<eos>']]

        # Add padding to source
        if len(source_tokenized) < args.max_seq_len:
            source_tokenized += [src_vocabulary['<pad>']] * (args.max_seq_len - len(source_tokenized))
        else: # Truncate
            source_tokenized = source_tokenized[:args.max_seq_len]
            source_tokenized[-1] = src_vocabulary['<eos>'] # Replace last token with <eos>
        # Add padding to target
        if len(target_tokenized) < args.max_seq_len:
            target_tokenized += [tgt_vocabulary['<pad>']] * (args.max_seq_len - len(target_tokenized))
        else: # Truncate
            target_tokenized = target_tokenized[:args.max_seq_len]
            target_tokenized[-1] = tgt_vocabulary['<eos>'] # Replace last token with <eos>

        # Convert each tokenized text to tensor
        source_tokenized = torch.tensor(source_tokenized, dtype=torch.long)
        target_tokenized = torch.tensor(target_tokenized, dtype=torch.long)

        # Append data to data_dict
        data_dict['test']['source_ids'].append(source_tokenized)
        data_dict['test']['reverse_source_ids'].append(source_tokenized.flip(0)) # Reverse source for paper implementation
        data_dict['test']['target_ids'].append(target_tokenized)
        del source_text, target_text, source_tokenized, target_tokenized # Delete variables after saving
    # Save data_dict as pickle file
    with open(os.path.join(preprocessed_path, f'test_processed.pkl'), 'wb') as f:
        pickle.dump(data_dict['test'], f)
    del test_data, data_dict['test'] # Delete variables after saving
    gc.collect() # Collect garbage

    # Train data is too large, so we split it into 10 parts and save each part -> merge them later
    train_split = 10
    train_split_size = len(train_data['source_text']) // train_split

    for split_idx in range(train_split):
        train_data_split = {
        'source_text': [],
        'target_text': [],
        }
        train_data_split['source_text'] = train_data['source_text'][split_idx * train_split_size: (split_idx + 1) * train_split_size]
        train_data_split['target_text'] = train_data['target_text'][split_idx * train_split_size: (split_idx + 1) * train_split_size]
        data_dict_split = {
            'source_ids': [],
            'reverse_source_ids': [],
            'target_ids': [],
            'src_vocabulary': src_vocabulary,
            'tgt_vocabulary': tgt_vocabulary,
            'src_vocab_size': args.src_vocab_size,
            'tgt_vocab_size': args.tgt_vocab_size,
            'pad_token_id': src_vocabulary['<pad>'],
            'unk_token_id': src_vocabulary['<unk>'],
            'bos_token_id': src_vocabulary['<bos>'],
            'eos_token_id': src_vocabulary['<eos>'],
        }
        for idx in tqdm(range(len(train_data_split['source_text'])), desc=f'Preprocessing train data (split {split_idx})'):
            # Get source and target text
            source_text = train_data_split['source_text'][idx]
            target_text = train_data_split['target_text'][idx]

            # Tokenize source and target text using tokenizer
            source_tokenized = tokenizer(source_text)
            target_tokenized = tokenizer(target_text)
            # Convert each token to index using vocabulary
            source_tokenized = [src_vocabulary[token] for token in source_tokenized]
            target_tokenized = [tgt_vocabulary[token] for token in target_tokenized]
            # Add <bos> and <eos> token
            source_tokenized = [src_vocabulary['<bos>']] + source_tokenized + [src_vocabulary['<eos>']]
            target_tokenized = [tgt_vocabulary['<bos>']] + target_tokenized + [tgt_vocabulary['<eos>']]

            # Add padding to source
            if len(source_tokenized) < args.max_seq_len:
                source_tokenized += [src_vocabulary['<pad>']] * (args.max_seq_len - len(source_tokenized))
            else: # Truncate
                source_tokenized = source_tokenized[:args.max_seq_len]
                source_tokenized[-1] = src_vocabulary['<eos>']

            # Add padding to target
            if len(target_tokenized) < args.max_seq_len:
                target_tokenized += [tgt_vocabulary['<pad>']] * (args.max_seq_len - len(target_tokenized))
            else: # Truncate
                target_tokenized = target_tokenized[:args.max_seq_len]
                target_tokenized[-1] = tgt_vocabulary['<eos>'] # Replace last token with <eos>

            # Convert each tokenized text to tensor
            source_tokenized = torch.tensor(source_tokenized, dtype=torch.long)
            target_tokenized = torch.tensor(target_tokenized, dtype=torch.long)

            # Append data to data_dict
            data_dict_split['source_ids'].append(source_tokenized)
            data_dict_split['reverse_source_ids'].append(source_tokenized.flip(0)) # Reverse source for paper implementation
            data_dict_split['target_ids'].append(target_tokenized)

            del source_text, target_text, source_tokenized, target_tokenized # Delete variables after saving

        # Save data_dict_split as pickle file
        with open(os.path.join(preprocessed_path, f'train_processed_{split_idx}.pkl'), 'wb') as f:
            pickle.dump(data_dict_split, f)
        del train_data_split, data_dict_split # Delete variables after saving

    gc.collect() # Collect garbage
    # Merge train data
    data_dict['train'] = {
        'source_ids': [],
        'reverse_source_ids': [],
        'target_ids': [],
        'src_vocabulary': src_vocabulary,
        'tgt_vocabulary': tgt_vocabulary,
        'src_vocab_size': args.src_vocab_size,
        'tgt_vocab_size': args.tgt_vocab_size,
        'pad_token_id': src_vocabulary['<pad>'],
        'unk_token_id': src_vocabulary['<unk>'],
        'bos_token_id': src_vocabulary['<bos>'],
        'eos_token_id': src_vocabulary['<eos>'],
    }

    for split_idx in range(train_split):
        with open(os.path.join(preprocessed_path, f'train_processed_{split_idx}.pkl'), 'rb') as f:
            data_dict_split = pickle.load(f)
        data_dict['train']['source_ids'] += data_dict_split['source_ids']
        data_dict['train']['reverse_source_ids'] += data_dict_split['reverse_source_ids']
        data_dict['train']['target_ids'] += data_dict_split['target_ids']
        del data_dict_split

    # Save data_dict as pickle file
    with open(os.path.join(preprocessed_path, 'train_processed.pkl'), 'wb') as f:
        pickle.dump(data_dict['train'], f)

def load_huggingface_data(args: argparse.Namespace) -> tuple:
    """
    Load data from huggingface datasets.
    If dataset is not in huggingface datasets, takes data from local directory.

    Args:
        args (argparse.Namespace): Arguments.

    Returns:
        train_data (dict): Training data. (source, target)
        valid_data (dict): Validation data. (source, target)
        test_data (dict): Test data. (source, target)
    """

    name = args.task_dataset.lower()

    train_data = {
        'source_text': [],
        'target_text': [],
    }
    valid_data = {
        'source_text': [],
        'target_text': [],
    }
    test_data = {
        'source_text': [],
        'target_text': [],
    }

    # Load the data
    source_lang = 'en'
    if name == 'wmt14_en_cs':
        target_lang = 'cs'
        dataset = load_dataset('wmt14', 'cs-en')
    elif name == 'wmt14_en_de':
        target_lang = 'de'
        dataset = load_dataset('wmt14', 'de-en')
    elif name == 'wmt14_en_fr':
        target_lang = 'fr'
        dataset = load_dataset('wmt14', 'fr-en')
    elif name == 'wmt14_en_hi':
        target_lang = 'hi'
        dataset = load_dataset('wmt14', 'hi-en')
    elif name == 'wmt14_en_ru':
        target_lang = 'ru'
        dataset = load_dataset('wmt14', 'ru-en')
    print(f"Loaded dataset {name} from huggingface datasets.")

    for each_translation in tqdm(dataset['train']['translation'], desc='Loading train data'):
        train_data['source_text'].append(each_translation[source_lang])
        train_data['target_text'].append(each_translation[target_lang])

    for each_translation in tqdm(dataset['validation']['translation'], desc='Loading valid data'):
        valid_data['source_text'].append(each_translation[source_lang])
        valid_data['target_text'].append(each_translation[target_lang])

    for each_translation in tqdm(dataset['test']['translation'], desc='Loading test data'):
        test_data['source_text'].append(each_translation[source_lang])
        test_data['target_text'].append(each_translation[target_lang])

    del dataset
    gc.collect() # Collect garbage
    return train_data, valid_data, test_data
