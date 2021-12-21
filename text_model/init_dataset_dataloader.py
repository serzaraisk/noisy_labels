from torch import float as torch_float
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, RawField, Iterator
from preprocess import tokenizer_only_tags, tokenizer_lemma_tags


def create_dataset(source_folder):
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch_float, is_target=True)
    text_field = Field(tokenize=tokenizer_only_tags,  include_lengths=True, batch_first=True)
    fields = [('query', text_field), ('label', label_field), ('answer', label_field)]

    # TabularDataset

    train, valid, test = TabularDataset.splits(path=source_folder, train='train.csv', validation='valid.csv', test='test.csv',
                                               format='CSV', fields=fields, skip_header=True) 
    
    all_for_vocab, _, _ = TabularDataset.splits(path=source_folder, train='all.csv', validation='valid.csv', test='test.csv',
                                               format='CSV', fields=fields, skip_header=True) 
    text_field.build_vocab(all_for_vocab)
    
    return train, valid, test, text_field.vocab
    
def create_iterators(train_dataset, valid_dataset, test_dataset, device):
    train_iter = Iterator(train_dataset, batch_size=32, sort_key=lambda x: len(x.text),
                                device=device, sort=False, sort_within_batch=False)
    valid_iter = Iterator(valid_dataset, batch_size=32, sort_key=lambda x: len(x.text),
                                device=device, sort=False, sort_within_batch=False)
    test_iter = Iterator(test_dataset, batch_size=32, sort_key=lambda x: len(x.text),
                                device=device, sort=False, sort_within_batch=False)
    return train_iter, valid_iter, test_iter