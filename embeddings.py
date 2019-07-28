from pathlib import Path

from abc import abstractmethod
import torch
import os
import re
import requests
import tempfile
import numpy as np
import shutil
import gensim
import bert_models
from utils import Tqdm, cache_root, cached_path
from urllib.parse import urlparse
import torch
from typing import List, Union, Dict



class WordEmbeddings():
    """Standard static word embeddings, such as GloVe"""

    def __init__(self, embeddings: str, field: str = None):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings :'glove'
        """
        self.embeddings = embeddings

        old_base_path = (
            "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/"
        )
       
        cache_dir = Path("wordembeddings")

        # GLOVE embeddings
        if embeddings.lower() == "glove" or embeddings.lower() == "en-glove":
            cached_path("{}glove.gensim.vectors.npy".format(old_base_path), cache_dir=cache_dir)
            embeddings = cached_path(
                "{}glove.gensim".format(old_base_path), cache_dir=cache_dir
            )
        elif not Path(embeddings).exists():
            raise ValueError(
                'The given embeddings "{}" is not available or is not a valid path.'.format(embeddings)
            )

        self.name = str(embeddings)
        self.static_embeddings = True

        if str(embeddings).endswith(".bin"):
            self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
                str(embeddings), binary=True
            )
        else:
            self.precomputed_word_embeddings = gensim.models.KeyedVectors.load(
                str(embeddings)
            )

        self.field = field

        self.__embedding_length= self.precomputed_word_embeddings.vector_size
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    
    def __str__(self):
        return self.name

class BertCosineSimilarity():
    """Pytorch BERT to generate sentence embedding"""
    def __init__(self, field: str = None):
        """
        Initializes bert embeddings. Constructor downloads required files if not there.
        :param embeddings :
        """
        self.embeddings = "bert"

        bert_model = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin'
        bert_config = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json'
        bert_vocab = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt'

       
        #self.cache_dir = Path("pytorchBertEmbeddings")
        self.cache_dir = Path("bert-base-uncased")
        print("Downloading bert-base-uncased")
    
        cached_path(bert_model, cache_dir=self.cache_dir)
        cached_path(bert_config, cache_dir=self.cache_dir)
        cached_path(bert_vocab, cache_dir=self.cache_dir)
        self.loadPreTrainedModel()

    def getModelConfig(self, fdir):
        fconfig = os.path.join(fdir, "bert_config.json")
        fweight = os.path.join(fdir, "pytorch_model.bin")

        config = bert_models.BertConfig(fconfig)
        return config, fweight
    
    def loadPreTrainedModel(self, emb_size = 1024):
        #config, fweight = self.getModelConfig(cache_root / self.cache_dir)
        config, fweight = self.getModelConfig("/Users/deepampatel/Desktop/deepam/Timepass/bert-cosine-sim/data/bert-base-uncased")
        emb_size = emb_size
        self.model = bert_models.BertPairSim(config, emb_size)
        self.doLoadModel(fweight)
        self.model

    def doLoadModel(self,fpath):
        state_dict = torch.load(fpath, map_location='cpu' if not torch.cuda.is_available() else None)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(self.model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            starext_prefix = 'bert.'
        load(self.model,prefix=start_prefix)
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                            self.model.__class__.__name__, "\n\t".join(error_msgs)))
    