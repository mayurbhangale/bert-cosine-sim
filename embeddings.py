from pathlib import Path

from abc import abstractmethod
import torch
import os
import re
import requests
import tempfile
import numpy as np
import shutil
from utils import Tqdm, cached_path
from urllib.parse import urlparse
import torch
from typing import List, Union, Dict



class WordEmbeddings():
    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, embeddings: str, field: str = None):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove'
        """
        self.embeddings = embeddings

        old_base_path = (
            "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/"
        )
       
        cache_dir = Path("embeddings")

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


