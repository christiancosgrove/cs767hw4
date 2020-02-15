#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Teachers for the MovieDialog task.

From Dodge et al. '15. Link: https://arxiv.org/abs/1511.06931

Task 1: Closed-domain QA dataset asking templated questions about movies,
answerable from Wikipedia.

Task 2: Questions asking for movie recommendations.

Task 3: Dialogs discussing questions about movies as well as recommendations.

Task 4: Dialogs discussing Movies from Reddit (the /r/movies SubReddit).
"""
from parlai.core.teachers import DialogTeacher
from .build import build

import copy
import os
import pickle
from tqdm import tqdm


def _path(opt):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'MovieTriples_Dataset.tar')

class MovieTriplesTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # store identifier for the teacher in the dialog
        self.id = 'movie_triples'

        # store paths to images and labels
        opt['datafile'] = _path(opt)

        super().__init__(opt, shared)
    def setup_data(self, path):
        print('loading: ' + path)

        # open data file with labels
        # (path will be provided to setup_data from opt['datafile'] defined above)
        with open(os.path.join(path, 'Training.triples.pkl'), 'rb') as data_file:
            self.triples = pickle.load(data_file)
        with open(os.path.join(path, 'Training.dict.pkl'), 'rb') as data_file:
            self.dictionary = pickle.load(data_file)

        # Token to split different utterances
        dict_map = {i : word for word, i, _, _ in self.dictionary }

        split_token = next(v[1] for v in self.dictionary if v[0] == '</s>')
        # every episode consists of only one query in this task
        new_episode = True
        # define iterator over all queries
        for i in tqdm(range(len(self.triples))):

            trip_mapped = ' '.join(dict_map[w] for w in self.triples[i])
            # split = trip_mapped.split('</s>')
            # question = ''.join(split[:2])
            # label = split[2]
            # yield tuple with information and new_episode? flag (always True)
            # print('Mapped ', trip_mapped)
            yield (trip_mapped, None, None, None), new_episode
        
class DefaultTeacher(MovieTriplesTeacher):
    pass