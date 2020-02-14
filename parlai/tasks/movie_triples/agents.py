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
from parlai.core.teachers import FbDialogTeacher, MultiTaskTeacher
from .build import build

import copy
import os
import pickle


def _path(opt, filtered):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'Twitter', dt + '.txt')

class MnistQATeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        self.datatype = opt['datatype'].split(':')[0]

        # store identifier for the teacher in the dialog
        self.id = 'mnist_qa'

        # store paths to images and labels
        opt['datafile'], self.image_path = _path(opt)

        super().__init__(opt, shared)
    def setup_data(self, path):
        print('loading: ' + path)

        # open data file with labels
        # (path will be provided to setup_data from opt['datafile'] defined above)
        with open(path) as data_file:
            self.data = pickle.load(data_file)

        # every episode consists of only one query in this task
        new_episode = True

        # define iterator over all queries
        for i in range(len(self.data)):

            split = self.data[i][::-1].index(0)
            question = self.data[i][:-(split + 1)]
            label = self.data[i][split:]
            # yield tuple with information and new_episode? flag (always True)
            yield (question, label, None, None, i), new_episode
        
class DefaultTeacher(MnistQATeacher):
    pass