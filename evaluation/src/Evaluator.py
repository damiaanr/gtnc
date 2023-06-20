from typing import Union as U
from .Scores import Scores
import time
import itertools
import glob
import pickle
import os
import sys

# Unfortunately, I don't really see any other way of serving this code
# on a less cumbersome way (Python imports are a bit counter-intuitive
# sometimes...).
if '.' in __package__:  # considering 'evaluation' as external package
    from ..bases.wals import WalsEvaluator  # so, relative from here
else:  # run directly from within the folder
    from bases.wals import WalsEvaluator  # now, relative from the root


class Evaluator:
    """
    Main class that dynamically interacts with different evaluator
    classes that each encodes a different approach (e.g., comparing
    Wals features). Upon initiating an instance of this class, the
    scores will be evaluated (or loaded from cache) automatically and
    can be accessed through the scores property.
    """
    CACHE_FOLDER = 'scores/'
    LANGUAGES_FILE = 'languages.txt'
    METHODS = ['Wals']

    def __init__(self, method: str = 'Wals', load_from_cache: bool = False,
                 save_to_cache: bool = True, cache_file: str = None,
                 cache_folder: str = None,
                 languages_of_interest: U[list, str] = None,
                 ev_params: dict = {}, sc_params: dict = {}):
        """
        Init (see class description above).

        In:
          @method:          Evaluation `base' (key in global METHODS)
          @load_from_cache: if False, scores are freshly calculated and
                            saved to cache if @save_to_cache is True
          @save_to_cache:   if True, store scores after (re-)evaluation
          @cache_file:      Pickle file in global CACHE_FOLDER (incl.
                            extension); if not set, the latest file
                            in the folder is loaded
          @cache_folder:    directory to which to write cache files if
                            @save_to_cache is True
          @languages_of:    If set, either list of ISO639-3 codes or
           _interest        a string representing a relative file-path
                            with language codes seperated by newlines;
                            if None, global LANGUAGES_FILE will be
                            loaded (see load_languages_of_interest())
          @ev_params:       directory of parameters (keys corresponding
                            to parameter name, values to values) that
                            are passed to the base-specific evaluator
                            (e.g., WalsEvaluator)

        Out:
          - void
        """
        if method not in self.METHODS:
            raise Exception("Given method not supported")

        self.method = method
        # dynamic class load below
        self.service = globals()[method + "Evaluator"](**ev_params)

        if languages_of_interest is None or \
                type(languages_of_interest) is str:
            self.load_languages_of_interest()
        elif type(languages_of_interest) is list:
            self.languages = languages_of_interest
        else:
            raise Exception('Illegal format of languages of interest')

        if cache_folder is not None:
            self.CACHE_FOLDER = cache_folder

        try:  # loading from cache
            if not load_from_cache:
                raise Exception('loading from cache disabled')

            self.load_scores_from_cache(cache_file=cache_file)
        except Exception:  # calculating scores
            self.evaluate_scores(sc_params)

            if save_to_cache:
                self.save_scores_to_cache()

    def load_scores_from_cache(self, cache_file=None) -> None:
        """
        Loads pickle file from cache and stores scores within class.

        In:
          @cache_file: if not set, latest cache file will be loaded

        Out:
          - void
        """
        if not cache_file:
            latest_files = glob.glob(self.CACHE_FOLDER + self.method + '_*.p')

            if latest_files:
                cache_file = max(latest_files, key=os.path.getctime)
            else:
                raise Exception("Cache is empty")
        else:
            cache_file = self.CACHE_FOLDER + cache_file

        self.scores = pickle.load(open(cache_file, "rb"))

    def save_scores_to_cache(self) -> None:
        """
        Stores scores in a pickle file for quicker loading on
        consecutive runs. Files are stored in the globally denoted
        cache folder. File names are based on the current timestamp.

        In:
          - void

        Out:
          - void
        """
        cache_file = self.CACHE_FOLDER + self.method + '_' \
            + str(time.time()) + '.p'
        pickle.dump(self.scores, open(cache_file, 'wb'))

    def evaluate_scores(self, sc_params: dict = {}) -> None:
        """
        Calculates score for every possible language pair and stores it
        in the class instance. (Note: it uses combinations, not perms.)

        In:
          - void

        Out:
          - void
        """
        self.scores = Scores()

        for l1, l2 in itertools.combinations(self.languages, r=2):
            self.scores[(l1, l2)] = self.service.evaluate_score(l1, l2,
                                                                **sc_params)

    def load_languages_of_interest(self, file: str = None) -> None:
        """
        To-be-analysed languages can be loaded directly from a file,
        instead of manually inserted upon initialisation. The file
        should contain an ISO639-3 language code per line/language.

        In:
          @file: relative path to file containing language codes; if
                 not given, filename is taken from the global
                 LANGUAGES_FILE by default

        Out:
          - void
        """
        if not file:
            file = self.LANGUAGES_FILE

        if not os.path.isfile(file):
            raise Exception("No languages of interest specified")

        handle = open(file, 'r')
        data = handle.read()
        self.languages = data.splitlines()
        handle.close()
