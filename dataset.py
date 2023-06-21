import itertools
import glob
import gzip
import regex
import numpy as np
import math
import random
import pickle
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as patches
import matplotlib.colors as colors
import justext
from collections import defaultdict
from typing import Union as U, Tuple as T
from evaluation import *
from functions import *
import os


class SourceDataset:
    """
    This class holds all methodology required to generate and analyse a
    collection of source texts that may be translated using a translation
    service to obtain a collection of target texts, together forming a
    many-languages-to-one translation dataset. The source texts are sampled
    from NewsCrawl (https://aclanthology.org/2022.wmt-1.1/; Findings of the
    2022 Conference on Machine Translation (WMT22) (Kocmi et al., WMT 2022).

    The sample unit is a sentence: NewsCrawl consists of single sentences.

    Generally, the following stages are involved in creating the collection:

    [1] Picking and analysing languages
        - A set of languages must be given upon class initialisation; and
        - the languages may be analysed in terms of their diversity respective
          to each other (.diversity_info() and .diversity_heatmap()), and
          their average sample length (.length_info()).

    [2] Loading from NewsCrawl (implicit)
        - NewsCrawl contains sentences scraped from online news outlets
        - NewsCrawl data is already cleaned of duplicates
        - NewsCrawl data is already shuffled
        - NewsCrawl data is 'parallel by year': only sentences scraped from
          2020, 2021, and 2022 are included in the subset used by this class

    [3] Filtering and counting (explicit: .process_counts())
        - Samples with a length outside of pre-defined boundaries are removed
        - Samples that match a pre-defined regular expression are removed
        - Low-quality sentences are removed using JusText 3
            Note: this step is not supported for the following languages: am,
                  sn, ti, rw, om, pa, ha, so, ps, or, ja
        - The lengths of all remaining samples are counted and stored in cache

    [4] Align (explicit: .align())
        - Load sample length info from cache
        - Finds an equally sized sample for every language with a given
          average(/median) length
        - Returns locations of sentences within this sample (filenames, lines)

    ([5] Scoring with Monocleaner (outside of class))

    ([6] Translating with Google Cloud API (outside of class))

    In summary, given a list of languages as input, this class serves the
    locations of aligned, high-quality sentences as output product.
    """

    # Pre-defined config, may be changed upon initialisation
    config = {
        'MIN_LENGTH': 30,
        'MAX_LENGTH': 400,  # very unnatural
        'MAX_PARSE_PER_FILE': 1000000,  # way sufficient for <10K sents/lang
        'FOLDER': 'news-crawl/',
        'CACHE': 'cache/',
        'JUSTEXT_ACCEPT': ['neargood', 'good'],  # reject 'bad', 'short'

        # All RegExs match:
        #
        # - All characters that are not part of 'normal' sentences (i.e., not
        #   usual interpunction, such as ;:()!?, and equivalences in other
        #   languages). Non-latin-script languages do not allow latin letters,
        #   while latin-script languages do not allow characters outside of
        #   these interpunction marks and latin A-z.
        # - Characters that directly follow a period (.) and are not a
        #   whitespace, a digit, a question mark, another period or an
        #   exclamation mark.
        # - Four consecutive identical characters (e.g., wwww)
        # - Sentences that do not end with a period, exclamation mark,
        #   question mark, or equivalences in other languages.
        'REGEX_LATIN': r'[^\pL\pM\p{Nd},.:;\-\(\)!?\'"“”‘ ’\s]'
                       r'|\.(?![\d\s?.!])(?=[^\s.])|(\pL)\1{3}|(?<![.!?])$',
        'REGEX_NON_LATIN': r'[^\pL\pM\p{Nd}ല്‍ര്‍\-、“”‘ ’·《》«»，：～」「『』・）（；'
                           r';,ه‌ی‌ت\.:۰؛፧።፣।፤፥፦፡÷،؟？。！\(\)!?\'"\s]|[A-Za-z]'
                           r'|\.(?![\d\s?.!])(?=[^\s.])|(\pL)\1{3}'
                           r'|[\p{Nd}.,]{1,}[^\p{Nd}.,]{1,}[\p{Nd}.,]{1,}'
                           r'[^\p{Nd}.,]{1,}[\p{Nd}.,]{1,}'
                           r'|(?<![.!?፧።؟।？。！፡])$',
        # - Russian language RegEx explicitly matches Ukrainian-specific
        #   cyrillic charachters as Russian texts in NewsCrawl were found
        #   to contain Ukrainian texts as well.
        'REGEX_RUSSIAN': r'[^\pL\pM\p{Nd}ല്‍ര്‍\-、“”‘ ’·《》«»，：～」「『』・）（；'
                         r';,ه‌ی‌ت\.:۰؛፧።፣।፤፥፦፡÷،؟？。！\(\)!?\'"\s]'
                         r'|[A-Za-zҐґЄєЇїІі]'
                         r'|\.(?![\d\s?.!])(?=[^\s.])|(\pL)\1{3}'
                         r'|[\p{Nd}.,]{1,}[^\p{Nd}.,]{1,}[\p{Nd}.,]{1,}'
                         r'[^\p{Nd}.,]{1,}[\p{Nd}.,]{1,}|(?<![.!?፧።؟।？。！፡])$'
    }

    def __init__(self, languages: list, config: dict = None,
                 no_process: bool = False) -> None:
        """
        Init (see class description above).

        In:
          @languages:  list of languages that should contain a tuple per
                       language that contains [0] ISO639-1 language code,
                       [1] JusText 3 language model name or None if not
                       available or not intended [2] ISO639-3 code of
                       language [3] Boolean indicating whether language has
                       a latin script or not
          @config:     dict containing config elements to be optionally over-
                       written from the default values determined above
          @no_process: if True, do not immediately parse all dataset files to
                       filter and count all samples (adviced, e.g., when only
                       analysing diversity between languages)

        Out:
          n/a
        """
        self.languages = languages

        if config is not None:
            for key, value in config.items():
                self.config[key] = value

        if not no_process:
            self.process_counts()

    def diversity_info(self) -> None:
        """
        Prints the average overlap between WALS features of one language
        between all other languages present in the dataset, for every
        language.

        In:
          n/a

        Out:
          n/a
        """
        lang_codes = [lang[2] for lang in self.languages]
        wals_feats = {lang[2]: lang[3] for lang in self.languages}

        # Evaluator is a separate program written specifically to measure
        # language similarity or 'label fuzziness' in relation to the task
        # of Source Language Identification (SLI) in Machine Translation.
        e = Evaluator(load_from_cache=False,
                      cache_folder='evaluation/scores/',
                      languages_of_interest=lang_codes,
                      ev_params={
                        'load_from_cache': True,
                        'db_data_path': 'evaluation/bases/wals/data/',
                        'normalise': False,
                        'cache_folder': "evaluation/bases/wals/cache/"},
                      sc_params={
                        'cprefs': wals_code_preferences})

        scores = defaultdict(list)  # collect, then take average

        for l1, l2 in itertools.combinations(lang_codes, r=2):
            score = e.scores[(l1, l2)]
            if score > 0:
                scores[l1].append(score)
                scores[l2].append(score)

        for i in range(0, len(lang_codes), 6):
            print()  # clean line

            for language in lang_codes[i:i+6]:
                print(f'{language}'.rjust(10), end=' ')

            print()

            for language in lang_codes[i:i+6]:
                if len(scores[language]) > 0:
                    # average per lang
                    total = len(scores[language])
                    avg = sum(scores[language])/total

                    print(f'({total}) {avg:.2f}'.rjust(10), end=' ')
                else:
                    print(f'-'.rjust(10), end=' ')

            print()

            for language in lang_codes[i:i+6]:
                print(f'{wals_feats[language]}'.rjust(10), end=' ')

            print()

    def diversity_heatmap(self, areas_of_interest: list = [2, 6]) -> None:
        """
        Plots a qualitative 'heatmap' of all languages and their corresponding
        WALS features to visualise the diversity of the languages contained
        in the collection. Every row contains the class-indices of a certain
        language for every WALS feature (for every column). The color of each
        square (element within the matrix, row-column index) denotes a
        different class index (i.e., many different colors within a column
        indicates that this specific WALS feature is very much varied across
        languages and therefore a highly contrastive feature). The rows may be
        seen as the unique 'fingerprints' of a language in terms of its WALS
        features in relation to others.

        In:
          - @areas_of_interest: highlight specific areas of WALS features by
                                blue boundaries in the graph, choice of: (from
                                left to right; from WALS' areas.csv):

                                #1  Phonology
                                #2  Morphology
                                #3  Nominal Categories
                                #4  Nominal Syntax
                                #5  Verbal Categories
                                #6  Word Order
                                #7  Simple Clauses
                                #8  Complex Sentences
                                #9  Lexicon
                                #10 Sign Languages
                                #11 Other

                                per default, features in the areas of
                                morphology (left) and word order (right)
                                are marked
        Out:
          - n/a
        """
        lang_3_codes = [lang[2] for lang in self.languages]  # iso639-3
        lang_1_codes = [lang[0] for lang in self.languages]  # iso639-1

        # First, get the feature values for each language
        #  Note: Only feature values that occur in one of each of the defined
        #        languages will be included.
        features, parameters, _, lang_db = get_feature_vectors_for_languages(
                                                          lang_3_codes,
                                                          force_overlap=False)
        X = np.array(list(features.values()), dtype=int)  # the matrix to plot

        # Now, sort the features (y-axis) by area (e.g. morphology)
        chapters_data = LanguageDB.get_chapters_fields(
                                                  data_path=lang_db.DATA_PATH)

        areas = []

        for param in parameters:
            chapter_id = lang_db.params_chapter_mapping[param]
            area = chapters_data[chapter_id][3]
            areas.append(int(area))

        sorted_areas = np.argsort(areas, kind='stable')
        sorted_areas_values = np.array(areas)[sorted_areas]  # for boundaries

        X = X[:, sorted_areas]

        # Then, let most 'dense' languages (i.e., languages for which most
        # typological features are set) appear on the top of the matrix.
        #  Note: features not set are denoted by -1 in the features variable.
        sorted_indices = np.argsort(np.sum(X == -1, axis=1))
        sorted_labels = [lang_1_codes[i] for i in sorted_indices]

        X = X[sorted_indices]

        # Preparing plotting of the image
        fig, ax = plt.subplots()
        fig.set_dpi(100)  # as the matrix itself will always be saved as a PNG

        y_ticks = np.arange(1, X.shape[0], 2)  # two y axes are used as labels
        y_ticks2 = np.arange(0, X.shape[0], 2)  # are otherwise too dense

        # Left y-axis (starts at pos 1 insttead of 0, steps of 2)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(sorted_labels[1::2])

        # As labels are still too dense, move half of the labels more left
        tick_labels = ax.get_yticklabels()
        trans = ax.get_yaxis_transform()

        for i, label in enumerate(tick_labels):  # every label individually
            if i % 2 == 1:
                st = transforms.ScaledTranslation(-.25,
                                                  0,
                                                  plt.gcf().dpi_scale_trans)
                label.set_transform(trans + st)
                label.set_verticalalignment('center')
                ax.yaxis.get_major_ticks()[i].tick1line.set_markersize(13)

        # Right y-axis (no .twinx() as that will cause problems rendering)
        ax_r = ax.secondary_yaxis('right')
        # sorted_labels.reverse()  needed if .twinx()

        ax_r.set_yticks(y_ticks2)
        ax_r.set_yticklabels(sorted_labels[::2])

        tick_labels = ax_r.get_yticklabels()
        trans = ax_r.get_yaxis_transform()

        for i, label in enumerate(tick_labels):
            if i % 2 == 1:
                st = transforms.ScaledTranslation(.25, 0,
                                                  plt.gcf().dpi_scale_trans)
                label.set_transform(trans + st)
                label.set_verticalalignment('center')
                ax_r.yaxis.get_major_ticks()[i].tick2line.set_markersize(13)

        # Only ticks (for every feature) on the x-axis
        ax.set_xticks(range(X.shape[1]))
        ax.set_xticklabels([] * X.shape[1])

        # Normalised colormap, chosen to work well in highlighting diversity
        X = np.ma.masked_equal(X, -1)  # -1 values are always visualised white
        norm = colors.Normalize(vmin=-8, vmax=1 * X.max())

        # Finally, the image is plotted
        #  Note: interpolation='none' is needed when exporting as .PGF to
        #        produce a good-quality .PNG (a PNG is unavoidable for the
        #        matrix part of the image). If using twinx to generate the
        #        right y-axis (which is not advised), aspect='auto' helps
        #        solve the rendering bug.

        im = ax.imshow(X, cmap='tab20b', norm=norm, interpolation='none')
        ax.set_xlabel('WALS features', fontweight='bold')
        ax.set_ylabel('Languages', fontweight='bold')
        ax_r.set_ylabel('Languages', fontweight='bold')

        # Marking the given areas of interest (actually a rectangle, but
        # without showing its horizontal vertices).
        for area_of_interest in areas_of_interest:
            i_total = np.where(sorted_areas_values == area_of_interest)
            rect = patches.Rectangle((i_total[0][0] - 0.5, 0 - 0.5),
                                     len(i_total[0]),
                                     52,  # number of max langs or higher
                                     linewidth=1,
                                     edgecolor='blue',
                                     facecolor='none')
            ax.add_patch(rect)

        fig.set_size_inches(9, 3)  # for consistency in size
        plt.tight_layout()

        plt.show()

    def length_info(self) -> None:
        """
        Prints the average sentence length for each language.

        Note: all languages need to be processed first using process_counts().

        In:
          - n/a

        Out:
          - n/a
        """
        lengths_1 = {}  # iso639-1
        lengths_3 = {}  # iso639-3
        lang_codes = []

        for language in self.languages:
            iso_639_1, _, iso_639_3, _, _ = language
            cache_file = self.config['CACHE'] + iso_639_1 + '.p'
            lang_codes.append(iso_639_3)

            if not os.path.isfile(cache_file):
                raise Exception('Not all languages processed!')

            with open(cache_file, 'rb') as handle:
                # the total amount of characters and sentences kept
                # is already stored in a cachefile per language
                _, _, chars, kept = pickle.load(handle)

            lengths_1[iso_639_1] = chars/kept  # a bit redundant, but
            lengths_3[iso_639_3] = chars/kept  # never of a too big size

        for i in range(0, len(lang_codes), 6):
            print()

            for language in lang_codes[i:i+6]:
                print(f'{language}'.rjust(10), end=' ')

            print()

            for language in lang_codes[i:i+6]:
                print(f'{lengths_3[language]: .2f}'.rjust(10), end=' ')

            print()

    def align(self, sample_size: int,
              target_average: U[int, list]) -> T[dict, dict]:
        """
        Determines which samples to include from every file for every language
        to obtain a certain average and median length over the samples per
        language and equal size (i.e., number of samples) across languages.

        In:
          @sample_size:    number of samples to include for every language
          @target_average: desired average length of individual samples; if
                           int, all languages target the same average, if dict
                           keys denote language and values denote target avg

        Out:
          @selected_sentences: dict with ISO639-1 language codes as keys and
                               lists of tuples with (file_key, line) as values
          @file_keys:          index for file_keys per language; dict with
                               ISO639-1 language codes as keys and dicts as
                               values that in turn have values for file_key
                               (as given by @selected_sentences) as keys and
                               relative filepaths (str) as values
        """
        sentences = {}
        file_keys = {}

        selected_sentences = {}

        for language in self.languages:
            iso_639_1, _, _, _, _ = language
            cache_file = self.config['CACHE'] + iso_639_1 + '.p'

            if not os.path.isfile(cache_file):
                print('Not all languages processed!')
                return

            with open(cache_file, 'rb') as handle:
                length_counts, file_index, _, _ = pickle.load(handle)

            sentences[iso_639_1] = length_counts
            file_keys[iso_639_1] = file_index

        for language, lengths in sentences.items():
            # First, create a list of lengths and the line (and file) on
            # which the sentence with that length can be found.
            locations = [(length, line) for length, lines in lengths.items()
                         for line in lines]

            # Then, reduce bias (e.g., if samples stem only from 2020)
            random.shuffle(locations)

            if type(target_average) is dict:
                lang_target_avg = target_average[language]
            else:
                lang_target_avg = target_average

            # The distances of each sample from the target mean/median is used
            distances = [abs(val - lang_target_avg) for val, _ in locations]

            # Finally, sentences (denoted by line/file) are then sorted by
            # their distance from the target mean/median (the start of the
            # list now contains the 'best' samples)
            sorted_sentences = sorted(zip(locations, distances),
                                      key=lambda x: x[1])

            # Such that the list can simply be sliced to reach the sample size
            selected = [location
                        for location, _ in sorted_sentences[:sample_size]]

            selected_sentences[language] = selected

        return selected_sentences, file_keys

    def process_counts(self) -> None:
        """
        Reads all NewsCrawl files present in the dataset folder and counts the
        lengths of lines that pass the sanitisation tests stated in the class
        description. Resulting lengths are efficiently stored in cache (not
        samples itself).

        In:
          n/a

        Out:
          n/a
        """
        # Note: the 'regex' package is used instead of 're' as the latter does
        #       not support \p{L}-type unicode symbols.
        pattern_latin = regex.compile(self.config['REGEX_LATIN'])
        pattern_non_latin = regex.compile(self.config['REGEX_NON_LATIN'])
        pattern_russian = regex.compile(self.config['REGEX_RUSSIAN'])

        for language in self.languages:
            iso_639_1, justex_list, iso_639_3, _, latin_script = language
            cache_file = self.config['CACHE'] + iso_639_1 + '.p'

            if os.path.isfile(cache_file):
                continue  # already processed

            # Extra information to be stored
            length_counts = defaultdict(list)
            file_index = {}
            total_characters = 0

            # For printing statistics after processing
            min_length_removals = 0
            max_length_removals = 0
            regex_removals = 0
            justext_removals = 0
            kept = 0

            lang_path = self.config['FOLDER'] + iso_639_1 + "/*.gz"
            for k, file in enumerate(glob.glob(lang_path)):
                with gzip.open(file, 'rt', encoding='utf-8') as data:
                    file_index[k] = file
                    print('\r\n file ', file)

                    for line, sentence in enumerate(data):
                        if line >= self.config['MAX_PARSE_PER_FILE']:
                            print('Reached file parsing limit')
                            break

                        print(line, end="\r")

                        sentence = sentence.strip()  # remove trailing /n etc.

                        # Removing below/above min/max length
                        if len(sentence) < self.config['MIN_LENGTH']:
                            min_length_removals += 1
                            continue
                        elif len(sentence) > self.config['MAX_LENGTH']:
                            max_length_removals += 1
                            continue

                        # Removing RegEx matches
                        if latin_script and regex.search(pattern_latin,
                                                         sentence):
                            regex_removals += 1
                            continue
                        elif iso_639_1 == 'ru' \
                                and regex.search(pattern_russian, sentence):
                            regex_removals += 1
                            continue
                        elif iso_639_1 != 'ru' and not latin_script \
                                and regex.search(pattern_non_latin, sentence):
                            regex_removals += 1
                            continue

                        # Removing boilerplate/low-quality using JusText 3
                        if justex_list is not None:
                            stoplist = justext.get_stoplist(justex_list)
                            paragraphs = justext.justext(
                                                     bytes(sentence, 'utf-8'),
                                                     stoplist,
                                                     max_link_density=0)

                            # Occassionally, a sentence may be split into
                            # multiple 'paragraphs' by JusText 3, in case the
                            # 'sentence' is probably of low quality anyway and
                            # just skipped.
                            if len(paragraphs) > 1:
                                justext_removals += 1
                                continue

                            if paragraphs[0]['cfclass'] not in \
                                    self.config['JUSTEXT_ACCEPT']:
                                justext_removals += 1
                                continue

                        # Otherwise kept!
                        kept += 1

                        sentence_length = len(sentence)

                        total_characters += sentence_length
                        length_counts[sentence_length].append((k, line + 1))

            # Saving in cache to avoid long processing the next time
            with open(cache_file, 'wb') as handle:
                pickle.dump((length_counts,
                             file_index,
                             total_characters,
                             kept),
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

            print('\r\n [%s]:' % iso_639_1)

            print('Removed %d (< min length) + %d (> max) + %d (regex)'
                  ' + %d (justext) and kept %d (%.1f avg. chars)'
                  % (min_length_removals,
                     max_length_removals,
                     regex_removals,
                     justext_removals,
                     kept,
                     total_characters/kept if kept > 0 else 0))
