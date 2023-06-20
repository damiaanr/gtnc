from evaluation.bases.wals.src.Language import *
from evaluation.bases.wals.src.LanguageDB import *
from google.cloud import translate
import os
import numpy as np

# The following line may be needed to connect toa Google Cloud service account
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ""

# Some ISO639-3 codes are ambiguous within WALS (e.g., WALS 'gti'
# (German spoken in the Romanian city of Timișoara) and WALS 'ger'
# (the general entry for the German language) both have the ISO639-3
# code 'deu'. To avoid selecting the wrong language within WALS, while
# still maintaining the project-wide convention of using ISO639-3 codes
# to identify languages, a list of WALS codes that have preference is
# defined below.
wals_code_preferences = ['ger', 'grk', 'spa', 'ind', 'ita', 'kgz', 'dut',
                         'rom', 'tml', 'shn', 'mnd']


def get_feature_vectors_for_languages(languages: list = None,
                                      filter_areas: list = None,
                                      filter_params: list = None,
                                      force_overlap: bool = True) \
                                        -> T[dict, list, dict, LanguageDB]:
    """
    Generates numerical feature vectors for a given set of languages.
    These can be used as labels for a task that predicts WALS features
    from a text.

    Note: This function is a bit too long and there is a tiny bit of
          redundancy present, but splitting it up in separate functions
          would not gain much (especially as this folder will contain
          all kinds of functionality related to feature prediction).

    In:
      @languages:     list of ISO639-3 language codes for which to cal-
                      culate the feature vectors; if not set, feature
                      vectors will be generated for all languages for
                      which an ISO639-3 code is stored(/exists)
      @filter_areas:  if set, features within areas of which the IDs
                      are not included in this list are excluded; note:
                      the AreaID is the number (one-based) listed in
                      areas.csv in the Wals database
      @filter_params: if set, parameters of which the IDs are not in-
                      cluded in this list are excluded
      @force_overlap: if True, parameters which do not occur for all
                      provided languages are excluded (otherwise
                      their values in the feature vectors will be set
                      to -1)

    Out:
      @feature_vectors: dict with ISO639-3 language ids as keys and
                        NumPy vectors as its values (the 'labels')
      @parameter_list:  list of parameter codes that correspond to the
                        indices of the values in @feature vectors
      @num_classes:     dict with parameter/feature codes as keys and
                        the number of possible values/classes as values
      @lang_db:         languageDB obj created during the call to this
                        function (can be used for analysis, such as in
                        explain_differences_between_vectors(); this is
                        more efficient than creating a new instant
    """
    wals_data_path = "evaluation/bases/wals/data/"
    lang_db = LanguageDB(data_path=wals_data_path)
    chapters_data = LanguageDB.get_chapters_fields(data_path=wals_data_path)

    if languages is None:
        languages = list(lang_db.languages_iso639_3.keys())  # ISO 639-3
        # wals_codes = lang_db.languages_wals

    wals_codes = []

    for language in languages:
        wals_code = lang_db.get_wals_code_by_iso639_3(
                                            language,
                                            preferences=wals_code_preferences)
        wals_codes.append(wals_code)

    lang_db.populate_characteristics(explicit=False, wals_codes=wals_codes,
                                     partial=True)
    language_objs = {wals_code: Language(wals_code, lang_db)
                     for wals_code in wals_codes}
    chapters_info = LanguageDB.get_chapters_fields(data_path=wals_data_path)

    feature_vectors = {}

    if force_overlap:  # extra loop to find overlap
        overlapping_params = None

        for i, iso639_3 in enumerate(languages):
            wals_code = wals_codes[i]
            params = language_objs[wals_code].get_characteristic().keys()

            if overlapping_params is None:
                overlapping_params = set(params)
            else:
                overlapping_params = overlapping_params & set(params)

    # determining the parameter indices within vector; if overlapping
    # parameters are not forced, this list is built while looping
    parameter_list = []

    if force_overlap:
        for param in overlapping_params:
            if filter_params is not None:
                if param not in filter_params:
                    continue

            if filter_areas is not None:
                chapter_id = lang_db.params_chapter_mapping[param]
                if int(chapters_data[chapter_id][3]) not in filter_areas:
                    continue

            parameter_list.append(param)

    # now the actual construction of the feature vectors
    for i, iso639_3 in enumerate(languages):
        wals_code = wals_codes[i]
        item_params = language_objs[wals_code].get_characteristic()

        feature_vector = []

        for listed_param in parameter_list:
            if listed_param in item_params.keys():
                # all non-explicit item parameter values are of the
                # format: <parameter_code>-<value_auto_increment_id>
                # (base-one; so one is subtracted)
                item_param_parts = item_params[listed_param].split('-')
                item_param_class_id = int(item_param_parts[1]) - 1
                feature_vector.append(item_param_class_id)
                del item_params[listed_param]
            else:
                feature_vector.append(-1)

        if not force_overlap:  # all occurring parameters are included
            for param_key, param_value in item_params.items():
                # due to the del operation in the previous loop, this
                # loop only walks over newly encountered parameters

                if filter_params is not None:
                    if param_key not in filter_params:
                        continue

                if filter_areas is not None:
                    chapter_id = lang_db.params_chapter_mapping[param_key]
                    if int(chapters_data[chapter_id][3]) not in filter_areas:
                        continue

                parameter_list.append(param_key)
                class_id = int(param_value.split('-')[1]) - 1
                feature_vector.append(class_id)

                for previous_lang in feature_vectors.keys():
                    # setting value to -1 for previously walked langs
                    # (that logically do not have the param set)
                    feature_vectors[previous_lang].append(-1)

        feature_vectors[iso639_3] = feature_vector

    # final loop to convert all lists to numpy arrays
    for lang in feature_vectors.keys():
        feature_vectors[lang] = np.array(feature_vectors[lang])

    # we will return only the number of classes for included parameters
    num_classes = {}
    for parameter, parameter_n_classes in lang_db.params_num_classes.items():
        if parameter in parameter_list:
            num_classes[parameter] = parameter_n_classes

    return feature_vectors, parameter_list, num_classes, lang_db


def translate_text(texts: list, src: str, trg: str, pid: str) -> list:
    """
    Translates a list of sentences using the Google Cloud Translation API.

    Note: This function is derived from the code posted in Google's API docs
          https://cloud.google.com/python/docs/reference (Translate V3).

    In:
      @text: list of sentences in the source language
      @src:  ISO639-1 code of the source language
      @trg:  ISO639-1 code of the target language
      @pid:  Google Cloud Project ID (see Google's API docs)

    Out:
      @translations: list of translated sentences
    """
    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{pid}/locations/{location}"
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": texts,
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": src,
            "target_language_code": trg,
        }
    )

    translations = []

    for translation in response.translations:
        translations.append(translation.translated_text)

    return translations


def detect_language(text: str, pid: str) -> list:
    """
    Detects the language of a text using the Google Cloud Translation API.

    Note: This function is derived from the code posted in Google's API docs
          https://cloud.google.com/python/docs/reference (Translate V3).

    In:
      @text: single string (one at a time)
      @pid: Google Cloud Project ID (see Google's API docs)

    Out:
      @languages: list of tuples that contains [0] ISO639-1 string of language
                  and [1] float representing the confidence of detection (0-1)
    """
    client = translate.TranslationServiceClient()
    location = "global"
    parent = f"projects/{pid}/locations/{location}"

    request = translate.DetectLanguageRequest(
        content=text,
        parent=parent,
    )

    response = client.detect_language(request=request)

    languages = []

    for lang in response.languages:
        languages.append((lang.language_code, lang.confidence))

    return languages
