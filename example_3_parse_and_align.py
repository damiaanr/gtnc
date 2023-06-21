from SourceDataset import SourceDataset
from fifty_languages import languages_to_include
from collections import defaultdict
from functions import translate_text
import os
import gzip
import pickle
import numpy as np

# Fill in the fields below (or modify code to accommodate other use cases)

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./<filename>.json"
pid = ""


# Code starts here (but contains guiding steps throughout the scripts)
ds = SourceDataset(languages_to_include)

# It would be advisable to first obtain a character-to-character ratio:
# a 100-length sentence in one language does generally not translate into
# a 100-length English sentence. A dataset of same-length English sentences
# should therefore take these ratios into account. Here, we take a 100-length
# sample from every language to calculate these ratios. Just in case, these
# are also stored in cache.
cache_ratio_sample_s = ds.config['CACHE'] + 'ratio_100s.p'

if os.path.isfile(cache_ratio_sample_s):
    with open(cache_ratio_sample_s, 'rb') as handle:
        samples = pickle.load(handle)
else:
    sents, file_keys = ds.align(100, 100)

    samples = defaultdict(list)

    for lang, lang_sents in sents.items():
        files_lines = defaultdict(list)

        for sent_length, (sent_file_key, sent_line) in lang_sents:
            sent_file = file_keys[lang][sent_file_key]

            files_lines[sent_file].append(sent_line)

        for file, lines in files_lines.items():
            data = gzip.open(file, 'rt', encoding='utf-8').readlines()

            for line in lines:
                samples[lang].append(data[line-1].rstrip())

    with open(cache_ratio_sample_s, 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Now, we translate these obtained sentences; the translations we also store
# in cache, just in case.
cache_ratio_sample_t = ds.config['CACHE'] + 'ratio_100t.p'
if os.path.isfile(cache_ratio_sample_t):
    with open(cache_ratio_sample_t, 'rb') as handle:
        samples_translated = pickle.load(handle)
else:
    samples_translated = defaultdict(list)

    for lang, texts in samples.items():
        translations = translate_text(texts, lang, 'en', pid=pid)
        samples_translated[lang].extend(translations)

    with open(cache_ratio_sample_t, 'wb') as handle:
        pickle.dump(samples_translated,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

# Now, we actually calculate the ratios and store these in a dict
lengths = defaultdict(list)
ratios = {}

for lang, translations in samples_translated.items():
    for translation in translations:
        lengths[lang].append(len(translation))

    lengths_array = np.array(lengths[lang])

    mean = np.mean(lengths_array)
    zero_mean = lengths_array - mean

    ratio_to_en = mean/100
    ratio_from_en = 100/mean

    ratios[lang] = ratio_from_en

# Finally, we can generate the sample and find the source collection!
# Again, we store these in cache.
num_samples = 7500
target_length = 125

cache_collection = ds.config['CACHE'] + 'collection_' + str(num_samples) \
                   + '_' + str(target_length) + '.p'

if os.path.isfile(cache_collection):
    with open(cache_collection, 'rb') as handle:
        samples = pickle.load(handle)
else:
    size_per_language = {lang: round(target_length*ratio)
                         for lang, ratio in ratios.items() if lang != 'en'}

    sents, file_keys = ds.align(num_samples, size_per_language)

    samples = defaultdict(list)

    for lang, lang_sents in sents.items():
        files_lines = defaultdict(list)

        for sent_length, (sent_file_key, sent_line) in lang_sents:
            sent_file = file_keys[lang][sent_file_key]

            files_lines[sent_file].append(sent_line)

        for file, lines in files_lines.items():
            data = gzip.open(file, 'rt', encoding='utf-8').readlines()

            for line in lines:
                samples[lang].append(data[line-1].rstrip())

    with open(cache_collection, 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Which we write into an output directory here (one sentence per line)
total_length = 0
for lang, sentences in samples.items():
    for sentence in sentences:
        total_length += len(sentence)

    f = open("output/source/" + lang + ".src", "w", encoding="utf-8")
    f.writelines([sentence + '\n' for sentence in sentences])
    f.close()

print('Total length: %d' % total_length)
