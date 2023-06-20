from functions import detect_language
from collections import defaultdict
import numpy as np
import pickle
import os
import glob

# Fill in the fields below (or modify code to accommodate other use cases)

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./<filename>.json"
pid = ""
dataset_folder = "output/"
cache_folder = "cache/"
samples_per_lang = 750  # 10% of 7,500 set in example 3


# Code starts here (but contains guiding steps throughout the scripts)
language_detections = {}

# First, we simply perform language detection and write the results into the
# dataset folder. The raw results are stored in cache per language (to avoid
# losing all data when something goes wrong).
for source_file in glob.glob(dataset_folder + "source/*.src"):
    lang = os.path.basename(source_file)[0:2]
    cache_detection = cache_folder + 'detection_' + lang + '.p'

    if os.path.isfile(cache_detection):
        with open(cache_detection, 'rb') as handle:
            detections = pickle.load(handle)
    else:
        with open(source_file, encoding='utf-8') as f:
            samples = f.read().splitlines()

        fw = open("output/source/" + lang + ".detect", "w", encoding="utf-8")

        detections = []

        for sample in samples[0:samples_per_lang]:
            result = detect_language(sample, pid)
            detections.append(result)

            fw.write(",".join([d[0] + " " + str(d[1]) for d in result])+"\n")
            fw.flush()  # immediately write to avoid losing data

        fw.close()

        with open(cache_detection, 'wb') as handle:
            pickle.dump(detections, handle, protocol=pickle.HIGHEST_PROTOCOL)

    language_detections[lang] = detections

# Now, we can calculate a global detection score per language (weighted by
# confidence)
language_total_scores = defaultdict(lambda: defaultdict(list))

for lang, detections in language_detections.items():
    for detection in detections:
        for (dlang, confidence) in detection:
            language_total_scores[lang][dlang].append(confidence)

for lang, detections in language_total_scores.items():
    print('[%s]' % lang,
          ', '.join(['%s: %.3f' % (lang, np.sum(confs)/samples_per_lang)
                     for lang, confs in detections.items()]))
