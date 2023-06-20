from functions import translate_text
import os
import glob

# Fill in the fields below (or modify code to accommodate other use cases)
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./<filename>.json"
pid = ""
dataset_folder = "output/"
chunk_size = 150  # (max 1024 sentences and 30720 characters)


# Translating the dataset has actually become a relatively simple task now,
# with no need to store files in cache, etc., as we write during translating.
for source_file in glob.glob(dataset_folder + "source/*.src"):
    lang = os.path.basename(source_file)[0:2]

    trg_file = "output/translated/" + lang + ".trg"

    if not os.path.isfile(trg_file):
        with open(source_file, encoding='utf-8') as f:
            samples = f.read().splitlines()

        fw = open(trg_file, "w", encoding="utf-8")

        for i in range(0, len(samples), chunk_size):
            chunk = samples[i:i+chunk_size]
            translations = translate_text(chunk, lang, 'en', pid=pid)

            for translation in translations:
                fw.write(translation+"\n")
                fw.flush()  # immediately write to avoid losing data
