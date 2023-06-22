# Google Translations from NewsCrawl (GTNC)
This repository contains (1) a *many-to-one* dataset of original texts in 50 source languages and their corresponding translations into English using a recent version of Google Translate, and (2) the Python code that was used to generate it.

## 1. About the dataset: use the data we generated!
The dataset can be found in the `output` folder in this directory. The dataset contains two folders:

1. `source`: Source text in original languages (contained in NewsCrawl)
2. `translated`: Corresponding English translations of source texts

Every folder contains either a `.src` or a `.trg` file for every language containing either source or translated (target) samples in line-by-line manner. If available, corresponding fluency scores are stored in `.scores` files (see below under *Cleaning steps*). Additionally, line-by-line Google Language Detection analyses for the first 750 samples are contained in `.detect` files (the ISO639-1 code of the detected language is followed by the confidence value).

### Details
The dataset contains **7,500** samples of **~125 characters**-long translations in English (and their source texts) for **50** languages\* with the following ISO639-1 codes: `am`, `ar`, `bg`, `bn`, `cs`, `de`, `el`, `en`, `es`, `et`, `fa`, `fi`, `fr`, `gu`, `ha`, `hi`, `hr`, `hu`, `id`, `ig`, `is`, `it`, `ja`, `kn`, `ko`, `ky`, `lt`, `lv`, `mk`, `ml`, `mr`, `nl`, `om`, `or`, `pa`, `pl`, `ps`, `pt`, `ro`, `ru`, `sn`, `sw`, `ta`, `te`, `ti`, `tl`, `tr`, `uk`, `yo`, and `zh`

Original data was taken from the 2023 version of NewsCrawl (see [Findings of the 2022 Conference on Machine Translation (WMT22)](https://aclanthology.org/2022.wmt-1.1) (Kocmi et al., WMT 2022) and [the data directory](https://data.statmt.org/news-crawl/) itself). This data consists of individual sentences that were scraped from a wide range of online news outlets across different locales. The data were translated using the 19/06/2023-20/06/2023 (DD/MM/YYYY) v3 version of Google's Translation API using default edition settings. Several NewsCrawl-languages were excluded (9 out of 59) as they either contained too less data (`tig`, `bm`), were not supported by Google Translate (`nr`), were found to be too noisy (`rw`, `so`), were close to other languages and therefore not contributing to the diversity of the dataset (`bs`, `sr`, `kk`), or the former in combination with containing too few [WALS](https://wals.info/) features, making them less usable (`af`). The 50 remaining languages are of high typological diversity. 

*\* `en` did not need to be translated*

### Cleaning steps 
The following steps were performed on the data to ensure that individual samples are of high quality (*i.e.*, fluency):

1. Relevant steps already taken by NewsCrawl
    - Across all languages, data is ‘parallel by year’: all sentences are sampled from news articles that appeared in 2020, 2021, or 2022.
	- Data was cleaned of duplicates.
2. Extra steps taken by us before translating
    - Samples below 30 characters in length and above 400 were deleted.
	- Samples that do not appear to be ‘standard sentences’ were deleted in greedy manner (*e.g.*, samples in a non-latin script language that contain any of `A-Za-z`; see comments in the code for the full specifics and the regular expressions).
	- Samples deemed `short` or `bad` by the [JusText 3](https://github.com/miso-belica/jusText) module were deleted. Not supported for `am`, `sn`, `ti`, `om`, `pa`, `ha`, `ps`, `or`, and `ja` (note: the samples for these languages might therefore be of lesser quality).
3. Steps taken after translating
    - Original (source) and translated (English) samples were scored by Monocleaner (see [Prompsit’s submission to WMT 2018 Parallel Corpus Filtering shared task](https://aclanthology.org/W18-6488) (Sánchez-Cartagena et al., WMT 2018) and the [module's code](https://github.com/bitextor/monocleaner)) to indicate fluency. Source scores not supported for `am`, `ar`, `bn`, `cs`, `fa`, `gu`, `ha`, `hi`, `id`, `ig`, `ja`, `kn`, `ko`, `ky`, `ml`, `mr`, `om`, `or`, `pa`, `ps`, `ru`, `sn`, `sw`, `ta`, `te`, `ti`, `tl`, `yo`, and `zh`.
	- Google's **Language Detection** service was subsequently called on 10% of all translated samples to quantify potential noise stemming from NewsCrawl's source sets containing sentences from other languages. The API outputs a single predicted language and a confidence value (between 0 and 1) corresponding to the prediction (see under *Language Detection* for an overview per language).

### Length alignment
Mainly due to API credit constraints, only a small subset of all sentences present in NewsCrawl were translated.

To provide models with data that is *as parallel as possible* (which is an inherent problem for many-to-one datasets), in addition to thorough cleaning (see previous heading), for every source language, samples of specific lengths were drawn in order to preserve a fixed mean and median length (set to 125 characters) in English translations. The samples drawn to achieve this mean/median length are re-shuffled to avoid most samples being drawn from earlier years for larger languages.

To achieve an approximately fixed average character length of 125 in English translations across all languages, 100 samples of 100 characters were translated for every language to first obtain a frequentist character-to-character ratio for every language translation pair. These ratios determined the varying lengths in the source files and are shown below (direction is from source to English):

`am: 1.56`, `ar: 1.30`, `bg: 1.04`, `bn: 1.11`, `cs: 1.13`, `de: 0.94`, `el: 0.92`, `en: 1.00`, `es: 0.95`, `et: 1.11`, `fa: 1.19`, `fi: 1.05`, `fr: 0.92`, `gu: 1.08`, `ha: 1.00`, `hi: 1.17`, `hr: 1.11`, `hu: 1.07`, `id: 1.00`, `ig: 1.09`, `is: 1.03`, `it: 0.97`, `ja: 2.37`, `kn: 1.02`, `ko: 2.29`, `ky: 1.02`, `lt: 1.08`, `lv: 1.10`, `mk: 1.00`, `ml: 0.88`, `mr: 1.03`, `nl: 0.95`, `om: 0.82`, `or: 1.07`, `pa: 1.01`, `pl: 1.03`, `ps: 1.15`, `pt: 1.01`, `ro: 0.97`, `ru: 1.07`, `sn: 1.01`, `sw: 1.00`, `ta: 0.89`, `te: 1.06`, `ti: 1.31`, `tl: 0.91`, `tr: 1.05`, `uk: 1.11`, `yo: 1.08`, and `zh: 4.22`.

Ultimately, **42,667,664** source characters (1/ratio\*125\*7500 per language) were translated into **46,460,290** English characters (a mean of ~126.42 characters per sentence; not including `en`). 

### Language detection
To provide a small additional overview of the quality of the dataset in terms of noise from other languages being present in the individual source files, 10% of all source sentences (*i.e.*, 750 samples per language) were run trough Google's language detection API. The scores below represent the fractions of detected languages, weighted by the corresponding confidence values.

| lang   | analysis                | lang   | analysis                            | lang   | analysis               | lang   | analysis                            | lang   | analysis                                         |
|--------|-------------------------|--------|-------------------------------------|--------|------------------------|--------|-------------------------------------|--------|--------------------------------------------------|
| `am`   | am: 0.999<br>ti: 0.001  | `ar`   | ar: 1.000                           | `bg`   | bg: 1.000              | `bn`   | bn: 1.000                           | `cs`   | cs: 0.998<br>sk: 0.001                           |
| `de`   | de: 0.995               | `el`   | el: 1.000                           | `en`   | -                      | `es`   | es: 0.962<br>ca: 0.006<br>gn: 0.001 | `et`   | et: 1.000                                        |
| `fa`   | fa: 1.000               | `fi`   | fi: 1.000                           | `fr`   | fr: 0.958              | `gu`   | gu: 1.000                           | `ha`   | ha: 0.995<br>es: 0.001<br>gn: 0.001<br>en: 0.001 |
| `hi`   | hi: 0.998               | `hr`   | hr: 0.544<br>bs: 0.167<br>en: 0.004 | `hu`   | hu: 0.999<br>en: 0.001 | `id`   | id: 0.943<br>ms: 0.009<br>su: 0.001 | `ig`   | ig: 0.999<br>en: 0.001                           |
| `is`   | is: 0.999<br>en: 0.001  | `it`   | it: 0.998                           | `ja`   | ja: 1.000              | `kn`   | kn: 1.000                           | `ko`   | ko: 1.000                                        |
| `ky`   | ky: 1.000               | `lt`   | lt: 1.000                           | `lv`   | lv: 0.999<br>en: 0.001 | `mk`   | mk: 1.000                           | `ml`   | ml: 1.000                                        |
| `mr`   | mr: 0.998               | `nl`   | nl: 0.998<br>fy: 0.001              | `om`   | om: 0.999<br>en: 0.001 | `or`   | or: 1.000                           | `pa`   | pa: 1.000                                        |
| `pl`   | pl: 0.999<br>en: 0.001  | `ps`   | ps: 0.899<br>fa: 0.101              | `pt`   | pt: 0.996<br>gl: 0.003 | `ro`   | ro: 1.000                           | `ru`   | ru: 0.993                                        |
| `sn`   | sn: 0.999<br>zu: 0.001  | `sw`   | sw: 1.000                           | `ta`   | ta: 1.000              | `te`   | te: 1.000                           | `ti`   | ti: 1.000                                        |
| `tl`   | tl: 0.985<br>ilo: 0.001 | `tr`   | tr: 0.999                           | `uk`   | uk: 1.000              | `yo`   | yo: 0.975<br>en: 0.001              | `zh`   | zh-CN: 0.989<br>zh-TW: 0.011                     |

A single listed language with a score of `1.000` means that every sample was detected as that language with full confidence (a perfect score). A single listed language with a score below perfect indicates that no other language was detected in any of the samples, but that Google was not always completely confident in its detection. In the latter case, considering that the authors of NewsCrawl deliberately scraped sentences of particular languages, the source set is highly likely to contain solely sentences of the intended source language. For other source languages, the language of some of the samples was detected to be a typologically similar one to the intended language that is often subject to *language-or-dialect* debates (*e.g.*, `cs` and `sk`, or `hr` and `bs`). However, in the case of `so` (Somali) and `rw` (Kinyarwanda), the source set turned out to contain a high number of English sentences and was therefore deemed less useful and consequently deleted from the dataset (including in the above specifications). While the original NewsCrawl dataset contains more noisy source sets (such as a large number of Ukrainian sentences being present in the Russian source set), these seem to have been effectively filtered using the provided cleaning steps (see above).

## 2. About the script: create a better version of GTNC!
We encourage everyone to create newer versions of the dataset; either with a larger amount of samples (requiring more Cloud Translation API Credit) or using more recent versions of Google Translate or NewsCrawl(-like datasets). The code in this repository can be used out-of-the-box to create your own many-languages-to-one dataset. All code is thoroughly documented, type-annotated, and PEP8-compliant, and should be straightforward to understand. A `requirements.txt` file is provided to help you set up a working environment. In addition to the code, a couple of tips are shown below. Note that the `evaluation` folder contains code from a stand-alone accuracy evaluation program based on WALS from which only part of the functionality is used to create the dataset.

### Connecting with Google's API services
Although Google provides documentation on [Authenticating and ‘how to start’](https://developers.google.com/people/quickstart/python) and on [how to use it's Python library to translate text](https://cloud.google.com/python/docs/reference/translate/latest/google.cloud.translate_v3.services.translation_service.TranslationServiceClient#google_cloud_translate_v3_services_translation_service_TranslationServiceClient_translate_text), it may be handy to be aware of the following:

- After creating a project, setting up a billing account, and enabling the *Cloud Translation API*, it is adviced to create a *Service Account* (under *Credentials*, *Create Credentials*) to connect to the API hassle-free.
- Create a JSON key and save locally (*e.g.*, in the folder containg the source files). Then, after `import os`, authentication may be performed by inserting `os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./<filename>.json"` in the top lines of any of the given example scripts that use translation services.
- Remember to always set usage quota in case anything goes wrong! **Per the default settings, running all examples will cost approximately $1,000 in API credits as of 20/06/23.**

### Loading NewsCrawl
The code works with all NewsCrawl or ‘NewsCrawl-like’ datasets. This simply means that:

- A separate folder must be created for the dataset, *e.g.*, *news-crawl* (if diverging from the default name, remember to initialise `SourceDataset()` with the `config={'FOLDER': '<yourfoldername>'}` parameter)
- This folder must contain a new folder for every language contained in the dataset, represented by ISO639-1 code, *e.g.*, `nl` or `bg`
- These language-folders must contain `.gz` files that contain one sample (*i.e.*, a sentence) per line

### Examples
Five separate example scripts are provided to generate newer versions of GTNC. When proceeding in sequence, these samples will together produce a new dataset from scratch. It is recommended to read and modify each file accordingly before running. Most of the steps involve creating and loading cache files to prevent data from being lost. (Create a `cache` folder before you start!)

1. This script first prints a language similarity metric for every language, calculated as the average overlap in [WALS](https://wals.info/) features (as a fraction) between the single language and all other languages. A low score indicates that the language adds a high amount of diversity to the dataset. For every language, the number of filled-in features is shown on the bottom. Afterwards, the script plots a matrix indicating the diversity of the languages contained in the dataset. Rows indicate languages, columns indicate WALS features, every square (element) is colored differently based on the class index corresponding to a certain language for a certain feature.
2. This script parses the relevant NewsCrawl files (if not stored in cache), and displays the average length of all samples, per language.
3. This script parses the relevant NewsCrawl files (if not stored in cache), calculates the character-to-character ratios (as described under *Length alignment*) and uses these to sample an equal subset of sentences for every language while preserving a target mean/median length within the translations. Source files are created.
4. This script reads the source files (created while running example 3) and performs language detection on part of the samples for each language. Language detection annotation files are created.
5. This script reads the source files (created while running example 3) and translates these into English. Target files are created.

*For calculating Monocleaner scores, [Bitextor's repository](https://github.com/bitextor/monocleaner) already contains clear instructions.*

## Contact with authors
Please contact us if you have any questions... (details should follow below)
