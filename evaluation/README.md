# Evaluation
This small framework provides an easy means for calculating a 'correctness score' for a language prediction using different kinds of approaches.

*Note: `requirements.txt` file is missing as the code only uses Python's standard libraries*

## Usage
This is not a command-line program. Example usage is shown in `example.py` for pairs of languages listed in `languages.txt`.

## Bases
Currently only **one** approach is supported.

*Note: every base (or approach) will yield different scores for each language pair and may have hard-coded parameter settings which influence the scores.*

### WalsEvaluator
The `WalsEvaluator` evaluates the correctness of a prediction of a language by comparing the overlap between the values for WALS-features of the predicted language with those of the ground-truth language. The evaluator reports a score between `0` and `1` for language pairs for which both languages occur in the top-`0.05%` similar languages of each other *and* have a minimum of `30` WALS features set. Original WALS data is included in the `data` folder in `bases/wals`.