from __future__ import annotations
from typing import Union as U


class Language:
    """
    The Language class holds is instantiated for every language
    documented within WALS and provides access to general information
    as well as language-specific characteristics. Each instance can
    also store similarity scores relative to other language instances.
    """
    def __init__(self, wals_code: str, language_db: LanguageDB):
        """
        Init (see class description above).

        In:
          @wals_code:   language code within WALS
          @language_db: LanguageDB()-object

        Out:
          n/a
        """
        self.language_db = language_db  # simple pointer

        if not self.language_db.verify_wals_code(wals_code):
            raise Exception("Language code does not exist within WALS")

        self.wals_code = wals_code
        self.data = language_db.languages_wals[wals_code]
        self.similarity_scores = {}  # stored for each pair in each instance

    def get_characteristic(self, key: str = None) -> U[str, dict]:
        """
        Returns language-specific value for a structural parameter.
        Note that the key may be either a code or an explicit name
        based on the settings profile of populating in LanguageDB().
        If @key is None, all language-specific features for the current
        language are returned.

        In:
          @key: parameter name/key (structural indicator); if None,
                all characteristics are returned

        Out:
          @value: either language-specific value/characteristic (if
                  @key set) or a dict with as keys either parameter
                  name or key of feature (depending on settings)
        """
        if not self.language_db.characteristics_populated:
            self.language_db.populate_characteristics([self.wals_code])

        out = self.language_db.languages_wals[self.wals_code]
        out = out['Characteristics']

        if key is not None:
            out = out[key]

        return out

    def get_mutual_characteristics(self, contrast: Language) -> list:
        """
        Returns intersection of set parameter values for a contrasting
        language.

        In:
          @contrast: another instance of Language()

        Out:
          @keys: list of parameter keys (note: may be codes or names;
                 see documentation for get_characteristic())
        """
        if not self.language_db.characteristics_populated:
            self.language_db.populate_characteristics([self.wals_code,
                                                       contrast.wals_code])

        # use same DB to prevent explicit/non-explicit intersections
        c1 = self.language_db.languages_wals[self.wals_code]
        c2 = self.language_db.languages_wals[contrast.wals_code]

        c1 = c1['Characteristics'].keys()
        c2 = c2['Characteristics'].keys()

        return list(c1 & c2)
