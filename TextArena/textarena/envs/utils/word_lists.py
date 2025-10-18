"""Utils for Word Lists, common to language games. (by Dilan Hillier)"""

import importlib.resources
import re
from collections import defaultdict

# Load NLTK word list
import nltk
from nltk.corpus import words



def _parse_affix_rules(file_content: list[str]):
    # Regular expression pattern to match the affix rules
    pattern = re.compile(r"^(PFX|SFX)\s+(\w+)\s+(Y|N)\s+(\S+)\s+([\'|\w]+)\s+(\S+)$")

    # Lists to store the parsed rules
    prefixes = defaultdict(list)
    suffixes = defaultdict(list)

    # Iterate over each line in the file content
    for line in file_content:
        match = pattern.match(line.strip())
        if match:
            affix_type, flag, can_merge, strip_char, affix, condition = match.groups()
            rule = {
                "flag": flag,
                "can_merge": can_merge == "Y",
                "strip_char": strip_char if strip_char != "0" else None,
                "affix": affix,
                "condition": condition if condition != "." else None,
            }
            if affix_type == "PFX":
                prefixes[flag].append(rule)
            else:
                suffixes[flag].append(rule)

    return prefixes, suffixes


def _parse_condition(condition):
    """Parse the condition string in the affix rules."""
    if not condition:
        return None, None
    pattern = re.compile(r"(\[\^\S+\])?(\S+)?")
    match = pattern.match(condition)
    if match:
        exclude, include = match.groups()
        return exclude, include
    return None, None


def _split(line: str) -> tuple[str, str]:
    """Split a line from a .dic file into a word and its flags."""
    if "/" in line:
        return tuple(line.split("/"))
    return line.strip(), ""


class EnglishDictionary:
    """Dictionary Utils for English words."""

    def __init__(
        self, keep_proper_nouns=False, include_nltk=True, keep_non_alpha=False
    ):
        """Initialize the dictionary."""
        nltk.download("words")
        
        self.include_nltk = include_nltk
        self.keep_non_alpha = keep_non_alpha
        self.keep_proper_nouns = keep_proper_nouns
        self.uk_words = self._load_dic("en_GB.dic", "en.aff")
        # self.uk_words = self.expand(self.uk_words, self.uk_affs)
        self.us_words = self._load_dic("en_US.dic", "en.aff")
        # self.us_words = self.expand(self.us_words, self.us_affs)
        self.nltk_words = self._load_nltk(basic=False) if include_nltk else set()
        self.nltk_basic_words = self._load_nltk(basic=True) if include_nltk else set()

    def _filter(
        self,
        word_set: set[str],
    ) -> set[str]:
        filtered = set()
        for word in word_set:
            if word[0].isalpha() or self.keep_non_alpha:
                if word[0].islower() or self.keep_proper_nouns:
                    filtered.add(word)
        return filtered

    def _load_dic(
        self,
        filename: str,
        acc_filename: str,
    ) -> set[str]:
        """Load words from a .dic file inside the package's data folder."""
        with (
            importlib.resources.files("textarena.envs.utils.data")
            .joinpath(filename)
            .open("r") as f
        ):
            lines = f.readlines()[1:]  # Skip first line (word count)
        with (
            importlib.resources.files("textarena.envs.utils.data")
            .joinpath(acc_filename)
            .open("r") as f
        ):
            acc_lines = f.readlines()
        prefixes, suffixes = _parse_affix_rules(acc_lines)
        entries = set(_split(line) for line in lines)
        filtered = self._filter(entries)
        all_words = set()
        # first we add the base words
        for word, flag in filtered:
            all_words.add(word)
        prefixed_words = set()
        for word, flags in filtered:
            for flag in list(flags):
                if flag in prefixes:
                    # apply prefix
                    for rule in prefixes[flag]:
                        # check if the condition is met
                        exclude, include = _parse_condition(rule["condition"])
                        if exclude and any(
                            [word.endswith(exclude_char) for exclude_char in exclude]
                        ):
                            continue
                        if include and not word.endswith(include):
                            continue
                        if rule["strip_char"]:
                            if word.startswith(rule["strip_char"]):
                                new_word = (
                                    word[len(rule["strip_char"]) :] + rule["affix"]
                                )
                                all_words.add(new_word)
                                prefixed_words.add((new_word, flags))
                        else:
                            new_word = rule["affix"] + word
                            all_words.add(new_word)
                            prefixed_words.add(
                                (new_word, flags)
                            )  # may still need to apply suffixes
        for word, flags in filtered:
            for flag in list(flags):
                if flag in suffixes:
                    # apply suffix
                    for rule in suffixes[flag]:
                        # check if the condition is met
                        exclude, include = _parse_condition(rule["condition"])
                        if exclude and any(
                            [word.endswith(exclude_char) for exclude_char in exclude]
                        ):
                            continue
                        if include and not word.endswith(include):
                            continue
                        if rule["strip_char"]:
                            if word.endswith(rule["strip_char"]):
                                new_word = (
                                    word[: -len(rule["strip_char"])] + rule["affix"]
                                )
                                all_words.add(new_word)
                        else:
                            new_word = word + rule["affix"]
                            all_words.add(new_word)
        # finally we do merged prefixes and suffixes
        for word, flags in prefixed_words:
            for flag in list(flags):
                if flag in suffixes:
                    # apply suffix
                    for rule in suffixes[flag]:
                        # continue if flag is not mergeable
                        if not rule["can_merge"]:
                            continue
                        # check if the condition is met
                        exclude, include = _parse_condition(rule["condition"])
                        if exclude and word.endswith(exclude):
                            continue
                        if include and not word.endswith(include):
                            continue
                        if rule["strip_char"]:
                            if word.endswith(rule["strip_char"]):
                                new_word = (
                                    word[: -len(rule["strip_char"])] + rule["affix"]
                                )
                                all_words.add(new_word)
                        else:
                            new_word = word + rule["affix"]
                            all_words.add(new_word)
        all_words = self._filter(all_words)
        return all_words

    def _load_nltk(self, basic: bool) -> set[str]:
        nltk_words = set(words.words("en-basic") if basic else words.words("en"))
        return self._filter(nltk_words)

    def is_english_word(self, word: str) -> bool:
        """Check if a word is in the UK and/or US and/or NLTK English dictionary."""
        word = word.lower()
        return word in self.uk_words or word in self.us_words or word in self.nltk_words

    def get_all_words(self) -> set[str]:
        """Get all words in the dictionary as a set"""
        return self.uk_words | self.us_words | self.nltk_words

    def get_basic_words(self) -> set[str]:
        """Get all words in the basic NLTK dictionary as a set"""
        return self.nltk_basic_words
