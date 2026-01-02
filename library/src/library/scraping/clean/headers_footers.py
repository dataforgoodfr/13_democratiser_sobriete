import os
import re
import string

import numpy as np
import pandas as pd


class TrieNode:
    def __init__(self):
        self.children = {}
        self.count = 0


class PrefixFinder:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, text):
        node = self.root
        for word in text:
            if word not in node.children:
                node.children[word] = TrieNode()
            node = node.children[word]
            node.count += 1

    def find_longest_shared_prefix(self, text):
        node = self.root
        longest_prefix = ""
        current_prefix = []
        current_count = 0

        for word in text:
            node = node.children[word]
            # If this character is shared by more than 1 page
            if node.count > 1:
                current_prefix.append(word)
                longest_prefix = " ".join(current_prefix)
                current_count = node.count
            else:
                break

        return longest_prefix, current_count


def find_common_prefixes(pages, reverse=False, max_words=50):
    trie = PrefixFinder()

    cleaned_pages = []
    for page in pages:
        cp = normalize_whitespaces(page.replace("\n", " ")).split()
        if reverse:
            cp = cp[::-1]
        cleaned_pages.append(cp[:max_words])

    # First pass: Build the Trie with all pages
    for page in cleaned_pages:
        trie.insert(page)

    # Second pass: Find the longest shared prefix for each page
    results = []
    prefixes = set()
    for page in cleaned_pages:
        prefix, count = trie.find_longest_shared_prefix(page)

        if not reverse:
            # many pages start with table or figure, but they're not part of the header
            if prefix and prefix.split()[-1].lower() in ["table", "figure"]:
                prefix = " ".join(prefix.split()[:-1])

        if reverse:
            results.append(
                {
                    "suffix": " ".join(prefix.split()[::-1]),
                    "count": count,
                }
            )
        else:
            results.append(
                {
                    "prefix": prefix,
                    "count": count,
                }
            )
        prefixes.add(prefix)

    return results


def find_headers_footers(pages: list[str], min_words: int = 3, min_occurrences: int = 3):
    headers = find_common_prefixes(pages, reverse=False)
    footers = find_common_prefixes(pages, reverse=True)
    headers = [
        h["prefix"]
        if len(h["prefix"].split()) >= min_words and h["count"] >= min_occurrences
        else None
        for h in headers
    ]
    footers = [
        f["suffix"]
        if len(f["suffix"].split()) >= min_words and f["count"] >= min_occurrences
        else None
        for f in footers
    ]
    return headers, footers


def remove_headers_footers(
    pages: list[str], headers: list[str], footers: list[str]
) -> list[str]:
    cleaned_pages = []
    for page, h, f in zip(pages, headers, footers, strict=True):
        for t in [h, f]:
            if t is None:
                continue
            # ignore random whitespaces like \n or multiple spaces
            words = t.split()
            pattern = r"\s*".join(re.escape(word) for word in words)
            if t == h:
                # Remove header
                page = re.sub(r"^" + pattern + r"\s*", "", page)
            else:
                # Remove footer
                page = re.sub(r"\s*" + pattern + r"$", "", page)

        cleaned_pages.append(page)
    return cleaned_pages
