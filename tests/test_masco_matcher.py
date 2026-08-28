from __future__ import annotations

import re
import unittest

import numpy as np

from api.masco_matcher import GranularMascoMatcher, compose_text


class DummyEncoder:
    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        vectors = np.ones((len(texts), 384), dtype=float)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class GranularMascoMatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matcher = GranularMascoMatcher(encoder=DummyEncoder())

    def assert_six_digit_matches(self, result: dict) -> None:
        self.assertEqual(len(result["matches"]), 3)
        for match in result["matches"]:
            self.assertRegex(match["masco_code"], r"^\d{6}$")
            self.assertRegex(match["masco_code_printed"], r"^\d{4}-\d{2}$")
            self.assertEqual(
                match["masco_code"], match["masco_code_printed"].replace("-", "")
            )
            self.assertTrue(match["requires_user_confirmation"])

    def test_artifact_is_jobhop_only_and_catalog_is_granular(self) -> None:
        self.assertEqual(
            self.matcher.artifact["resume_dataset"],
            "JobHop v2 confirmed active 2019+ only",
        )
        self.assertEqual(len(self.matcher.catalog), 258)
        self.assertEqual(len(self.matcher.classes), 19)
        self.assertTrue(
            all(re.fullmatch(r"\d{6}", code) for code in self.matcher.classes)
        )

    def test_tfidf_predictions_use_only_six_digit_masco_codes(self) -> None:
        result = self.matcher.predict_tfidf(
            "Software Engineer", ["Python", "API design", "software testing"], 4.5
        )
        self.assertEqual(result["label_format"], "six-digit MASCO")
        self.assert_six_digit_matches(result)

    def test_minilm_predictions_use_only_six_digit_masco_codes(self) -> None:
        result = self.matcher.predict_minilm(
            "Graphic Designer", ["visual design", "digital media"], 3.0
        )
        self.assertEqual(result["label_format"], "six-digit MASCO")
        self.assert_six_digit_matches(result)

    def test_input_adapter_matches_jobhop_structure(self) -> None:
        text = compose_text("Software Engineer", ["Python", "SQL"], 2.25)
        self.assertEqual(
            text,
            "title software developer ; skills python, sql ; duration 2.25 years",
        )


if __name__ == "__main__":
    unittest.main()
