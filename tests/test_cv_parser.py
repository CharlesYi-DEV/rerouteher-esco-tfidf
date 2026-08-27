from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from docx import Document

from api.cv_parser import CvParseError, parse_cv


SAMPLE_CV = """JANE DOE
PROFESSIONAL EXPERIENCE
Software Engineer | Acme Sdn Bhd
January 2022 - Present
Built Python APIs and automated software testing.
Data Analyst | Example Ltd
June 2019 - December 2021
Prepared reports and managed databases.
SKILLS
Python, SQL, API design, software testing, databases
EDUCATION
Bachelor of Computer Science | 2015 - 2019
"""


class CvParserTests(unittest.TestCase):
    def test_txt_extracts_latest_role_skills_and_duration(self) -> None:
        parsed = parse_cv("jane_cv.txt", SAMPLE_CV.encode())
        self.assertEqual(parsed.job_title, "Software Engineer")
        self.assertIn("Python", parsed.skills)
        self.assertGreater(parsed.work_length_years or 0, 4)
        self.assertGreater(parsed.total_experience_years or 0, parsed.work_length_years or 0)
        self.assertEqual(parsed.employment_date_ranges_found, 2)

    def test_docx_is_supported(self) -> None:
        document = Document()
        for line in SAMPLE_CV.splitlines():
            document.add_paragraph(line)
        buffer = io.BytesIO()
        document.save(buffer)
        parsed = parse_cv("jane_cv.docx", buffer.getvalue())
        self.assertEqual(parsed.job_title, "Software Engineer")
        self.assertIn("SQL", parsed.skills)

    def test_overlapping_roles_are_not_double_counted(self) -> None:
        text = """WORK EXPERIENCE
Project Manager
2020 - 2023
Business Analyst
2021 - 2022
SKILLS
Planning, reporting
"""
        parsed = parse_cv("overlap.txt", text.encode())
        self.assertEqual(parsed.total_experience_years, 4.0)

    def test_role_wins_over_education_without_standard_headings(self) -> None:
        text = """Jane Doe
Software Engineer
Acme
2022 - Present
Bachelor of Science
2019 - 2021
Skills: Python, SQL
"""
        parsed = parse_cv("unstructured.txt", text.encode())
        self.assertEqual(parsed.job_title, "Software Engineer")
        self.assertIn("No standard experience heading", parsed.warnings[0])

    def test_two_column_pdf_is_read_in_column_order(self) -> None:
        left = """PROFILE
Data science postgraduate student
SKILLS
Python Programming
SQL Proficient
Business Relationship
Management
Project Management
LANGUAGE
English
"""
        right = """EDUCATION
Master of Data Science 2025 - Present
EXPERIENCE
Technical Sales Engineer Apr. 2024 - Dec. 2024
Example Machinery Ltd
Co-Founder Oct. 2021 - Jun. 2025
Example Store
PROJECTS
Machine Learning Project 2024
"""

        class FakeCrop:
            def __init__(self, text: str) -> None:
                self.text = text

            def extract_text(self, **_: object) -> str:
                return self.text

        class FakePage:
            bbox = (0.0, 0.0, 600.0, 800.0)

            def extract_text(self, **_: object) -> str:
                return "SKILLS EXPERIENCE\nApr. 2024 - Dec. 2024\nOct. 2021 - Jun. 2025"

            def crop(self, bbox: tuple[float, float, float, float]) -> FakeCrop:
                return FakeCrop(left if bbox[0] == 0.0 else right)

        class FakePdf:
            pages = [FakePage()]

            def __enter__(self) -> "FakePdf":
                return self

            def __exit__(self, *_: object) -> None:
                return None

        class FakeReader:
            is_encrypted = False
            pages = [object()]

        with patch("api.cv_parser.PdfReader", return_value=FakeReader()), patch(
            "api.cv_parser.pdfplumber.open", return_value=FakePdf()
        ):
            parsed = parse_cv("two-column.pdf", b"%PDF synthetic two-column resume")

        self.assertEqual(parsed.job_title, "Co-Founder")
        self.assertEqual(parsed.work_length_years, 3.75)
        self.assertEqual(parsed.total_experience_years, 3.75)
        self.assertEqual(
            parsed.skills,
            ["Python Programming", "SQL Proficient", "Business Relationship Management", "Project Management"],
        )
        self.assertNotIn("No standard experience heading", parsed.warnings)

    def test_unsupported_file_is_rejected(self) -> None:
        with self.assertRaises(CvParseError):
            parse_cv("resume.rtf", b"plain text that is long enough to parse")


if __name__ == "__main__":
    unittest.main()
