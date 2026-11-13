import csv
import tempfile
import unittest
from pathlib import Path

from src.auditing import (
    write_grounding_precheck,
    write_metadata_precheck,
    write_source_unit_inventory,
)
from src.models import DocumentPage


class AuditPrecheckTests(unittest.TestCase):
    def test_prechecks_fill_objective_fields_but_leave_human_scores_blank(self) -> None:
        page = DocumentPage(
            "book",
            "source.txt",
            1,
            "Alice found the key.",
            {
                "unit_id": "book-unit-001",
                "section_id": "chapter-i",
                "section": "Chapter I",
                "unit_start_char": 0,
                "unit_end_char": 20,
                "char_count": 20,
            },
        )
        hit = {
            "document_id": "book",
            "source": "source.txt",
            "page": 1,
            "chunk_id": "chunk-1",
            "text": "Alice found the key.",
            "metadata": {
                "document_id": "book",
                "source": "source.txt",
                "page": 1,
                "section_id": "chapter-i",
            },
        }
        trace = {
            "question_id": "q1",
            "split": "test",
            "strategy": "C",
            "question": "What did Alice find?",
            "filters": {"section_id": "chapter-i"},
            "evidence_quote": "Alice found the key.",
            "gold_locations": [{"gold_source_id": "book", "gold_page": 1}],
            "hits": [hit],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            inventory_path = write_source_unit_inventory(output_dir, [page])
            metadata_path = write_metadata_precheck(output_dir, [page], [trace])
            grounding_path = write_grounding_precheck(output_dir, [trace])

            with metadata_path.open(encoding="utf-8", newline="") as file:
                metadata_row = next(csv.DictReader(file))
            with grounding_path.open(encoding="utf-8", newline="") as file:
                grounding_row = next(csv.DictReader(file))
            with inventory_path.open(encoding="utf-8", newline="") as file:
                inventory_row = next(csv.DictReader(file))

        self.assertEqual(metadata_row["page_exists"], "True")
        self.assertEqual(metadata_row["required_metadata_present"], "True")
        self.assertEqual(metadata_row["metadata_matches_source_page"], "True")
        self.assertEqual(metadata_row["filter_compliant"], "True")
        self.assertEqual(metadata_row["metadata_useful"], "")
        self.assertEqual(grounding_row["exact_evidence_ranks"], "1")
        self.assertEqual(grounding_row["human_score_0_to_2"], "")
        self.assertEqual(inventory_row["unit_id"], "book-unit-001")
        self.assertEqual(inventory_row["char_count"], "20")


if __name__ == "__main__":
    unittest.main()
