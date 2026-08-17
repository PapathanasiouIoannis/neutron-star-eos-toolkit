"""Structural and headless execution checks for the experiment notebook."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import nbformat
    from nbclient import NotebookClient
except ModuleNotFoundError:  # Core installations intentionally omit notebook tools.
    nbformat = None
    NotebookClient = None


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks" / "eos_experiments.ipynb"


@unittest.skipUnless(
    nbformat is not None and NotebookClient is not None,
    "install the notebook-test optional dependencies to execute notebook checks",
)
class ExperimentNotebookTests(unittest.TestCase):
    def read_notebook(self):
        return nbformat.read(NOTEBOOK_PATH, as_version=4)

    def test_committed_notebook_is_valid_and_has_no_outputs(self) -> None:
        notebook = self.read_notebook()
        nbformat.validate(notebook)
        self.assertEqual(notebook.metadata.kernelspec.name, "python3")
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            self.assertIsNone(cell.execution_count)
            self.assertEqual(cell.outputs, [])

        markdown = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "markdown"
        )
        headings = (
            "## 1. Quick start",
            "## 2. Load the selected EoS",
            "## 3. Thermodynamic validation",
            "## 4. One-star TOV calculation",
            "## 5. Mass–radius sequence",
            "## 6. Optional advanced controls",
            "## 7. Optional saving and provenance",
        )
        positions = [markdown.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("uv sync --all-extras", markdown)
        self.assertIn("ipykernel install", markdown)

    def test_default_notebook_executes_headlessly_without_modifying_source(
        self,
    ) -> None:
        original = NOTEBOOK_PATH.read_bytes()
        notebook = self.read_notebook()
        resources = {"metadata": {"path": str(REPOSITORY_ROOT / "notebooks")}}
        client = NotebookClient(
            notebook,
            timeout=300,
            kernel_name="python3",
            allow_errors=False,
            resources=resources,
        )
        with patch.dict(os.environ, {"MPLBACKEND": "Agg"}):
            executed = client.execute()

        nbformat.validate(executed)
        rendered_text: list[str] = []
        for cell in executed.cells:
            if cell.cell_type != "code":
                continue
            for output in cell.outputs:
                if output.output_type == "stream":
                    rendered_text.append(str(output.text))
                elif output.output_type in {"execute_result", "display_data"}:
                    rendered_text.append(str(output.data.get("text/plain", "")))
        self.assertIn("EOS_NOTEBOOK_EXECUTION_OK", "\n".join(rendered_text))
        self.assertEqual(NOTEBOOK_PATH.read_bytes(), original)

        with tempfile.TemporaryDirectory() as temporary:
            executed_path = Path(temporary) / "eos_experiments.executed.ipynb"
            nbformat.write(executed, executed_path)
            self.assertTrue(executed_path.is_file())


if __name__ == "__main__":
    unittest.main()
