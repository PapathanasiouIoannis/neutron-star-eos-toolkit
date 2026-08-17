"""Load and inspect one downloaded cold, beta-equilibrated CompOSE EoS.

Run acquisition first if needed:
``uv run python experiments/compose_comparison/acquire.py --model sly4``
"""

from pathlib import Path

try:
    from neutron_star_eos import open_eos
except ModuleNotFoundError as exc:
    raise SystemExit("Install the project first: uv sync --all-extras") from exc

# ------------------------- parameters to edit -------------------------
MODEL_SLUG = "sly4"
MODEL_ID = "RG(SLy4)"
COMPOSE_PAGE = "https://compose.obspm.fr/eos/134"
ORDERING_POLICY = "strict"

repository = Path(__file__).resolve().parents[1]
archive = (
    repository
    / "experiments"
    / "compose_comparison"
    / "data"
    / "raw"
    / MODEL_SLUG
    / "archive.zip"
)
if not archive.is_file():
    raise SystemExit(
        f"CompOSE archive not found: {archive}\n"
        f"Acquire it with: uv run python experiments/compose_comparison/acquire.py "
        f"--model {MODEL_SLUG}"
    )

model = open_eos(
    archive,
    kind="compose",
    model_id=MODEL_ID,
    source_url=COMPOSE_PAGE,
    matter="cold_beta_equilibrated",
    includes_leptons=True,
    ordering_policy=ORDERING_POLICY,
)

print(model.summary())
print("Available thermodynamic views:", ", ".join(model.thermodynamics().roles))
