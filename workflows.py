import json
from pathlib import Path


WORKFLOWS_SOURCE = "./workflows"


def get_workflows():
    workflows = []
    workflows_dir = Path(WORKFLOWS_SOURCE)
    for f in workflows_dir.rglob("*"):
        f: Path
        if not f.is_file() or f.suffix.lower() != ".json":
            continue

        workflows.append(f)

    return sorted(workflows)


def load_workflow(workflow: Path):
    return json.load(workflow.open())
