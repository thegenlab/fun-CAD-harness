"""Free creation mode; reuse the existing author/feedback loop."""
from __future__ import annotations
import json
from pathlib import Path
from ..mechanism.io import write_json_atomic

MODES = ("free",)


def available_modes(workspace=None):
    return MODES


def resolve_task(task=None):
    """Require the user's natural-language task before creating a project or calling a model."""
    if task is None or not task.strip():
        raise ValueError("Provide --task or --brief for free creation")
    return task


def configure(mode="free"):
    if mode not in MODES:raise ValueError("Free creation is the only construction mode")
    return {"mode":mode}


def install(project, configuration):
    value=dict(configuration)
    write_json_atomic(Path(project)/"mode.json",value)
    return value


def settings(project):
    path=Path(project)/"mode.json"
    if not path.is_file():return None
    value=json.loads(path.read_text(encoding="utf-8"))
    if value.get("mode") not in MODES:raise ValueError("Unknown stored project mode")
    return value


def author_instructions(project):
    config=settings(project)
    if not config:return ""
    language = ("Use English for all newly authored code comments, documentation, parameter explanations, "
                "public progress messages and reports. " if config.get("language") == "en" else "")
    return ("\nEXPLICIT MODE: "+config["mode"]+". "+language+
            "Decompose the new prompt into task points and create task_template.py using the public "
            "benchmark_templates.runtime Template/Trial/Action/Check API. Declare PARAMETERS and build(p). "
            "The independent Harness reviewer reviews this template and fills the actual instance. "
            "Deliver model source, STEP, scene.json and source-bound parts.json. Source scene tests may be empty: "
            "the template owns the functional tests. Do not write binding.py or fill your own template instance. "
            "The Harness review agent maps actual parts, operations, measurements and parameters. "
            "You decide tradeoffs between mechanical design, allowed flexible models and task difficulty. "
            "To adjust difficulty, write template_adjustment.json with parameters (name/value object) and reason. "
            "The reviewer implements your parameter decisions; its advice is not a veto. Preserve the original "
            "prompt and core functions; report the actual adjusted target. Do not replace public physics or tests. "
            "All new template sources and dependencies stay in this design directory.\n")
