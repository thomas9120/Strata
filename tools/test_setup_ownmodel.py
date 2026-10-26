"""setup.py's model question takes "I already have the model files" (START-HERE.bat / setup.sh): setup reads the names and
the headers of what is in the folder, says which model and size they are, and installs that from where they are - no
model file is downloaded and the size question is not asked (their files are the size).  The checker is
strata_control.models.plan_import, the same one the app and the Terminal use, and nothing of the person's is changed.

    python -m unittest tools.test_setup_ownmodel
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import test_setup_golden as g  # noqa: E402

# the GGUF builder of the Terminal's own-files tests (a tiny qwen4exp file with one routed-expert tensor per type named)
_spec = importlib.util.spec_from_file_location("own_files_helper", ROOT / "tests" / "terminal" / "test_own_files.py")
_helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helper)
gguf_bytes, Q2, IQ3XXS = _helper.gguf_bytes, _helper.Q2, _helper.IQ3XXS
UNSLOTH_IQ4 = _helper.UNSLOTH_IQ4

PC64 = (63.7, [g.card(0, "NVIDIA GeForce RTX 5070", 11.9, "120")])
READER = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"


def tiny(folder: Path, names: list[str], experts: list[str]) -> Path:
    """A folder with tiny stand-ins for the real shards: the names (and the header) are what setup reads."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(gguf_bytes(experts))
    return folder


def arch_gguf(arch: str) -> bytes:
    """A tiny GGUF of another architecture (Qwen3.6's qwen35moe): its header and no tensors."""
    header = (b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + _helper.string("general.architecture") + struct.pack("<I", 8)
              + _helper.string(arch))
    return header + b"\x00" * (-len(header) % 32)


Q36 = "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


class Detection(unittest.TestCase):
    """What setup says a folder of model files is (strata_control.models.plan_import, read-only)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-own-model-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def test_a_published_set_is_named_by_its_files(self):
        folder = tiny(self.base / "Q2", Q2, ["Q2_0"])
        plan, out = quiet(setup.own_files_plan, str(folder))
        self.assertEqual(plan["candidate_selection"]["family"], "qwen")
        self.assertEqual(plan["candidate_selection"]["model"], "Q2_0")
        self.assertEqual(plan["model_dir"], str(folder))
        self.assertEqual(setup.own_files_problem(plan), ([], []))
        self.assertEqual(out, "")                              # reading a folder says nothing and writes nothing

    def test_the_folder_above_the_model_files_is_accepted(self):
        folder = tiny(self.base / "repo" / "IQ3_XXS", IQ3XXS, ["IQ3_XXS"])
        plan, _ = quiet(setup.own_files_plan, str(self.base / "repo"))
        self.assertEqual(plan["model_dir"], str(folder))       # (a whole Hugging Face download)
        self.assertEqual(plan["candidate_selection"]["model"], "IQ3_XXS")

    def test_the_reader_beside_the_files_is_reported(self):
        folder = tiny(self.base / "Q2", Q2, ["Q2_0"])
        (folder / READER).write_bytes(b"not read by this test")
        plan, _ = quiet(setup.own_files_plan, str(folder))
        self.assertEqual([row["name"] for row in plan["projectors"]], [READER])
        self.assertEqual(plan["candidate_selection"]["model"], "Q2_0")

    def test_a_model_strata_does_not_know_says_so(self):
        folder = tiny(self.base / "ft", ["my-finetune-00001-of-00002.gguf", "my-finetune-00002-of-00002.gguf"], ["IQ3_S"])
        plan, _ = quiet(setup.own_files_plan, str(folder))
        self.assertIsNone(plan["candidate_selection"])
        wrong, _todo = setup.own_files_problem(plan)
        self.assertEqual(wrong, ["this is not a model size Strata knows"])


class TheQuestion(unittest.TestCase):
    """own_files_ask: what setup says for a folder, a file, a wrong path or Enter."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-own-ask-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def ask(self, answers):
        it = iter(answers)
        prompts = []

        def fake_input(prompt=""):
            prompts.append(prompt)
            try:
                return next(it)
            except StopIteration:
                raise EOFError from None

        a = argparse.Namespace(family=None, model=None, gguf_dir=None, own_files=False)
        with mock.patch("builtins.input", fake_input):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                return setup.own_files_ask(a), out.getvalue(), a, prompts

    def test_a_folder_of_files_set_up_that_size(self):
        folder = tiny(self.base / "Q2", Q2, ["Q2_0"])
        family, out, a, _prompts = self.ask([str(folder)])
        self.assertEqual(family, "qwen")
        self.assertEqual((a.family, a.model, a.gguf_dir, a.own_files), ("qwen", "Q2_0", str(folder), True))
        self.assertIn(f"found Qwen3.8-Flash-Next Q2_0 in {folder}", out)
        self.assertIn("no model file is downloaded", out)

    def test_one_model_file_uses_the_folder_it_is_in(self):
        folder = tiny(self.base / "IQ3", IQ3XXS, ["IQ3_XXS"])
        family, _out, a, _prompts = self.ask([str(folder / IQ3XXS[0])])
        self.assertEqual((family, a.gguf_dir), ("qwen", str(folder)))

    def test_three_shards_are_found_by_their_own_count(self):
        folder = tiny(self.base / "IQ4", UNSLOTH_IQ4, ["IQ4_XS"])
        family, out, a, _prompts = self.ask([str(folder)])
        self.assertEqual((family, a.model), ("unsloth", "UD-IQ4_XS"))
        self.assertIn("3 model files", out)

    def test_a_deepseek_set_goes_to_its_own_installer(self):
        folder = tiny(self.base / "ds", ["any-00001-of-00009.gguf"], ["IQ2_XS"])
        plan = {"model_dir": str(folder), "file_count": 9, "total_bytes": 2.57e11, "file_set_status": "plausible",
                "candidate_selection": {"family": "deepseek41", "model": "IST-DAS-2.00bpw"}, "plain_issues": [],
                "messages": ["This is DeepSeek-V4.1-Flash, 2.00 bpw: all 9 model files are here."]}
        a = argparse.Namespace(family=None, model=None, gguf_dir=None, own_files=False)
        with mock.patch("builtins.input", lambda prompt="": str(folder)), \
                mock.patch.object(setup, "own_files_plan", lambda raw: plan):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                family = setup.own_files_ask(a)
        self.assertEqual((family, a.model, a.gguf_dir), ("deepseek41", "IST-DAS-2.00bpw", str(folder)))
        self.assertIn("This is DeepSeek-V4.1-Flash, 2.00 bpw: all 9 model files are here.", out.getvalue())

    def test_a_deepseek_set_names_the_shared_files_still_downloaded(self):
        folder = tiny(self.base / "ds", ["any-00001-of-00004.gguf"], ["IQ2_XS"])
        plan = {"model_dir": str(folder), "file_count": 4, "total_bytes": 1.38e11, "file_set_status": "plausible",
                "candidate_selection": {"family": "deepseek41", "model": "IST-DAS-2.00bpw"}, "plain_issues": [],
                "messages": []}
        engram = setup.DSV41_ISTA_ENGRAM

        def ask(beside):
            a = argparse.Namespace(family=None, model=None, gguf_dir=None, own_files=False,
                                   models_dir=str(self.base / "models"))
            out = io.StringIO()
            with mock.patch("builtins.input", lambda prompt="": str(folder)), \
                    mock.patch.object(setup, "own_files_plan", lambda raw: plan), \
                    mock.patch.dict(setup.DSV41_ALL_FILES, {n: (5, "0" * 64) for n in beside}), \
                    contextlib.redirect_stdout(out):
                for n in beside:
                    (folder / n).write_bytes(b"12345")
                setup.own_files_ask(a)
            return out.getvalue()

        out = ask([])
        self.assertIn("Still downloaded, once, into", out)
        self.assertIn(f"{engram} (110.6 GB)", out)
        self.assertIn(f"{setup.DSV41_DSPARK} (8.0 GB)", out)
        self.assertNotIn("no model file is downloaded", out)
        shared = setup.MULTI_FILE_FAMILIES["deepseek41"]["models"]["IST-DAS-2.00bpw"]["shared"]
        out = ask(shared)                                       # all of them beside the person's shards
        self.assertIn("no model file is downloaded", out)
        self.assertNotIn("Still downloaded", out)

    def test_a_qwen36_file_goes_to_its_own_installer(self):
        folder = self.base / "q36"
        folder.mkdir()
        (folder / Q36).write_bytes(arch_gguf("qwen35moe"))
        for typed in (str(folder), str(folder / Q36)):
            family, out, a, _prompts = self.ask([typed])
            self.assertEqual((family, a.model, a.gguf_dir, a.own_files), ("qwen36", "UD-Q4_K_XL", str(folder), True))
            self.assertIn(f"found Qwen3.6-35B-A3B UD-Q4_K_XL in {folder}", out)
            self.assertNotIn("llama-server", out)

    def test_a_renamed_qwen36_file_is_told_its_published_name_not_llama_server(self):
        folder = self.base / "q36"
        folder.mkdir()
        (folder / "my-qwen.gguf").write_bytes(arch_gguf("qwen35moe"))
        family, out, _a, _prompts = self.ask([str(folder), ""])
        self.assertIsNone(family)                               # (Enter: back to the list)
        self.assertIn("published names", out)
        self.assertIn(Q36, out)
        self.assertNotIn("llama-server", out)

    def test_a_set_recognised_by_its_header_is_used_under_its_own_names(self):
        renamed = ["my-q2-00001-of-00002.gguf", "my-q2-00002-of-00002.gguf"]
        folder = tiny(self.base / "mine", renamed + ["other-00001-of-00002.gguf", "other-00002-of-00002.gguf"], ["Q2_0"])
        plan = {"model_dir": str(folder), "file_count": 2, "total_bytes": 2e10, "file_set_status": "plausible",
                "candidate_selection": {"family": "qwen", "model": "Q2_0"}, "plain_issues": [], "messages": [],
                "files": [{"name": n} for n in renamed]}
        a = argparse.Namespace(family=None, model=None, gguf_dir=None, own_files=False)
        with mock.patch("builtins.input", lambda prompt="": str(folder / renamed[0])), \
                mock.patch.object(setup, "own_files_plan", lambda raw: plan), contextlib.redirect_stdout(io.StringIO()):
            family = setup.own_files_ask(a)
        self.assertEqual((family, a.gguf_first), ("qwen", renamed[0]))
        shards = setup.gguf_dir_shards(folder, setup.FAMILIES["qwen"], "Q2_0", a.gguf_first)
        self.assertEqual([s.name for s in shards], renamed)    # not refused for a name setup does not know

    def test_enter_goes_back_to_the_model_list(self):
        family, _out, a, prompts = self.ask([""])
        self.assertIsNone(family)
        self.assertIsNone(a.gguf_dir)
        self.assertIn("Enter = back to the list", prompts[0])

    def test_a_path_that_is_not_there_is_said_then_the_folder_is_taken(self):
        folder = tiny(self.base / "Q2", Q2, ["Q2_0"])
        family, out, _a, _prompts = self.ask([str(self.base / "nope"), str(folder)])
        self.assertEqual(family, "qwen")
        self.assertIn("does not exist", out)
        self.assertTrue(out.index("does not exist") < out.index("found Qwen3.8-Flash-Next"))

    def test_a_folder_with_several_models_asks_for_the_one_folder(self):
        one = tiny(self.base / "both" / "Q2_0", Q2, ["Q2_0"])
        tiny(self.base / "both" / "IQ3_XXS", IQ3XXS, ["IQ3_XXS"])
        family, out, _a, _prompts = self.ask([str(self.base / "both"), str(one)])
        self.assertEqual(family, "qwen")
        self.assertIn("the folder holds more than one model", out)

    def test_input_that_ends_stops_with_the_way_to_do_it_without_questions(self):
        a = argparse.Namespace(family=None, model=None, gguf_dir=None, own_files=False)

        def fake_input(prompt=""):
            raise EOFError

        out = io.StringIO()
        with mock.patch("builtins.input", fake_input), contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as ctx:
                setup.own_files_ask(a)
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("--gguf-dir", out.getvalue())


class Installation(unittest.TestCase):
    """setup.main() with the answer and the mocked PC of tools/test_setup_golden.py: no model is downloaded, the size
    question is not asked, and the run config is the one --family qwen --model Q2_0 --gguf-dir DIR writes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-own-install-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def menu_number(self) -> str:
        return str(len(setup.menu_families(False)) + len(setup.OWN_PATH_FAMILIES)
                   + len(setup.MULTI_FILE_FAMILIES) + 1)

    def install(self, folder: Path, ram_cards=PC64, extra=()):
        downloads, runs = [], []

        def fake_download(url, dst, what=None):
            downloads.append(Path(dst))

        def fake_run(cmd, *a, **k):
            runs.append([str(c) for c in cmd])

        answers = {"Which model?": self.menu_number(), "Folder or file": str(folder)}
        patches = [mock.patch.object(setup, "download", fake_download), mock.patch.object(setup, "run", fake_run),
                   mock.patch.object(setup, "whole_shard", lambda s: True),      # the stand-ins are a few hundred bytes
                   *extra]
        code, out, cfg, asked = g.install(ram_cards[0], ram_cards[1], ["--no-start"], answers=answers, extra=patches)
        return code, out, cfg, asked, downloads, runs

    def test_the_model_question_offers_the_files_you_already_have(self):
        folder = tiny(self.base / "Q2", Q2, ["Q2_0"])
        (folder / READER).write_bytes(b"the person's own picture reader")
        code, out, cfg, asked, downloads, runs = self.install(folder)
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("I already have the model files", out)
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-q2_0")
        self.assertFalse([a for a in asked if "Which size?" in a], asked)     # their files are the size
        self.assertNotIn("IQ2_XS", out)                                       # so no list of sizes to download
        self.assertEqual(downloads, [])                                       # nothing was downloaded at all
        pack = next(r for r in runs if any("iq_pack.py" in c for c in r))
        self.assertEqual(pack[pack.index("--gguf") + 1], str(folder / Q2[0]))  # packed from the person's own shard

    def test_the_reader_in_the_folder_is_used_and_nothing_is_downloaded(self):
        folder = tiny(self.base / "Q2", Q2, ["Q2_0"])
        (folder / READER).write_bytes(b"the person's own picture reader")
        code, out, _cfg, _asked, downloads, _runs = self.install(folder)
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(downloads, [])
        self.assertIn(f"Your folder has {READER}", out)

    def test_without_a_reader_setup_downloads_it_beside_the_models_not_into_the_folder(self):
        folder = tiny(self.base / "Q2", Q2, ["Q2_0"])
        code, out, _cfg, _asked, downloads, _runs = self.install(folder)
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual([p.name for p in downloads], [READER])
        self.assertEqual(downloads[0].parent.name, "models")                   # (--models-dir of the test's setup run)
        self.assertFalse((folder / READER).exists())                          # never written into the person's folder

    def test_a_renamed_set_is_packed_from_the_file_the_check_recognised(self):
        renamed = ["my-q2-00001-of-00002.gguf", "my-q2-00002-of-00002.gguf"]
        folder = tiny(self.base / "mine", renamed, ["Q2_0"])
        plan = {"model_dir": str(folder), "file_count": 2, "total_bytes": 2e10, "file_set_status": "plausible",
                "candidate_selection": {"family": "qwen", "model": "Q2_0"}, "plain_issues": [], "messages": [],
                "files": [{"name": n} for n in renamed]}
        code, out, _cfg, _asked, downloads, runs = self.install(
            folder, extra=[mock.patch.object(setup, "own_files_plan", lambda raw: plan)])
        self.assertEqual(code, 0, out[-3000:])
        self.assertNotIn(setup.model_file(setup.FAMILIES["qwen"], "Q2_0", 1), [p.name for p in downloads])
        pack = next(r for r in runs if any("iq_pack.py" in c for c in r))
        self.assertEqual(pack[pack.index("--gguf") + 1], str(folder / renamed[0]))

    def test_the_disk_space_check_is_the_folder_setup_writes_to_not_theirs(self):        # nothing is written into it
        folder = tiny(self.base / "Q2", Q2, ["Q2_0"])
        code, out, _cfg, _asked, _downloads, _runs = self.install(
            folder, extra=[mock.patch.object(setup, "free_gb", lambda p: 0.5)])
        self.assertEqual(code, 1, out[-3000:])
        line = next(x for x in out.splitlines() if "not enough free disk space in" in x)
        path = line.split("not enough free disk space in ")[1].split(": need")[0]
        self.assertTrue(path.endswith("data"), line)                        # the data folder (packs, draft layer, reader)
        self.assertNotIn(str(folder), line)


if __name__ == "__main__":
    unittest.main()
