import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import annotation.annotator as annotator_module
from annotation.annotator import annotate_pair, load_config, run_job
from annotation.checkpoint import append_jsonl
from annotation.schema import DIMENSIONS
from annotation.validator import AnnotationValidationError, parse_and_validate


def good_annotation(brand_id="b01", creator_id="c01"):
    return {
        "brand_id":brand_id,"creator_id":creator_id,"label":"good",
        "reasoning":{
            "niche_match":"strong","audience_match":"strong","geography_match":"strong",
            "platform_match":"strong","budget_fit":"fit","creator_size_fit":"fit","campaign_fit":"strong",
        },
        "reason":"Strong alignment across the supplied campaign criteria.",
    }


def profile_fixture(n):
    brands = [{"brand_id":"b01","brand_name":"Example","industry":"Fitness","product":"Equipment",
        "campaign_title":"Launch","campaign_goal":"Awareness","campaign_description":"Campaign details",
        "target_audience":"Adults","target_age_range":"18-40","target_gender":"all","target_locations":["US"],
        "required_creator_niches":["fitness"],"preferred_creator_niches":["fitness","wellness"],
        "creator_size_preference":"mid-tier","minimum_followers":50000,"maximum_followers":500000,
        "budget":10000,"currency":"USD","content_types":["workout"],"platforms":["youtube"],"tone":"energetic",
        "mandatory_requirements":[],"preferred_traits":[],"excluded_traits":[]}]
    creators = [{"creator_id":f"c{i:02}","name":f"Creator {i}","primary_niche":"fitness","secondary_niches":["wellness"],
        "bio":"Fitness creator","location":"New York, US","languages":["English"],"platforms":["youtube"],
        "followers":100000,"engagement_rate":.04,"average_views":40000,"audience_age_range":"18-40",
        "audience_gender_distribution":{"female":.5,"male":.5},"audience_locations":["US"],"content_types":["workout"],
        "content_style":"energetic","rate_card":"$1,000–$2,000","rate_currency":"USD"} for i in range(1,n+1)]
    return brands, creators


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        item = self.responses.pop(0) if self.responses else good_annotation()
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, dict):
            context_text=prompt.split("PAIR CONTEXT (JSON)\n",1)[1]
            context,_=json.JSONDecoder().raw_decode(context_text)
            item=dict(item)
            item["brand_id"]=context["brand"]["brand_id"]
            item["creator_id"]=context["creator"]["creator_id"]
            return json.dumps(item)
        return item


class AnnotationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.brands, self.creators = profile_fixture(5)
        self.selected = {"pairs":[{"brand_id":"b01","creator_id":c["creator_id"]} for c in self.creators]}
        self.labels = {"labels":[good_annotation("b01","c01")]}
        self.paths = {
            "annotation_path":self.root/"selected.json", "labels_path":self.root/"labels.json",
            "brands_path":self.root/"brands.json", "creators_path":self.root/"creators.json",
            "checkpoint_path":self.root/"artifacts"/"annotations.jsonl",
            "failures_path":self.root/"artifacts"/"failures.jsonl",
        }
        for key, value in (("annotation_path",self.selected),("labels_path",self.labels),
                           ("brands_path",self.brands),("creators_path",self.creators)):
            self.paths[key].write_text(json.dumps(value),encoding="utf-8")
        # This completed pair must be skipped on resume.
        append_jsonl(self.paths["checkpoint_path"],good_annotation("b01","c02"))

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_annotation_passes_validation(self):
        self.assertEqual(parse_and_validate(good_annotation(),"b01","c01"),good_annotation())

    def test_invalid_label_rejected(self):
        value=good_annotation(); value["label"]="excellent"
        with self.assertRaises(AnnotationValidationError): parse_and_validate(value,"b01","c01")

    def test_missing_reasoning_dimension_rejected(self):
        value=good_annotation(); del value["reasoning"]["campaign_fit"]
        with self.assertRaises(AnnotationValidationError): parse_and_validate(value,"b01","c01")

    def test_wrong_brand_id_rejected(self):
        with self.assertRaises(AnnotationValidationError): parse_and_validate(good_annotation("b02"),"b01","c01")

    def test_wrong_creator_id_rejected(self):
        with self.assertRaises(AnnotationValidationError): parse_and_validate(good_annotation("b01","c02"),"b01","c01")

    def test_malformed_json_rejected(self):
        with self.assertRaises(AnnotationValidationError): parse_and_validate("{bad", "b01","c01")

    def test_markdown_wrapped_json_rejected(self):
        with self.assertRaises(AnnotationValidationError): parse_and_validate("```json\n{}\n```", "b01","c01")

    def test_retry_after_invalid_output(self):
        bad=good_annotation(); bad["label"]="invalid"
        client=FakeClient([bad,good_annotation()])
        result=annotate_pair(self.brands[0],self.creators[0],client)
        self.assertEqual(result["label"],"good"); self.assertEqual(client.calls,2)

    def test_failed_pair_is_recorded_after_retry_exhaustion(self):
        bad=good_annotation(); bad["label"]="invalid"
        client=FakeClient([bad,bad,bad])
        with contextlib.redirect_stdout(io.StringIO()):
            result=run_job(client,limit=1,**self.paths)
        rows=[json.loads(line) for line in self.paths["failures_path"].read_text().splitlines()]
        self.assertEqual(result["failed"],1); self.assertEqual(rows[0]["attempts"],3)
        self.assertEqual(rows[0]["error_type"],"validation_error")
        self.assertNotIn(("b01","c03"),{(r["brand_id"],r["creator_id"]) for r in __import__("annotation.checkpoint",fromlist=["read_jsonl"]).read_jsonl(self.paths["checkpoint_path"])})

    def test_existing_labeled_pairs_are_skipped(self):
        client=FakeClient([])
        with contextlib.redirect_stdout(io.StringIO()): result=run_job(client,limit=1,**self.paths)
        rows=[json.loads(line) for line in self.paths["checkpoint_path"].read_text().splitlines()]
        self.assertNotIn(("b01","c01"),{(r["brand_id"],r["creator_id"]) for r in rows})
        self.assertEqual(client.calls,1); self.assertEqual(result["processed"],1)

    def test_checkpointed_pairs_are_skipped(self):
        client=FakeClient([])
        with contextlib.redirect_stdout(io.StringIO()): result=run_job(client,limit=1,**self.paths)
        self.assertEqual(client.calls,1)
        rows=[json.loads(line) for line in self.paths["checkpoint_path"].read_text().splitlines()]
        self.assertEqual(sum(r["creator_id"]=="c02" for r in rows),1)

    def test_successful_annotation_is_persisted(self):
        client=FakeClient([good_annotation("b01","c03")])
        with contextlib.redirect_stdout(io.StringIO()): run_job(client,limit=1,**self.paths)
        rows=[json.loads(line) for line in self.paths["checkpoint_path"].read_text().splitlines()]
        self.assertIn(("b01","c03"),{(r["brand_id"],r["creator_id"]) for r in rows})

    def test_resume_after_interruption(self):
        interrupted=FakeClient([good_annotation("b01","c03"),KeyboardInterrupt()])
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt): run_job(interrupted,limit=2,**self.paths)
        resumed=FakeClient([good_annotation("b01","c04")])
        with contextlib.redirect_stdout(io.StringIO()): result=run_job(resumed,limit=1,**self.paths)
        self.assertEqual(resumed.calls,1); self.assertEqual(result["processed"],1)
        rows=[json.loads(line) for line in self.paths["checkpoint_path"].read_text().splitlines()]
        self.assertEqual(len({(r["brand_id"],r["creator_id"]) for r in rows}),3)

    def test_dry_run_makes_zero_llm_calls_and_writes(self):
        client=FakeClient([])
        before=self.paths["checkpoint_path"].read_text()
        with contextlib.redirect_stdout(io.StringIO()): result=run_job(client,dry_run=True,**self.paths)
        self.assertEqual(client.calls,0); self.assertEqual(result["calls"],0)
        self.assertEqual(before,self.paths["checkpoint_path"].read_text())

    def test_limit_three_processes_at_most_three_pairs(self):
        client=FakeClient([])
        with contextlib.redirect_stdout(io.StringIO()): result=run_job(client,limit=3,**self.paths)
        self.assertEqual(client.calls,3); self.assertEqual(result["processed"],3)

    def test_cli_limit_three_uses_production_path_and_processes_at_most_three(self):
        client=FakeClient([])
        with patch.dict("os.environ",{"GROQ_API_KEY":"test-key","GROQ_MODEL":"llama-3.3-70b-versatile"}), \
             patch.object(annotator_module,"create_provider_client",return_value=client), \
             patch.object(annotator_module,"ANNOTATION_PATH",self.paths["annotation_path"]), \
             patch.object(annotator_module,"LABELS_PATH",self.paths["labels_path"]), \
             patch.object(annotator_module,"BRANDS_PATH",self.paths["brands_path"]), \
             patch.object(annotator_module,"CREATORS_PATH",self.paths["creators_path"]), \
             patch.object(annotator_module,"CHECKPOINT_PATH",self.paths["checkpoint_path"]), \
             patch.object(annotator_module,"FAILURES_PATH",self.paths["failures_path"]), \
             contextlib.redirect_stdout(io.StringIO()):
            code=annotator_module.main(["--limit","3"])
        self.assertEqual(code,0); self.assertEqual(client.calls,3)

    def test_groq_configuration_uses_env_key_and_selected_model(self):
        with patch.dict("os.environ",{"GROQ_API_KEY":"groq-test","GROQ_MODEL":"llama-3.3-70b-versatile"},clear=True):
            config=load_config(require_api_key=True)
        self.assertEqual(config["model"],"llama-3.3-70b-versatile")
        self.assertEqual(config["api_key"],"groq-test")


if __name__ == "__main__":
    unittest.main()
