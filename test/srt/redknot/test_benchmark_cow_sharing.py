"""CPU-only contract tests; no model, CUDA, download, or remote execution."""
import copy
import json
from types import SimpleNamespace

import pytest

import benchmark_cow_sharing as bench


def workload():
    document=[list(range(64))]
    prefixes=[[100,i] for i in range(32)]
    query=[200]
    item=dict(key=bench.ITEM_KEYS[0],primary_qa_key="hotpotqa:76",variant="long",
        augmentation="complete real distractor plus complete primary context",
        dataset="hotpotqa",document_token_ids=document,context_tokens=64,
        document_tokens_sha256=bench.digest(document),prefix_token_ids=prefixes,
        query_token_ids=query,answers=["answer"],
        exact_prompt_sha256=[bench.digest(p+document[0]+query) for p in prefixes])
    return dict(schema="redknot-nonprefix-frozen-v1",request_replicas=32,
        max_new_tokens=32,items=[item])


def plan(**kwargs):
    return bench.prepare_plan(workload(),bench.ITEM_KEYS[0],0,16,**kwargs)


def row(arm,repeat=0,seconds=10.):
    records=[dict(request_id=f"request:{i}",status="completed",document_ids=["doc"],
        max_new_tokens=32,frozen_prefix_index=i,exact_prompt_sha256=plan()["case"]["exact_prompt_sha256"][i],
        output_ids=[1,2],text="answer",accuracy=dict(f1=0.,em=0.),capacity_blocks=0)
        for i in range(32)]
    return dict(arm=arm,repeat=repeat,status="complete",cleanup_status="complete",fatal=False,
        audit_errors=[],service=dict(status="complete",errors=[],ignore_eos=False,
            completed_requests=32,failed_requests=0,cancelled_requests=0,
            completed_output_tokens=64,total_wall_seconds=seconds,records=records,
            peak_active_requests=16,max_forward_batch=16))


@pytest.mark.parametrize("geometry",bench.GEOMETRIES)
@pytest.mark.parametrize("key",bench.ITEM_KEYS)
def test_six_predeclared_cells(geometry,key):
    w=workload();w["items"][0]["key"]=key
    p=bench.prepare_plan(w,key,*geometry,repeats=2)
    assert p["case"]["requests"]==32 and p["case"]["chunks"]==16
    assert p["total_kv_budget_bytes"]==48*1024**3
    assert p["candidate"]["row_fraction"]==.5 and p["candidate"]["probe_layers"]==8
    assert p["arm_order_by_repeat"]==[list(bench.ARMS),list(reversed(bench.ARMS))]
    assert p["independent_primary_qa_count"]==1


@pytest.mark.parametrize("mutation",["hash","shortprefix","duplicateprefix","augmentation","context"])
def test_frozen_inputs_fail_closed(mutation):
    w=workload();i=w["items"][0]
    if mutation=="hash":i["document_tokens_sha256"]="wrong"
    if mutation=="shortprefix":i["prefix_token_ids"].pop()
    if mutation=="duplicateprefix":i["prefix_token_ids"][1]=i["prefix_token_ids"][0]
    if mutation=="augmentation":i["augmentation"]=None
    if mutation=="context":i["context_tokens"]+=1
    with pytest.raises(ValueError):bench.prepare_plan(w,bench.ITEM_KEYS[0],0,16)


def test_unregistered_geometry():
    with pytest.raises(ValueError):bench.prepare_plan(workload(),bench.ITEM_KEYS[0],63,4)


def test_metadata_off_preserves_original_frozen_plan_hash():
    original="6b248694188c706a5f02cb21bd3162c50542216767801210b6600bfb9f039983"
    assert bench.digest(plan())==original
    assert plan(metadata_only_alias_sources=False)==plan()
    enabled=plan(metadata_only_alias_sources=True)
    assert enabled.pop("metadata_only_alias_sources") is True
    assert enabled==plan()


@pytest.mark.parametrize("invalid",[None,1,"true"])
def test_metadata_flag_requires_boolean(invalid):
    with pytest.raises(ValueError):plan(metadata_only_alias_sources=invalid)
    with pytest.raises(ValueError):bench.arm_factory(lambda *a,**k:k,SimpleNamespace,
        arm="redknot_shared",candidate=plan()["candidate"],cache_bytes=1024,page_bytes=256,
        receipt={},metadata_only_alias_sources=invalid)


@pytest.mark.parametrize("arm",bench.ARMS)
@pytest.mark.parametrize("diagnostic",[False,True])
@pytest.mark.parametrize("metadata",[False,True])
def test_factory_identical_selective_cache_and_correct_storage_mode(arm,diagnostic,metadata):
    receipt={};calls=[]
    def Engine(model,cfg,**kwargs):calls.append(kwargs);return kwargs
    factory=bench.arm_factory(Engine,SimpleNamespace,arm=arm,candidate=plan()["candidate"],
        cache_bytes=1024,page_bytes=256,receipt=receipt,publication_diagnostics=diagnostic,
        metadata_only_alias_sources=metadata)
    result=factory(None,None,capacity_pages=bench.TOTAL_BYTES//256,share_local=arm!="redknot_private")
    expected=0 if arm==bench.ARMS[0] else 1024
    assert receipt["canonical_cache_reservation_bytes"]==expected
    assert result["capacity_pages"]*256+expected==bench.TOTAL_BYTES
    assert result.get("publication_diagnostics",False)==diagnostic
    if metadata and arm=="redknot_shared":
        assert result["metadata_only_alias_sources"] is True
    else:
        assert "metadata_only_alias_sources" not in result
    if expected:
        cfg=result["selective_recovery"]
        assert (cfg.recovery_mode,cfg.selection_mode,cfg.row_fraction)==("hidden_residual_int8","query_attention",.5)
        assert cfg.max_offline_kv_bytes==1024
    else:assert "selective_recovery" not in result


def test_factory_rejects_storage_mode_drift():
    factory=bench.arm_factory(lambda *a,**k:k,SimpleNamespace,arm="redknot_private",
        candidate=plan()["candidate"],cache_bytes=1024,page_bytes=256,receipt={})
    with pytest.raises(ValueError):factory(None,None,capacity_pages=bench.TOTAL_BYTES//256,share_local=True)


@pytest.mark.parametrize("arm",bench.ARMS)
def test_factory_rejects_metadata_override(arm):
    factory=bench.arm_factory(lambda *a,**k:k,SimpleNamespace,arm=arm,
        candidate=plan()["candidate"],cache_bytes=1024,page_bytes=256,receipt={},metadata_only_alias_sources=True)
    with pytest.raises(ValueError):factory(None,None,capacity_pages=bench.TOTAL_BYTES//256,
        share_local=arm!="redknot_private",metadata_only_alias_sources=arm!="redknot_shared")


def overlap_service():
    return dict(records=[dict(request_id=x,document_ids=["doc"],status="completed") for x in ("a","b")],
        total_wall_seconds=5.,trace=[dict(event=e,seconds=t,request_id=r) for e,t,r in
            [("admitted",1.,"a"),("admitted",2.,"b"),("released",3.,"a"),("released",4.,"b")]])


def test_actual_not_requested_sharing_overlap():
    s=overlap_service();r=bench.overlap_receipt(s,source_bytes_by_document={"doc":128},share_local=True)
    assert r["max_same_document_owners"]==2
    assert r["any_same_document_overlap_seconds"]==1
    assert r["seconds_by_max_same_document_owners"]=={"0":2.,"1":2.,"2":1.}
    assert r["concurrent_duplicate_source_bytes_upper_bound"]==128
    assert r["all_traced_owners_released"]


@pytest.mark.parametrize("mutation",["unknown","missing","double","clock"])
def test_bad_trace_rejected(mutation):
    s=overlap_service()
    if mutation=="unknown":s["trace"][0]["request_id"]="x"
    if mutation=="missing":s["trace"]=[]
    if mutation=="double":s["trace"][1]["request_id"]="a"
    if mutation=="clock":s["trace"][1]["seconds"]=.5
    with pytest.raises(ValueError):bench.overlap_receipt(s)


def test_zero_accuracy_does_not_filter_throughput():
    rows=[row(a,seconds=10-i) for i,a in enumerate(bench.ARMS)]
    s=bench.summarize(rows,plan())
    assert s["comparisons"][-1]["qps_ratio"]==1.25
    assert s["comparisons"][-1]["token_identical"]==32
    assert s["arms"][bench.ARMS[0]]["descriptive_all_prefix_quality"]=={"f1":0.,"em":0.}
    assert s["quality_gate_applied"] is False
    assert s["comparisons"][1]["storage_ablation_outputs_match"] is True


def test_private_cow_mismatch_is_visible_without_score_filter():
    rows=[row(a) for a in bench.ARMS]
    rows[-1]["service"]["records"][0].update(text="different",output_ids=[4])
    comparison=bench.summarize(rows,plan())["comparisons"][1]
    assert comparison["storage_ablation_outputs_match"] is False
    assert comparison["exact_text_and_ids_identical"]==31
    assert comparison["qps_ratio"]==1.


@pytest.mark.parametrize("failure",["cleanup","fatal","audit","incomplete","repeat","eos"])
def test_technical_failure_never_has_headline(failure):
    rows=[row(a) for a in bench.ARMS];r=rows[-1]
    if failure=="cleanup":r["cleanup_status"]="failed"
    if failure=="fatal":r["fatal"]=True
    if failure=="audit":r["audit_errors"]=["budget"]
    if failure=="incomplete":r["service"]["completed_requests"]=31
    if failure=="repeat":r["repeat"]=1
    if failure=="eos":r["service"]["ignore_eos"]=True
    s=bench.summarize(rows,plan())
    assert s["comparisons"][-1]["qps_ratio"] is None
    assert len(s["arm_statuses"])==3


def test_diagnostic_has_no_uninstrumented_headline():
    s=bench.summarize([row(a) for a in bench.ARMS],plan(publication_diagnostics=True))
    assert s["diagnostic_only"]
    assert all(x["qps_ratio"] is None for x in s["comparisons"])
    assert s["arms"][bench.ARMS[0]]["observed_qps"]==3.2


@pytest.mark.parametrize("metadata",[False,True])
def test_prepare_cli_never_loads_model_and_refuses_overwrite(tmp_path,metadata):
    manifest=tmp_path/"manifest.json";manifest.write_text(json.dumps(workload()))
    args=["--model",str(tmp_path/"absent-model"),"--manifest",str(manifest),"--output",str(tmp_path/"new"),
          "--item-key",bench.ITEM_KEYS[0],"--global-prefix-layers","0","--concurrency","16","--prepare-only"]
    if metadata:args.append("--metadata-only-alias-sources")
    assert bench.main(args)==0
    frozen=json.loads((tmp_path/"new/frozen_cow_plan.json").read_text())
    assert frozen["case"]["requests"]==32
    assert frozen.get("metadata_only_alias_sources",False) is metadata
    with pytest.raises(FileExistsError):bench.main(args)
