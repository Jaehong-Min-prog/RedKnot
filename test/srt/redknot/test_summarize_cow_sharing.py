"""Small CPU fixtures for the independent COW screen/diagnostic report."""
import copy
import json
from pathlib import Path

import pytest

import benchmark_cow_sharing as bench
from benchmark_redknot_nonprefix import digest,scores
from benchmark_selective_recovery import memory_allocation
import summarize_cow_sharing as report
from test_benchmark_cow_sharing import workload


def fixture(tmp_path,*,diagnostic=False,metadata=False):
    expected=next(j for j in report.expected_jobs() if j["item_key"]==bench.ITEM_KEYS[0]
        and j["global_prefix_layers"]==0 and j["concurrency"]==16 and j["diagnostic_only"]==diagnostic)
    directory=tmp_path/expected["name"];directory.mkdir()
    source=tmp_path/"source";source.mkdir()
    hashes={}
    for name in report.SOURCE_FILES:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text("frozen: "+name)
        hashes["/remote/source/"+name]=digest(target.read_bytes())
    w=workload();p=bench.prepare_plan(w,bench.ITEM_KEYS[0],0,16,publication_diagnostics=diagnostic,
        metadata_only_alias_sources=metadata)
    cfg=dict(num_hidden_layers=64,num_key_value_heads=8,hidden_size=128,head_dim=128)
    cache=64*56*(128+4);page_bytes=131072
    raw=dict(schema="redknot-cow-sharing-v1",status="complete",plan=p,plan_sha256=digest(p),
        workload_sha256=digest(w),model_config=cfg,model_config_sha256=digest(cfg),source_hashes=hashes,
        runtime=dict(visible_cuda_device_count=1),results=[],
        hidden_cache_estimator=dict(document_token_lengths=p["case"]["chunk_lengths"],probe_layers=8,
            group_size=128,expected_bytes=cache,registered_corpora=1))
    for arm in bench.ARMS:
        candidate=arm!=bench.ARMS[0];local=435 if candidate else 0
        policies=[[dict(layer=l,kv_head=h,head_type="local" if l*8+h<local else "global") for h in range(8)] for l in range(64)]
        profile=dict(policies=policies,sha256=digest(policies),effective_local_ratio=local/512)
        budget=memory_allocation(bench.TOTAL_BYTES,cache if candidate else 0,page_bytes)
        local_bytes=16*local*page_bytes
        pool=dict(reserved_slab_bytes=budget["pool_reserved_bytes"],allocated_bytes=local_bytes,high_water_pages=100)
        stats=dict(pool=pool,documents=16,local_head_ratio=local/512,share_local=arm!="redknot_private",
            selective_recovery=dict(block_size=256,row_fraction=.5,probe_layers=8,max_offline_kv_bytes=cache,
                selection_mode="query_attention",recovery_mode="hidden_residual_int8") if candidate else None,
            offline_hidden_cache_bytes=cache if candidate else 0,offline_hidden_cache_limit_bytes=cache if candidate else 0,
            offline_hidden_cache_peak_bytes=cache if candidate else 0,
            offline_kv_cache_bytes=0,offline_kv_cache_limit_bytes=0,offline_kv_cache_peak_bytes=0)
        if metadata and arm=="redknot_shared":stats["metadata_only_alias_sources"]=True
        records=[];trace=[]
        for i in range(32):
            admission=dict(stage_timing_synchronizes=True,prefix_prefill_s=.1,online_recovery_s=.2,kv_publish_and_partition_s=.3,
                pool_after_admit=dict(live_pages=100+i,pinned_pages=90+i))
            if diagnostic:
                admission["publication_diagnostics"]=dict(schema="redknot-publication-stages-v1",diagnostic_only=True,
                    stage_seconds={k:.05 for k in report.PUBLICATION_STAGES},total_seconds=.25)
            records.append(dict(request_id=f"request:{i}",frozen_prefix_index=i,status="completed",max_new_tokens=32,
                document_ids=[f"case-0000:document:{j}" for j in range(16)],output_ids=[1,2],text="answer",
                accuracy=scores("answer",w["items"][0]["answers"]),output_text_sha256=digest(b"answer"),
                output_ids_sha256=digest([1,2]),exact_prompt_sha256=p["case"]["exact_prompt_sha256"][i],
                queue_wait_seconds=float(i),ttft_seconds=float(i)+2,service_ttft_seconds=2.,latency_seconds=float(i)+3,
                capacity_blocks=0,admission_stats=admission))
            trace.extend([dict(event="admitted",seconds=float(i),request_id=f"request:{i}"),
                          dict(event="released",seconds=float(i)+.5,request_id=f"request:{i}")])
        closed=dict(capacity_pages=budget["pool_capacity_pages"],free_pages=budget["pool_capacity_pages"],
            reserved_pages=0,live_pages=0,retiring_pages=0,pinned_pages=0,poisoned=False,high_water_pages=200)
        row=dict(arm=arm,repeat=0,status="complete",cleanup_status="complete",pool_after_close=closed,fatal=False,
            candidate=p["candidate"] if candidate else None,diagnostic_only=diagnostic,effective_profile=profile,
            memory_accounting=budget,service=dict(status="complete",errors=[],admission_policy="fcfs",ignore_eos=False,
                completed_requests=32,failed_requests=0,cancelled_requests=0,completed_output_tokens=64,total_wall_seconds=33.,
                peak_active_requests=1,max_forward_batch=1,records=records,trace=trace,
                engine_stats_before=stats,engine_stats_after=stats,
                stages_seconds=dict(admission=19.2,query_forward=1.,decode_forward=2.,release=3.,idle_wait=0.)))
        if metadata:row["metadata_only_alias_sources"]=arm=="redknot_shared"
        raw["results"].append(row)
    def save():
        for name,data in [("results.json",raw),("frozen_workload.json",w),("frozen_cow_plan.json",p)]:
            (directory/name).write_text(json.dumps(data))
    save()
    return directory,source,expected,raw,save


def test_full_fixture_reports_real_keys_and_parity(tmp_path):
    directory,source,expected,raw,_=fixture(tmp_path)
    result=report.summarize_job(directory,expected,source)
    assert result["status"]=="verified",result["errors"]
    assert len(result["verified_source_files"])==14
    arm=result["arms"]["redknot_shared"]
    assert arm["uninstrumented_qps"]==32/33
    assert arm["latency_seconds"]["service_ttft_seconds"]["p50"]==2
    assert arm["latency_seconds"]["ttft_seconds"]["p50"]==17.5
    assert arm["max_pool_high_water_pages"]==200
    assert arm["max_observed_live_pages"]==131
    assert arm["max_same_document_owners"]==1
    assert arm["recovery_publication_seconds"]["online_recovery_s"]==pytest.approx(6.4)
    assert result["comparisons"][1]["storage_ablation_outputs_match"] is True


def test_diagnostic_five_stages_never_enter_qps(tmp_path):
    directory,source,expected,raw,_=fixture(tmp_path,diagnostic=True)
    result=report.summarize_job(directory,expected,source)
    assert result["status"]=="verified",result["errors"]
    assert all(c["qps_ratio"] is None for c in result["comparisons"])
    d=result["arms"]["redknot_shared"]["publication_diagnostics"]
    assert d["requests_with_receipt"]==32
    assert d["stage_seconds"]["head_pack"]==pytest.approx(1.6)
    assert result["arms"]["redknot_shared"]["uninstrumented_qps"] is None


@pytest.mark.parametrize("diagnostic",[False,True])
def test_metadata_only_mode_verifies_actual_shared_stats(tmp_path,diagnostic):
    directory,source,expected,raw,_=fixture(tmp_path,metadata=True,diagnostic=diagnostic)
    result=report.summarize_job(directory,expected,source)
    assert result["status"]=="verified",result["errors"]
    assert result["metadata_only_alias_sources"] is True
    assert len(result["verified_source_files"])==14
    for arm in bench.ARMS:
        receipt=result["arms"][arm]["metadata_only_alias_sources_by_repeat"]
        enabled=arm=="redknot_shared"
        assert receipt==[dict(repeat=0,requested=enabled,before=enabled,after=enabled)]


@pytest.mark.parametrize("corruption",["missing_row","missing_actual","after","private","baseline","expected"])
def test_metadata_receipt_mismatch_fails_closed(tmp_path,corruption):
    directory,source,expected,raw,save=fixture(tmp_path,metadata=True)
    shared=raw["results"][-1]
    if corruption=="missing_row":shared.pop("metadata_only_alias_sources")
    if corruption=="missing_actual":shared["service"]["engine_stats_before"].pop("metadata_only_alias_sources")
    if corruption=="after":
        shared["service"]["engine_stats_after"]=copy.deepcopy(shared["service"]["engine_stats_after"])
        shared["service"]["engine_stats_after"]["metadata_only_alias_sources"]=False
    if corruption in ("private","baseline"):
        raw["results"][1 if corruption=="private" else 0]["service"]["engine_stats_before"]["metadata_only_alias_sources"]=True
    if corruption=="expected":expected["metadata_only_alias_sources"]=False
    save();result=report.summarize_job(directory,expected,source)
    assert result["status"]=="audit_failed"
    assert not result["comparisons"]


def test_unrequested_metadata_rejected_for_old_plan(tmp_path):
    directory,source,expected,raw,save=fixture(tmp_path)
    raw["results"][-1]["service"]["engine_stats_before"]["metadata_only_alias_sources"]=True
    save();assert report.summarize_job(directory,expected,source)["status"]=="audit_failed"


def test_runner_enrichment_rejects_unenabled_metadata(tmp_path):
    _,_,_,raw,_=fixture(tmp_path,metadata=True)
    row=raw["results"][-1]
    row["service"]["engine_stats_before"].pop("metadata_only_alias_sources")
    bench.enrich_row(row,plan=raw["plan"],budget=row["memory_accounting"],profile=row["effective_profile"],
        cache_bytes=raw["hidden_cache_estimator"]["expected_bytes"],page_bytes=131072)
    assert any("metadata-only" in error for error in row["audit_errors"])
    summary=bench.summarize(raw["results"],raw["plan"])
    assert summary["comparisons"][-1]["qps_ratio"] is None


@pytest.mark.parametrize("corruption",["source","score","output","budget","profile","diag","cache","cleanup"])
def test_corruption_cannot_produce_comparison(tmp_path,corruption):
    directory,source,expected,raw,save=fixture(tmp_path)
    row=raw["results"][-1];rec=row["service"]["records"][0]
    if corruption=="source":(source/report.SOURCE_FILES[0]).write_text("changed")
    if corruption=="score":rec["accuracy"]["f1"]=.5
    if corruption=="output":rec["output_ids"]=[999]
    if corruption=="budget":row["memory_accounting"]["pool_reserved_bytes"]+=1
    if corruption=="profile":row["effective_profile"]["sha256"]="wrong"
    if corruption=="diag":rec["admission_stats"]["publication_diagnostics"]={"diagnostic_only":True}
    if corruption=="cache":row["service"]["engine_stats_before"]["offline_kv_cache_bytes"]=1024
    if corruption=="cleanup":row["pool_after_close"]["pinned_pages"]=1
    save();result=report.summarize_job(directory,expected,source)
    assert result["status"]!="verified"
    assert all(c["qps_ratio"] is None for c in result["comparisons"])


def test_failed_arm_retained_without_qps(tmp_path):
    directory,source,expected,raw,save=fixture(tmp_path)
    raw["status"]="incomplete";raw["results"][-1].update(status="failed",error={"message":"actual failure"})
    save();result=report.summarize_job(directory,expected,source)
    assert result["status"]=="incomplete_or_failed"
    assert result["arms"]["redknot_shared"]["observed_repeat_statuses"][0]["error"]["message"]=="actual failure"
    assert all(c["qps_ratio"] is None for c in result["comparisons"])


def test_six_formal_and_two_diagnostic_always_present(tmp_path):
    source=tmp_path/"source";source.mkdir()
    result=report.summarize_matrix(tmp_path,source)
    assert len(result["screen_jobs"])==6 and len(result["diagnostic_jobs"])==2
    assert all(j["status"]=="missing" for j in result["screen_jobs"])
    assert result["independent_primary_qa_count"]==2


def test_unplanned_run_not_silently_lost(tmp_path):
    source=tmp_path/"source";source.mkdir();extra=tmp_path/"screen_extra";extra.mkdir();(extra/"results.json").write_text("{}")
    result=report.summarize_matrix(tmp_path,source)
    assert result["unplanned_results_retained_as_paths"]==[str(extra/"results.json")]


@pytest.mark.parametrize("value",["NaN","Infinity","1e309"])
def test_nonfinite_json_rejected(tmp_path,value):
    p=tmp_path/"x.json";p.write_text('{"number":'+value+'}')
    with pytest.raises(ValueError):report.read_json(p)


def test_atomic_output_refuses_overwrite(tmp_path):
    p=tmp_path/"report.json";report.write_new_json(p,{"first":True})
    with pytest.raises(FileExistsError):report.write_new_json(p,{"second":True})
    assert json.loads(p.read_text())=={"first":True}
    assert not list(tmp_path.glob(".cow-summary-*"))
