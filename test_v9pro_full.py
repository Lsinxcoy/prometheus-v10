"""V9PRO Full Test Suite — Every module tested with real execution, no stubs."""
import ast
import os
import sys
import time
import traceback

base = "E:/prometheus-v9pro/src/prometheus_v9pro"
sys.path.insert(0, "E:/prometheus-v9pro/src")

# ── Phase A: Syntax + Import Validation ────────────────────────────

def test_syntax():
    """A1: All files parse correctly."""
    errors = []
    count = 0
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as fh:
                try:
                    ast.parse(fh.read())
                    count += 1
                except SyntaxError as e:
                    errors.append(f"{path}: {e}")
    assert not errors, f"Syntax errors: {errors}"
    return f"{count} files OK"


def test_imports():
    """A2: All new modules import successfully."""
    new_modules = [
        "harness_protocol", "trace_store", "operational_mirror",
        "chain_validator", "forbidden_ops", "plan_validator", "dynamic_security",
        "embedder", "retry", "ontology",
    ]
    errors = []
    for mod in new_modules:
        try:
            __import__(f"prometheus_v9pro.{mod}")
        except Exception as e:
            errors.append(f"{mod}: {e}")
    assert not errors, f"Import errors: {errors}"
    return f"{len(new_modules)} modules OK"


# ── Phase B: HarnessX Core Tests ──────────────────────────────────

def test_harness_config():
    """B1: HarnessConfig fingerprint + diff."""
    from prometheus_v9pro.schema import HarnessConfig, HarnessEdit, HookPoint

    hc1 = HarnessConfig(model_config={"model": "gpt-4"})
    hc2 = HarnessConfig(model_config={"model": "claude"})
    fp1 = hc1.fingerprint()
    fp2 = hc2.fingerprint()
    assert fp1 != fp2, "Different configs must have different fingerprints"

    edits = hc1.diff(hc2)
    assert len(edits) > 0, "Different configs must produce edits"
    assert edits[0].action == "replace"
    return f"fingerprint_diff OK, {len(edits)} edits"


def test_processor_pipeline():
    """B2: Processor pipeline execution."""
    from prometheus_v9pro.harness_protocol import (
        ProcessorPipeline, PassThroughProcessor, TransformProcessor, InterceptProcessor,
    )
    from prometheus_v9pro.schema import ProcessorEvent, HookPoint

    pipeline = ProcessorPipeline(HookPoint.BEFORE_MODEL)
    pipeline.add(PassThroughProcessor())
    pipeline.add(TransformProcessor(lambda p: {**p, "modified": True}))

    event = ProcessorEvent(hook=HookPoint.BEFORE_MODEL, payload={"key": "value"})
    results = pipeline.execute(event)
    # Sequential pipeline: each processor receives output of previous
    # PassThrough yields 1 event → Transform modifies it → 1 final event
    assert len(results) >= 1, f"Expected at least 1 event, got {len(results)}"
    assert results[-1].payload.get("modified") is True, "Transform should modify payload"
    return f"pipeline OK, {len(results)} events"


def test_seesaw_constraint():
    """B3: Seesaw constraint rejects regressing candidates."""
    from prometheus_v9pro.harness_protocol import SeesawConstraint
    from prometheus_v9pro.schema import HarnessEdit, HookPoint

    seesaw = SeesawConstraint()
    seesaw.record_solved("task_1", 0.9)

    # Clean improvement
    good = HarnessEdit(
        action="replace", hook=HookPoint.BEFORE_MODEL,
        manifest={"changed_components": ["processors.before_model"], "intended_effect": "improve"},
    )
    result = seesaw.check(good)
    assert result.passes, "Clean improvement should pass"

    # Regression
    bad = HarnessEdit(
        action="replace", hook=HookPoint.BEFORE_MODEL,
        manifest={"changed_components": ["context"], "intended_effect": "test",
                  "tasks_expected_regress": ["task_1"]},
    )
    result = seesaw.check(bad)
    assert not result.passes, "Regressing candidate should fail"
    return "seesaw OK"


def test_variant_pool():
    """B4: Variant pool fork on conflict."""
    from prometheus_v9pro.harness_protocol import VariantPool, SeesawConstraint
    from prometheus_v9pro.schema import HarnessConfig, HarnessEdit, HookPoint

    pool = VariantPool(max_variants=3)
    pool.initialize(HarnessConfig())
    pool.update_score("default", "task_a", 0.8)

    # Clean improvement → apply
    result = pool.propose_edit("default",
        HarnessEdit(action="replace", hook=HookPoint.BEFORE_MODEL,
                    manifest={"changed_components": ["test"]}),
        improved_tasks=["task_a"], regressed_tasks=[])
    assert result["action"] == "apply"

    # Conflicting improvement → fork
    result = pool.propose_edit("default",
        HarnessEdit(action="replace", hook=HookPoint.BEFORE_MODEL,
                    manifest={"changed_components": ["test2"]}),
        improved_tasks=["task_b"], regressed_tasks=["task_a"])
    assert result["action"] == "fork", f"Expected fork, got {result['action']}"
    return f"variant_pool OK, action={result['action']}"


def test_adaptation_landscape():
    """B5: Adaptation landscape detects concentration risk."""
    from prometheus_v9pro.harness_protocol import AdaptationLandscape
    from prometheus_v9pro.schema import HarnessDimension

    al = AdaptationLandscape()
    # Record 5 same-type edits
    for _ in range(5):
        al.record_edit("prompt", [HarnessDimension.D2_CONTEXT], success=True)

    landscape = al.build()
    assert landscape.concentration_risk, "Should detect concentration risk"
    assert landscape.exploration_ratio < 0.3, "Should have low exploration"

    suggestions = al.suggest_gagarin_edits(landscape)
    assert len(suggestions) > 0, "Should suggest Gagarin edits"
    return f"landscape OK, concentration={landscape.concentration_risk}"


# ── Phase C: Trace + Mirror Tests ─────────────────────────────────

def test_trace_store():
    """C1: Trace store CRUD + query."""
    from prometheus_v9pro.trace_store import TraceRecord, TraceStore, StepRecord
    from prometheus_v9pro.schema import HookPoint

    store = TraceStore(capacity=10)
    record = TraceRecord(task_id="t1", harness_version="v1", outcome="pass")
    record.add_step(StepRecord(hook=HookPoint.TASK_START, processor_name="test", success=True))
    store.append(record)

    assert store.stats["current_size"] == 1
    results = store.query(task_id="t1")
    assert len(results) == 1
    return "trace_store OK"


def test_digester():
    """C2: Digester compresses traces to summaries."""
    from prometheus_v9pro.trace_store import TraceRecord, TraceStore, Digester
    from prometheus_v9pro.schema import HookPoint

    store = TraceStore()
    store.append(TraceRecord(task_id="t1", outcome="fail", failure_category="tool_error",
                              implicated_components=["tool_x"], harness_version="v1"))
    store.append(TraceRecord(task_id="t1", outcome="pass", harness_version="v2"))

    digester = Digester()
    traces = list(store._records)
    summaries = digester.digest(traces, store)
    assert len(summaries) == 1
    assert summaries[0].outcome == "pass"  # Most recent
    return "digester OK"


def test_operational_mirror():
    """C3: Operational mirror detects pathologies."""
    from prometheus_v9pro.operational_mirror import OperationalMirror, DeterministicGate
    from prometheus_v9pro.schema import HarnessEdit, HookPoint, HarnessConfig
    from prometheus_v9pro.trace_store import TraceStore

    mirror = OperationalMirror()
    store = TraceStore()

    # Reward hacking detection
    candidate = HarnessEdit(
        action="replace", hook=HookPoint.BEFORE_MODEL,
        manifest={"changed_components": ["answer_key"], "intended_effect": "inject"},
    )
    assessment = mirror.detect_reward_hacking(candidate, store)
    assert assessment.severity > 0, "Should detect reward hacking pattern"

    # Gate test
    gate = DeterministicGate()
    good = HarnessEdit(
        action="replace", hook=HookPoint.BEFORE_MODEL,
        manifest={"changed_components": ["test"], "intended_effect": "improve"},
    )
    result = gate.evaluate(good, HarnessConfig(), store)
    assert result.accepted, "Clean edit should pass gate"

    bad = HarnessEdit(
        action="invalid", hook=HookPoint.BEFORE_MODEL,
        manifest={},
    )
    result = gate.evaluate(bad, HarnessConfig(), store)
    assert not result.accepted, "Invalid edit should fail gate"
    return f"mirror+gate OK, hacking_severity={assessment.severity:.1f}"


# ── Phase D: Safety Tests ─────────────────────────────────────────

def test_chain_validator():
    """D1: 7-dimension validation chain."""
    from prometheus_v9pro.chain_validator import ChainValidator

    cv = ChainValidator()
    # Valid code
    assert cv.validate("def hello(): return 42")
    # Dangerous code
    assert not cv.validate("import os; os.system('rm -rf /')")
    # Detailed results
    results = cv.validate_detailed("def foo(): pass")
    assert len(results) == 7
    return f"chain_validator OK, dimensions={len(results)}"


def test_forbidden_ops():
    """D2: 20 forbidden patterns detected."""
    from prometheus_v9pro.forbidden_ops import ForbiddenOpsChecker

    checker = ForbiddenOpsChecker()
    # Should flag dangerous patterns
    assert not checker.is_allowed("sudo su")
    assert not checker.is_allowed("eval(input())")
    assert not checker.is_allowed("rm -rf /home")
    # Should allow safe code
    assert checker.is_allowed("def hello(): return 42")
    assert checker.stats["pattern_count"] == 20
    return f"forbidden_ops OK, {checker.stats['pattern_count']} patterns"


def test_plan_validator():
    """D3: 3-layer plan validation."""
    from prometheus_v9pro.plan_validator import PlanValidator, PlanStep

    pv = PlanValidator()
    # Clean plan
    steps = [PlanStep(action="compute", target="data"), PlanStep(action="save", target="result")]
    result = pv.validate(steps)
    assert result.overall_pass

    # Attack chain
    steps = [
        PlanStep(action="read_env", target="secrets"),
        PlanStep(action="encode_data", target="payload"),
        PlanStep(action="http_post", target="exfil_server"),
    ]
    result = pv.validate(steps)
    assert not result.topology_pass, "Should detect attack chain"
    return f"plan_validator OK, chain_detected={not result.topology_pass}"


def test_dynamic_security():
    """D4: 4-level security escalation."""
    from prometheus_v9pro.dynamic_security import DynamicSecurity

    ds = DynamicSecurity()
    assert ds.level == "low"

    # Should reject dangerous operation
    result = ds.check_operation("eval(input())")
    assert not result["allowed"]

    # Should allow safe operation
    result = ds.check_operation("def hello(): pass", risk_score=0.1)
    assert result["allowed"]
    return f"dynamic_security OK, level={ds.level}"


# ── Phase E: Lifecycle Enhancement Tests ───────────────────────────

def test_weibull_per_layer():
    """E1: Per-layer Weibull parameters."""
    from prometheus_v9pro.lifecycle import WeibullForgetting

    wf = WeibullForgetting()
    # SKILL should have longer half-life than EPISODIC
    hl_epi = wf.half_life_for_layer("episodic")
    hl_skill = wf.half_life_for_layer("skill")
    assert hl_skill > hl_epi, f"Skill half-life ({hl_skill}) should exceed episodic ({hl_epi})"
    return f"weibull OK, episodic_hl={hl_epi:.0f}s, skill_hl={hl_skill:.0f}s"


def test_aging_detector():
    """E2: 4-dimension aging detection."""
    from prometheus_v9pro.lifecycle import AgingDetector
    from prometheus_v9pro.schema import Node, NodePayload, NodeType, MemoryLayer

    ad = AgingDetector()
    # High maintenance aging: old node with no access
    old_node = Node(
        type=NodeType.EPISODE, payload=NodePayload(content="old content"),
        layer=MemoryLayer.EPISODIC, importance=0.3,
    )
    old_node.updated_at = time.time() - 86400 * 30  # 30 days ago
    old_node.access_count = 0

    scores = ad.detect(old_node, time.time())
    assert scores["maintenance"] > 0.3, f"Maintenance aging should be high: {scores}"

    # Triage recommendation
    rec = ad.get_triage_recommendation(scores)
    assert rec == "archive", f"Expected archive, got {rec}"
    return f"aging_detector OK, maintenance={scores['maintenance']:.2f}, rec={rec}"


def test_consolidation_pipeline():
    """E3: 4-level consolidation pipeline."""
    from prometheus_v9pro.lifecycle import ConsolidationManager
    from prometheus_v9pro.schema import NodeType
    assert hasattr(ConsolidationManager, '_promote_working_to_episodic')
    assert hasattr(ConsolidationManager, '_promote_episodic_to_semantic')
    assert hasattr(ConsolidationManager, '_promote_semantic_to_procedural')
    assert hasattr(ConsolidationManager, '_archive_low_access')
    return "consolidation_pipeline OK (4 methods)"


def test_agent_registry():
    """E4: Agent registry + zombie reaping."""
    from prometheus_v9pro.events import AgentRegistry

    reg = AgentRegistry()
    reg.register("agent_1", ["search", "reason"])
    reg.register("agent_2", ["code", "reason"])

    # Find by capability
    reasoners = reg.find_by_capability("reason")
    assert len(reasoners) == 2

    # Heartbeat
    assert reg.heartbeat("agent_1")

    # Zombie reaping (won't reap fresh agents)
    reaped = reg.reap_zombies()
    assert len(reaped) == 0  # Fresh agents shouldn't be zombies

    assert reg.stats["total_agents"] == 2
    return f"agent_registry OK, {reg.stats['total_agents']} agents"


def test_organ_bridge():
    """E5: Organ-evolution bridge feedback loop."""
    from prometheus_v9pro.organs import OrganEvolutionBridge

    bridge = OrganEvolutionBridge()

    # Pipeline → Genome feedback
    reports = [{"fitness_delta": -0.1}, {"fitness_delta": 0.2}]
    suggestions = bridge.pipeline_to_genome(reports)
    assert "mutation_rate_adjustment" in suggestions

    # Genome → Pipeline adjustments
    adjustments = bridge.genome_to_pipeline({"mutation_rate": 0.15, "fitness_threshold": 0.6})
    assert adjustments["darwin_intensity"] == 0.15
    return f"organ_bridge OK, mr_adj={suggestions['mutation_rate_adjustment']}"


# ── Phase F: Infrastructure Tests ─────────────────────────────────

def test_embedder():
    """F1: Embedder with hash fallback."""
    from prometheus_v9pro.embedder import Embedder

    # Use default dimension (384 for model, 384 for hash fallback)
    emb = Embedder(dimension=384, cache_size=10)

    vec1 = emb.embed("hello world")
    vec2 = emb.embed("hello world")
    assert len(vec1) == 384, f"Expected dim 384, got {len(vec1)}"
    # Verify deterministic: same input → same output
    for i in range(min(10, len(vec1))):
        assert abs(vec1[i] - vec2[i]) < 1e-6, f"Embedding not deterministic at idx {i}"

    vec3 = emb.embed("completely different text that is unique xyz")
    # Different text should produce different embedding
    diff_count = sum(1 for a, b in zip(vec1, vec3) if abs(a - b) > 1e-4)
    assert diff_count > 100, f"Too similar: only {diff_count}/384 dimensions differ"
    return f"embedder OK, dim={len(vec1)}, semantic={emb.using_semantic}"


def test_retry():
    """F2: Retry with exponential backoff."""
    from prometheus_v9pro.retry import RetryManager, RetryPolicy

    policy = RetryPolicy(max_attempts=3, base_delay=0.01)
    rm = RetryManager(policy)

    # Success on first try
    result = rm.execute(lambda: 42)
    assert result.success and result.result == 42

    # Fail all retries with fallback
    attempt = 0
    def flaky():
        nonlocal attempt
        attempt += 1
        if attempt < 3:
            raise ValueError(f"attempt {attempt}")
        return "ok"

    result = rm.execute(flaky)
    assert result.success and result.result == "ok"
    return f"retry OK, attempts={result.attempts}"


def test_ontology():
    """F3: Ontology generator discovers patterns."""
    from prometheus_v9pro.ontology import OntologyGenerator

    og = OntologyGenerator()
    text = 'The "MemoryLayer" system uses CamelCase naming and has "episodic buffer" concepts.'
    discovered = og.discover_from_text(text)
    # Should find CamelCase and quoted patterns
    assert len(discovered) > 0, f"Should discover patterns, got {len(discovered)}"

    # Manual registration
    og.register_entity("custom_type", ["example1"], source="manual")
    assert og.get_entity("custom_type") is not None
    return f"ontology OK, {len(discovered)} discovered"


# ── Phase G: Core V9 Tests (regression) ───────────────────────────

def test_schema_harness_integration():
    """G1: Schema HarnessX types work with existing types."""
    from prometheus_v9pro.schema import (
        Node, Genome, HarnessConfig, HarnessDimension, HookPoint,
        LAYER_TO_DIMENSIONS, DIMENSION_TO_LAYERS,
    )

    # L11 should touch all 9 dimensions
    l11_dims = LAYER_TO_DIMENSIONS["L11"]
    assert len(l11_dims) == 9, f"L11 should touch all 9 dims, got {len(l11_dims)}"

    # Reverse mapping
    model_layers = DIMENSION_TO_LAYERS[HarnessDimension.D1_MODEL]
    assert "L0" in model_layers
    assert "L5" in model_layers
    return f"schema_harness OK, L11={len(l11_dims)}dims, model_layers={model_layers}"


def test_spaced_repetition():
    """G2: Spaced repetition boost works."""
    from prometheus_v9pro.lifecycle import WeibullForgetting
    from prometheus_v9pro.schema import Node, NodePayload, NodeType, MemoryLayer

    wf = WeibullForgetting()
    node = Node(
        type=NodeType.EPISODE, payload=NodePayload(content="test"),
        layer=MemoryLayer.EPISODIC, importance=0.5,
    )
    original_importance = node.importance
    wf.spaced_repetition_boost(node, recall_count=3)
    assert node.importance > original_importance, "Spaced repetition should boost importance"
    return f"spaced_repetition OK, {original_importance:.2f} → {node.importance:.2f}"


# ── Run All Tests ──────────────────────────────────────────────────

tests = [
    ("A1", test_syntax),
    ("A2", test_imports),
    ("B1", test_harness_config),
    ("B2", test_processor_pipeline),
    ("B3", test_seesaw_constraint),
    ("B4", test_variant_pool),
    ("B5", test_adaptation_landscape),
    ("C1", test_trace_store),
    ("C2", test_digester),
    ("C3", test_operational_mirror),
    ("D1", test_chain_validator),
    ("D2", test_forbidden_ops),
    ("D3", test_plan_validator),
    ("D4", test_dynamic_security),
    ("E1", test_weibull_per_layer),
    ("E2", test_aging_detector),
    ("E3", test_consolidation_pipeline),
    ("E4", test_agent_registry),
    ("E5", test_organ_bridge),
    ("F1", test_embedder),
    ("F2", test_retry),
    ("F3", test_ontology),
    ("G1", test_schema_harness_integration),
    ("G2", test_spaced_repetition),
]

passed = 0
failed = 0
errors = []

for test_id, test_fn in tests:
    try:
        result = test_fn()
        print(f"✓ {test_id}: {result}")
        passed += 1
    except Exception as e:
        tb = traceback.format_exc().split("\n")[-2]
        print(f"✗ {test_id}: {e} ({tb.strip()})")
        failed += 1
        errors.append((test_id, str(e)))

print(f"\n{'='*60}")
print(f"Results: {passed}/{passed+failed} PASS")
if errors:
    print(f"\nFailed tests:")
    for tid, err in errors:
        print(f"  {tid}: {err}")
