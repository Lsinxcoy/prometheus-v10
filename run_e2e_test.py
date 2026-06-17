"""Prometheus V10 End-to-End Runtime Test — Make it actually run."""
import sys
import time
import traceback

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        traceback.print_exc()
        failed += 1

# ═══════════════════════════════════════
section("Phase 1: Life() 初始化")
# ═══════════════════════════════════════

life = None

def test_life_init():
    global life
    from prometheus_v10 import Life
    life = Life(db_path="data/v10_runtime_test.db")
    assert life.store is not None
    assert life.engine is not None
    assert life.safety is not None
    assert life.autonomy is not None

test("Life() 创建", test_life_init)

def test_life_status():
    status = life.status()
    assert "nodes" in status
    assert "generation" in status
    assert "fitness" in status
    print(f"    → {status}")

test("Life.status()", test_life_status)

# ═══════════════════════════════════════
section("Phase 2: 记忆系统 (remember/recall)")
# ═══════════════════════════════════════

def test_remember():
    nid = life.remember("V10 runtime test: ecosystem dynamics", node_type="note", layer="episodic", tags=["test", "v10"])
    assert nid, "remember should return node ID"
    print(f"    → node_id={nid}")

test("Life.remember()", test_remember)

def test_remember_multiple():
    for i, (content, layer) in enumerate([
        ("Search skill improved by 15%", "semantic"),
        ("Tool timeout pattern detected", "episodic"),
        ("Anti-pattern: retry loop on flaky API", "semantic"),
    ]):
        nid = life.remember(content, node_type="note", layer=layer, tags=["evolution"])
        print(f"    → [{layer}] {content[:40]}... id={nid[:8]}")

test("Life.remember() × 3", test_remember_multiple)

def test_recall():
    results = life.recall("ecosystem", limit=5)
    assert isinstance(results, list)
    print(f"    → found {len(results)} results")
    for r in results:
        print(f"      [{r['type']}] {r['content'][:50]}...")

test("Life.recall()", test_recall)

# ═══════════════════════════════════════
section("Phase 3: 进化引擎 (evolve)")
# ═══════════════════════════════════════

def test_evolve_one_step():
    results = life.evolve(steps=1)
    assert isinstance(results, list)
    assert len(results) == 1
    step = results[0]
    print(f"    → gen={step['generation']}, fitness={step['fitness']:.3f}")
    for layer in step['layers']:
        print(f"      L{layer['id']} {layer['name']}: {'✅' if layer['success'] else '❌'}")

test("Life.evolve(1)", test_evolve_one_step)

def test_evolve_multi_step():
    results = life.evolve(steps=3)
    assert len(results) == 3
    for r in results:
        print(f"    → gen={r['generation']}, fitness={r['fitness']:.3f}, "
              f"success={sum(1 for l in r['layers'] if l['success'])}/{len(r['layers'])}")

test("Life.evolve(3)", test_evolve_multi_step)

# ═══════════════════════════════════════
section("Phase 4: 安全治理 (safety + governance)")
# ═══════════════════════════════════════

def test_safety_check():
    result = life.safety.check("search", "def safe_function(): return 42")
    assert isinstance(result, dict)
    print(f"    → allowed={result.get('allowed', 'N/A')}")

test("SafetyManager.check() safe code", test_safety_check)

def test_safety_dangerous():
    result = life.safety.check("exec", "import os; os.system('rm -rf /')")
    print(f"    → allowed={result.get('allowed', 'N/A')} (should be False)")
    assert not result.get('allowed', True), "Dangerous code should be rejected"

test("SafetyManager.check() dangerous code", test_safety_dangerous)

def test_initiative():
    result = life.initiative.propose("read", "safe content for reading")
    if isinstance(result, dict):
        approved = result.get("approved", "N/A")
    else:
        approved = getattr(result, "approved", "N/A")
    print(f"    → approved={approved}, action=read")

test("InitiativeLayer.propose()", test_initiative)

# ═══════════════════════════════════════
section("Phase 5: CORAL 反思")
# ═══════════════════════════════════════

def test_coral():
    result = life.reflect("ecosystem integration", "success", insights=["LV dynamics stable"], mistakes=[])
    assert result["task"] == "ecosystem integration"
    print(f"    → task={result['task']}, outcome={result['outcome']}")

test("Life.reflect()", test_coral)

def test_dream():
    result = life.dream_cycle()
    assert "insights" in result
    print(f"    → insights={result['insights']}, integrated={result['integrated']}")

test("Life.dream_cycle()", test_dream)

# ═══════════════════════════════════════
section("Phase 6: V10 新模块集成")
# ═══════════════════════════════════════

def test_ecosystem():
    from prometheus_v10.ecosystem import SkillEcosystem
    eco = SkillEcosystem()
    eco.record_execution("search_skill", "qa", 0.85, 0.5, ["validate_skill"])
    eco.record_execution("validate_skill", "qa", 0.75, 0.5, ["search_skill"])
    score = eco.score_for_retrieval("search_skill", 0.8)
    print(f"    → ecosystem score for search_skill: {score:.3f}")
    print(f"    → stats: {eco.stats()}")

test("SkillEcosystem 集成", test_ecosystem)

def test_anti_pattern():
    from prometheus_v10.anti_pattern import AntiPatternMemory, FailureSignature, CausalAttribution, Remedy
    apm = AntiPatternMemory()
    sig = FailureSignature(failure_type="tool_timeout", component="search",
                           error_pattern="Timeout after 30s", context_keywords=["search", "timeout"])
    attr = CausalAttribution(root_cause="network_instability", affected_tools=["search"])
    r1 = apm.record_failure(sig, attr, [Remedy(action_type="wrap", target="search", description="Add retry")])
    r2 = apm.record_failure(sig, attr)
    print(f"    → vetted after 2nd: {r2.vetted}")
    should_veto, reason = apm.should_veto("delete", "search")
    print(f"    → should_veto(delete, search): {should_veto}")

test("AntiPatternMemory 集成", test_anti_pattern)

def test_preflection():
    from prometheus_v10.preflection import PreflectionEngine
    pe = PreflectionEngine()
    pe.rule_bank.add_rule("Avoid search in degraded mode", ["degraded", "search"],
                          "avoid", "search", 0.8)
    pe.rule_bank.increment_episode()
    result = pe.prelect("search", ["degraded", "search"])
    print(f"    → recommendation={result.recommendation}, success={result.predicted_success:.3f}")
    print(f"    → confidence={result.confidence:.3f}")

test("PreflectionEngine 集成", test_preflection)

def test_tool_lifecycle():
    from prometheus_v10.tool_lifecycle import ToolLifecycleManager, ToolSpec, LifecycleEdit, LifecycleOp
    tlm = ToolLifecycleManager()
    tlm.register_tool(ToolSpec(name="search", interface="query→results"))
    tlm.register_tool(ToolSpec(name="validate", interface="results→bool"))
    edit = LifecycleEdit(op=LifecycleOp.COMPOSE, source_tools=["search", "validate"])
    success, reason = tlm.apply_edit(edit)
    print(f"    → COMPOSE: success={success}, tools={list(tlm._tools.keys())}")
    assert tlm.get_tool("composed_search_validate") is not None

test("ToolLifecycleManager 集成", test_tool_lifecycle)

def test_speculative_evolution():
    from prometheus_v10.speculative_evolution import SpeculativeEvolver
    ev = SpeculativeEvolver()
    candidates = ev.process_reasoning_chunk(
        "We analyzed the code and therefore we choose alpha=0.5 as the optimal parameter", layer=3)
    print(f"    → forked {len(candidates)} candidates from reasoning chunk")
    stats = ev.stats()
    print(f"    → forks: {stats['forks']['total_forked']}, resources: {stats['resources']['utilization']:.1%}")

test("SpeculativeEvolver 集成", test_speculative_evolution)

def test_skill_eval():
    from prometheus_v10.skill_eval import SkillEvaluator
    se = SkillEvaluator()
    se.generate_tasks("search_skill", "search workflow for documents")
    result = se.evaluate("search_skill",
                         with_skill_output={"output": "success completed search results found"},
                         without_skill_output={"output": "error timeout"})
    print(f"    → instruction_following={result.instruction_following:.3f}, goal_completion={result.goal_completion:.3f}")
    print(f"    → marginal_value={result.marginal_value:.3f}")

test("SkillEvaluator 集成", test_skill_eval)

def test_workflow_memory():
    from prometheus_v10.workflow_memory import WorkflowMemory
    wm = WorkflowMemory()
    tpl = wm.store_template("Employee Onboarding", "HR", [
        {"description": "Create employee case", "action": "create_case", "tool": "hrms"},
        {"description": "Get Provost approval", "action": "approve", "tool": "email",
         "condition": "is_tenure_track", "branch": "if_true"},
        {"description": "Get Dept head approval", "action": "approve", "tool": "email",
         "condition": "is_tenure_track", "branch": "if_false"},
    ], conditions=[{"condition": "is_tenure_track"}])

    wf_tt = wm.resolve_workflow(tpl.template_id, {"is_tenure_track": True})
    wf_nott = wm.resolve_workflow(tpl.template_id, {"is_tenure_track": False})
    print(f"    → Tenure-track: {[s.description for s in wf_tt.resolved_steps]}")
    print(f"    → Non-tenure:    {[s.description for s in wf_nott.resolved_steps]}")

test("WorkflowMemory 集成", test_workflow_memory)

def test_benchmark_adapter():
    from prometheus_v10.benchmark_adapter import BenchmarkAdapter
    ba = BenchmarkAdapter()
    for i in range(15):
        ba.record(f"task_{i}", "qa", passed=(i % 2 == 0), score=0.4 + i*0.04, internal_fitness=0.3 + i*0.05)
    report = ba.compute_correlation()
    print(f"    → pearson_r={report.pearson_r:.3f}, sample_size={report.sample_size}")
    print(f"    → calibrated: {ba.is_calibrated()}")

test("BenchmarkAdapter 集成", test_benchmark_adapter)

def test_equilibrium():
    from prometheus_v10.equilibrium import EquilibriumGuard
    eg = EquilibriumGuard(epsilon=0.3)
    for i in range(5):
        utilities = {"taotie": 0.7 + i*0.01, "nuwa": 0.65 + i*0.01, "pangu": 0.6 + i*0.01}
        state = eg.record_round(utilities, 1.0 - i*0.05)
        print(f"    → round {i}: ε={state.nash_epsilon:.3f}, stable={state.is_stable}")
    print(f"    → at_equilibrium: {eg.is_at_equilibrium()}")

test("EquilibriumGuard 集成", test_equilibrium)

def test_coevolution():
    from prometheus_v10.coevolution_layer import CoEvolutionLayer
    cel = CoEvolutionLayer()
    analysis = cel.analyze_failure({
        "error_type": "tool_error",
        "error_message": "search returned KeyError: 'results'",
        "affected_skills": ["qa_skill", "code_skill"],
        "affected_tools": ["search"],
    })
    bundle = cel.propose_bundle(analysis)
    print(f"    → vertical={analysis.vertical}, horizontal={analysis.horizontal}")
    print(f"    → bundle: {len(bundle.skill_edits)} skill edits, {len(bundle.tool_edits)} tool edits")
    print(f"    → severity={analysis.severity:.2f}")

test("CoEvolutionLayer 集成", test_coevolution)

# ═══════════════════════════════════════
section("Phase 7: 跨模块联动 (真实工作流)")
# ═══════════════════════════════════════

def test_full_evolution_workflow():
    """完整进化工作流: evolve → detect failure → anti-pattern → co-evolution → preflection → re-evolve"""
    from prometheus_v10.ecosystem import SkillEcosystem
    from prometheus_v10.anti_pattern import AntiPatternMemory, FailureSignature, CausalAttribution
    from prometheus_v10.coevolution_layer import CoEvolutionLayer
    from prometheus_v10.preflection import PreflectionEngine
    from prometheus_v10.tool_lifecycle import ToolLifecycleManager, ToolSpec, LifecycleEdit, LifecycleOp

    # 1. Ecosystem tracks skill interactions
    eco = SkillEcosystem()
    eco.record_execution("search", "qa", 0.4, 0.5, ["validate"])  # low score!
    eco.record_execution("validate", "qa", 0.85, 0.5, ["search"])

    # 2. Search is declining — anti-pattern records the failure
    apm = AntiPatternMemory()
    sig = FailureSignature(failure_type="tool_error", component="search",
                           error_pattern="KeyError", context_keywords=["search", "qa"])
    attr = CausalAttribution(root_cause="tool_deficiency", affected_tools=["search"], affected_skills=["qa_skill"])
    apm.record_failure(sig, attr)
    apm.record_failure(sig, attr)  # vetted

    # 3. Co-evolution analyzes the failure and proposes a bundle
    cel = CoEvolutionLayer(ecosystem=eco, anti_pattern=apm)
    analysis = cel.analyze_failure({
        "error_type": "tool_error", "error_message": "search KeyError",
        "affected_skills": ["qa_skill"], "affected_tools": ["search"],
    })
    bundle = cel.propose_bundle(analysis)

    # 4. Preflection checks if the proposed action is safe
    pe = PreflectionEngine()
    pre_result = pe.prelect("edit", ["search", "tool_fix"])
    print(f"    → preflection: {pre_result.recommendation} (confidence={pre_result.confidence:.3f})")

    # 5. Tool lifecycle applies the bundle
    tlm = ToolLifecycleManager()
    tlm.register_tool(ToolSpec(name="search", interface="query→results"))
    if pre_result.recommendation != "avoid":
        success, reason = tlm.apply_bundle(bundle)
        print(f"    → bundle applied: {success}")
    else:
        print(f"    → bundle blocked by preflection: {pre_result.predicted_risks}")

    # 6. Check ecosystem stats after intervention
    declining = eco._lv_updater.get_declining_skills()
    print(f"    → declining skills: {declining}")
    print(f"    → anti-pattern stats: {apm.stats()}")

test("Full cross-module evolution workflow", test_full_evolution_workflow)

# ═══════════════════════════════════════
section("Phase 8: V10新模块注入Life")
# ═══════════════════════════════════════

def test_life_with_v10_modules():
    """V10 新模块应该能和 Life 协同工作"""
    from prometheus_v10.ecosystem import SkillEcosystem
    from prometheus_v10.preflection import PreflectionEngine
    from prometheus_v10.equilibrium import EquilibriumGuard

    # Life 的进化结果可以被 EquilibriumGuard 监控
    eg = EquilibriumGuard(epsilon=0.5)
    results = life.evolve(steps=2)
    for r in results:
        # 用 fitness 作为 potential value
        utilities = {"evolution": r['fitness']}
        state = eg.record_round(utilities, r['fitness'])
        print(f"    → gen={r['generation']} fitness={r['fitness']:.3f} ε={state.nash_epsilon:.3f} stable={state.is_stable}")

    # Preflection 可以预判 Life 的下一步行动
    pe = PreflectionEngine()
    pre = pe.prelect("evolve", ["evolution", "mutation"])
    print(f"    → preflection for next evolve: {pre.recommendation}")

test("Life × V10 modules 联动", test_life_with_v10_modules)

# ═══════════════════════════════════════
# Final Summary
# ═══════════════════════════════════════

print(f"\n{'='*60}")
print(f"  运行测试完成")
print(f"{'='*60}")
print(f"  ✅ 通过: {passed}")
print(f"  ❌ 失败: {failed}")
print(f"  总计: {passed + failed}")
if failed == 0:
    print(f"\n  🚀 Prometheus V10 全部跑通！")
else:
    print(f"\n  ⚠️ {failed} 个测试需要修复")
sys.exit(0 if failed == 0 else 1)
