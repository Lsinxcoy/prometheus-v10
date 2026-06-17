"""Cross-file interface verification for V10 (Rule J)."""
from prometheus_v10.coevolution_layer import CoEvolutionLayer, FailureAnalysis
from prometheus_v10.anti_pattern import AntiPatternMemory, FailureSignature, CausalAttribution
from prometheus_v10.ecosystem import SkillEcosystem
from prometheus_v10.tool_lifecycle import ProposalBundle, LifecycleEdit, LifecycleOp
from prometheus_v10.preflection import PreflectionEngine, RuleBank
from prometheus_v10.speculative_evolution import SpeculativeEvolver, ForkController
from prometheus_v10.skill_eval import SkillEvaluator
from prometheus_v10.workflow_memory import WorkflowMemory
from prometheus_v10.benchmark_adapter import BenchmarkAdapter
from prometheus_v10.equilibrium import EquilibriumGuard

# 1. CoEvolutionLayer ↔ AntiPatternMemory ↔ SkillEcosystem
eco = SkillEcosystem()
apm = AntiPatternMemory()
cel = CoEvolutionLayer(ecosystem=eco, anti_pattern=apm)

sig = FailureSignature(failure_type="test", component="c", context_keywords=["k"])
attr = CausalAttribution(root_cause="test", affected_skills=["s"])
apm.record_failure(sig, attr)
apm.record_failure(sig, attr)

analysis = cel.analyze_failure({
    "error_type": "tool_error",
    "error_message": "test error",
    "affected_skills": ["s"],
    "affected_tools": ["c"],
})
bundle = cel.propose_bundle(analysis)

assert isinstance(bundle, ProposalBundle), f"Expected ProposalBundle, got {type(bundle)}"
assert all(isinstance(e, LifecycleEdit) for e in bundle.tool_edits), "tool_edits type mismatch"
print("✅ CoEvolution ↔ AntiPattern ↔ Ecosystem: OK")

# 2. PreflectionEngine ↔ RuleBank
pe = PreflectionEngine()
pe.rule_bank.add_rule("test", ["k1"], "prefer", "search", 0.7)
pe.rule_bank.increment_episode()
result = pe.prelect("search", ["k1"])
assert hasattr(result, 'predicted_success'), "PreflectionResult missing predicted_success"
assert hasattr(result, 'recommendation'), "PreflectionResult missing recommendation"
print("✅ PreflectionEngine ↔ RuleBank: OK")

# 3. Ecosystem full chain
eco2 = SkillEcosystem()
eco2.record_execution("s1", "qa", 0.8, 0.5, ["s2"])
eco2.record_execution("s2", "qa", 0.7, 0.5, ["s1"])
score = eco2.score_for_retrieval("s1", 0.8)
assert 0.0 <= score <= 1.0, f"score out of range: {score}"
print("✅ SkillEcosystem full chain: OK")

# 4. WorkflowMemory full chain
wm = WorkflowMemory()
tpl = wm.store_template("Test", "HR", [
    {"description": "Step A", "action": "a"},
    {"description": "Step B", "action": "b", "condition": "flag", "branch": "if_true"},
], conditions=[{"condition": "flag"}])
wf = wm.resolve_workflow(tpl.template_id, {"flag": True})
assert wf is not None, "resolve_workflow returned None"
assert len(wf.resolved_steps) == 2, f"Expected 2 steps, got {len(wf.resolved_steps)}"
print("✅ WorkflowMemory full chain: OK")

# 5. EquilibriumGuard full chain
eg = EquilibriumGuard(epsilon=0.5)
state = eg.record_round({"a": 0.5, "b": 0.5}, 1.0)
assert hasattr(state, 'nash_epsilon'), "EquilibriumState missing nash_epsilon"
print("✅ EquilibriumGuard full chain: OK")

# 6. SkillEvaluator full chain
se = SkillEvaluator()
se.generate_tasks("test_skill", "search workflow")
result = se.evaluate("test_skill", {"output": "success"}, {"output": "fail"})
utility = se.compute_utility("test_skill")
assert utility.skill_name == "test_skill"
print("✅ SkillEvaluator full chain: OK")

# 7. BenchmarkAdapter full chain
ba = BenchmarkAdapter()
for i in range(12):
    ba.record(f"t{i}", "qa", passed=i%2==0, score=0.5+i*0.03, internal_fitness=0.4+i*0.04)
report = ba.compute_correlation()
assert report.sample_size == 12
print("✅ BenchmarkAdapter full chain: OK")

# 8. SpeculativeEvolver full chain
ev = SpeculativeEvolver()
candidates = ev.process_reasoning_chunk("therefore we choose alpha=0.5", layer=2)
print(f"✅ SpeculativeEvolver full chain: OK ({len(candidates)} candidates)")

print("\n🎉 All cross-file interface checks passed!")
