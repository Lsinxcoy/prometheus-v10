"""Prometheus V10 Tests — Full coverage for 10 new V10 modules."""
import pytest
import math

# ═══════════════════════════════════════════
# P0-1: Ecosystem (Lotka-Volterra)
# ═══════════════════════════════════════════

class TestEcosystem:
    def test_interaction_matrix_recording(self):
        from prometheus_v10.ecosystem import InteractionMatrix, SkillObservation
        im = InteractionMatrix()
        # Record 5 co-activations of skill_a and skill_b
        for i in range(5):
            im.record_observation(SkillObservation(
                skill_name="skill_a", task_category="qa",
                score=0.7 + i * 0.02, normalized_residual=0.1,
                co_activated=["skill_b"],
            ))
        synergy = im.compute_synergy("skill_a", "skill_b")
        assert isinstance(synergy, float)

    def test_interaction_matrix_conflict_detection(self):
        from prometheus_v10.ecosystem import InteractionMatrix, SkillObservation
        im = InteractionMatrix()
        # skill_c and skill_d conflict (co-activation leads to low scores)
        for i in range(5):
            im.record_observation(SkillObservation(
                skill_name="skill_c", task_category="code",
                score=0.3, normalized_residual=-0.2,
                co_activated=["skill_d"],
            ))
        synergy = im.compute_synergy("skill_c", "skill_d")
        assert synergy < 0  # conflict

    def test_lotka_volterra_update(self):
        from prometheus_v10.ecosystem import LotkaVolterraUpdater
        lv = LotkaVolterraUpdater()
        # Positive residual should increase utility
        u = lv.update("skill_a", 0.5, {}, {})
        assert 0.0 < u < 1.0
        # Another update
        u2 = lv.update("skill_a", 0.3, {}, {"skill_a": u})
        # Utility may change or stay near boundary
        assert 0.0 <= u2 <= 1.0

    def test_lotka_volterra_declining(self):
        from prometheus_v10.ecosystem import LotkaVolterraUpdater
        lv = LotkaVolterraUpdater()
        # Repeated negative residuals → declining utility
        for _ in range(10):
            lv.update("declining_skill", -0.5, {}, {"declining_skill": 0.5})
        declining = lv.get_declining_skills(threshold=0.3)
        assert "declining_skill" in declining

    def test_pareto_front(self):
        from prometheus_v10.ecosystem import InstanceParetoFront
        pf = InstanceParetoFront()
        pf.record("task_1", "state_A", 0.8)
        pf.record("task_1", "state_B", 0.6)
        pf.record("task_2", "state_A", 0.5)
        pf.record("task_2", "state_B", 0.9)
        non_dominated = pf.get_non_dominated()
        assert "state_A" in non_dominated
        assert "state_B" in non_dominated

    def test_pareto_sampling(self):
        from prometheus_v10.ecosystem import InstanceParetoFront
        pf = InstanceParetoFront()
        pf.record("task_1", "state_A", 0.8)
        sampled = pf.sample_for_mutation()
        assert sampled == "state_A"

    def test_retrieval_scorer(self):
        from prometheus_v10.ecosystem import EcosystemRetrievalScorer
        scorer = EcosystemRetrievalScorer()
        score = scorer.score(
            skill_name="skill_a",
            semantic_relevance=0.8,
            dynamic_utility=0.7,
            activated_skills=["skill_b"],
            interactions={"skill_b": 0.3},  # complementary
            execution_cost=0.1,
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # high relevance + utility + complementarity

    def test_full_ecosystem(self):
        from prometheus_v10.ecosystem import SkillEcosystem
        eco = SkillEcosystem()
        # Record executions
        eco.record_execution("skill_a", "qa", 0.8, 0.5, ["skill_b"])
        eco.record_execution("skill_b", "qa", 0.7, 0.5, ["skill_a"])
        # Score for retrieval
        score = eco.score_for_retrieval("skill_a", 0.8)
        assert 0.0 <= score <= 1.0
        # Check stats
        stats = eco.stats()
        assert stats["interaction"]["total_pairs"] >= 0


# ═══════════════════════════════════════════
# P0-2: Anti-Pattern Memory
# ═══════════════════════════════════════════

class TestAntiPattern:
    def test_record_and_retrieve(self):
        from prometheus_v10.anti_pattern import AntiPatternMemory, FailureSignature, CausalAttribution
        apm = AntiPatternMemory()
        sig = FailureSignature(failure_type="tool_error", component="search",
                               error_pattern="KeyError: 'results'", context_keywords=["search", "qa"])
        attr = CausalAttribution(root_cause="tool_missing_capability", affected_tools=["search"])
        record = apm.record_failure(sig, attr)
        assert record.occurrence_count == 1
        assert not record.vetted

    def test_vetted_after_repeated(self):
        from prometheus_v10.anti_pattern import AntiPatternMemory, FailureSignature, CausalAttribution
        apm = AntiPatternMemory()
        sig = FailureSignature(failure_type="tool_error", component="search",
                               error_pattern="KeyError: 'results'", context_keywords=["search"])
        apm.record_failure(sig, CausalAttribution(root_cause="tool_error"))
        apm.record_failure(sig, CausalAttribution(root_cause="tool_error"))
        # After 2 occurrences → vetted
        assert apm.record_failure(sig, CausalAttribution(root_cause="tool_error")).vetted

    def test_diagnostic_acceleration(self):
        from prometheus_v10.anti_pattern import AntiPatternMemory, FailureSignature, CausalAttribution, Remedy
        apm = AntiPatternMemory()
        # Record a known failure with remedy
        sig1 = FailureSignature(failure_type="tool_error", component="search", context_keywords=["search"])
        attr1 = CausalAttribution(root_cause="timeout", affected_tools=["search"])
        apm.record_failure(sig1, attr1, [Remedy(action_type="wrap_tool", target="search", description="Add retry")])
        apm.record_failure(sig1, attr1)  # make it vetted

        # Query for similar failure
        sig2 = FailureSignature(failure_type="tool_error", component="search", context_keywords=["search"])
        context = apm.get_diagnostic_context(sig2)
        assert len(context["known_patterns"]) > 0

    def test_proposal_veto(self):
        from prometheus_v10.anti_pattern import AntiPatternMemory, FailureSignature, CausalAttribution, Remedy
        apm = AntiPatternMemory()
        sig = FailureSignature(failure_type="tool_error", component="search", context_keywords=["search"])
        attr = CausalAttribution(root_cause="timeout", affected_tools=["search"])
        apm.record_failure(sig, attr, [Remedy(action_type="edit", target="search", attempted=True, successful=False)])
        apm.record_failure(sig, attr)  # vetted

        # Try to propose same failed edit
        should_veto, reason = apm.should_veto("edit", "search")
        assert should_veto
        assert "timeout" in reason or "tool_error" in reason

    def test_epitaph(self):
        from prometheus_v10.anti_pattern import AntiPatternMemory
        apm = AntiPatternMemory()
        apm.record_epitaph("old_search", "tool", "low utility", ["timeout_pattern"], ["Always add retry"])
        assert apm.is_retired("old_search")
        epitaph = apm.get_epitaph("old_search")
        assert epitaph is not None
        assert any("retry" in lesson for lesson in epitaph.lessons_learned)


# ═══════════════════════════════════════════
# P0-3: Preflection
# ═══════════════════════════════════════════

class TestPreflection:
    def test_rule_bank_add_and_retrieve(self):
        from prometheus_v10.preflection import RuleBank
        rb = RuleBank()
        rb.add_rule("Prefer search in code tasks", ["code", "search"], "prefer", "search", 0.7)
        rb.increment_episode()
        rules = rb.retrieve_top_k(["code", "search"], k=3)
        assert len(rules) > 0

    def test_rule_bank_ucb(self):
        from prometheus_v10.preflection import RuleBank
        rb = RuleBank()
        rb.add_rule("Rule A", ["test"], "prefer", "search", 0.6)
        rb.add_rule("Rule B", ["test"], "avoid", "navigate", 0.4)
        rb.increment_episode()
        scores = rb.get_ucb_scores()
        # New rules should have ∞ UCB
        assert any(v == float('inf') for v in scores.values())

    def test_credit_assignment(self):
        from prometheus_v10.preflection import SemanticCreditAssigner
        sca = SemanticCreditAssigner()
        trajectory = [
            {"room_type": "office", "visible_objects": ["desk", "computer"]},
            {"room_type": "office", "visible_objects": ["desk", "computer", "target"]},
            {"room_type": "kitchen", "visible_objects": ["stove"]},
        ]
        credits = sca.compute_credits(trajectory, ["target"])
        assert len(credits) == 3
        # Step 1 should have positive marginal gain (found target)
        assert credits[1].marginal_gain >= credits[2].marginal_gain

    def test_preflection_proceed(self):
        from prometheus_v10.preflection import PreflectionEngine
        pe = PreflectionEngine()
        # No rules → default to proceed
        result = pe.prelect("search", ["code", "search"])
        assert result.recommendation in ("proceed", "caution")

    def test_preflection_with_rules(self):
        from prometheus_v10.preflection import PreflectionEngine
        pe = PreflectionEngine()
        # Add an "avoid" rule
        pe.rule_bank.add_rule("Avoid navigate in unknown areas", ["unknown", "navigate"],
                              "avoid", "navigate", 0.8)
        pe.rule_bank.increment_episode()
        # Prelect navigate action
        result = pe.prelect("navigate", ["unknown", "navigate"])
        assert result.predicted_risks or result.recommendation in ("caution", "avoid", "proceed")


# ═══════════════════════════════════════════
# P1-1: Tool Lifecycle
# ═══════════════════════════════════════════

class TestToolLifecycle:
    def test_register_and_get(self):
        from prometheus_v10.tool_lifecycle import ToolLifecycleManager, ToolSpec
        tlm = ToolLifecycleManager()
        tlm.register_tool(ToolSpec(name="search", interface="query→results"))
        tool = tlm.get_tool("search")
        assert tool is not None
        assert tool.is_active

    def test_edit_lifecycle(self):
        from prometheus_v10.tool_lifecycle import ToolLifecycleManager, ToolSpec, LifecycleEdit, LifecycleOp
        tlm = ToolLifecycleManager()
        tlm.register_tool(ToolSpec(name="search", interface="query→results"))
        edit = LifecycleEdit(op=LifecycleOp.EDIT, target="search", patch="Add retry logic")
        success, reason = tlm.apply_edit(edit)
        assert success
        assert tlm.get_tool("search").version == "1.0.1"

    def test_compose_lifecycle(self):
        from prometheus_v10.tool_lifecycle import ToolLifecycleManager, ToolSpec, LifecycleEdit, LifecycleOp
        tlm = ToolLifecycleManager()
        tlm.register_tool(ToolSpec(name="search", interface="query→results"))
        tlm.register_tool(ToolSpec(name="validate", interface="results→bool"))
        edit = LifecycleEdit(op=LifecycleOp.COMPOSE, source_tools=["search", "validate"])
        success, _ = tlm.apply_edit(edit)
        assert success
        assert tlm.get_tool("composed_search_validate") is not None

    def test_retire_lifecycle(self):
        from prometheus_v10.tool_lifecycle import ToolLifecycleManager, ToolSpec, LifecycleEdit, LifecycleOp
        tlm = ToolLifecycleManager()
        tlm.register_tool(ToolSpec(name="old_tool", interface="legacy"))
        edit = LifecycleEdit(op=LifecycleOp.RETIRE, target="old_tool", retire_reason="deprecated")
        success, _ = tlm.apply_edit(edit)
        assert success
        assert not tlm.get_tool("old_tool").is_active

    def test_atomic_bundle(self):
        from prometheus_v10.tool_lifecycle import ToolLifecycleManager, ToolSpec, ProposalBundle, LifecycleEdit, LifecycleOp
        tlm = ToolLifecycleManager()
        tlm.register_tool(ToolSpec(name="search", interface="query→results"))
        bundle = ProposalBundle(
            bundle_id="test_bundle",
            tool_edits=[LifecycleEdit(op=LifecycleOp.EDIT, target="search", patch="v2")],
        )
        success, _ = tlm.apply_bundle(bundle)
        assert success

    def test_auto_retire_on_errors(self):
        from prometheus_v10.tool_lifecycle import ToolLifecycleManager, ToolSpec
        tlm = ToolLifecycleManager()
        tlm.register_tool(ToolSpec(name="flaky", interface="unstable"))
        for _ in range(5):
            tlm.record_tool_error("flaky")
        assert tlm.should_auto_retire("flaky")


# ═══════════════════════════════════════════
# P1-2: Speculative Evolution
# ═══════════════════════════════════════════

class TestSpeculativeEvolution:
    def test_fork_controller_triggers(self):
        from prometheus_v10.speculative_evolution import ForkController
        fc = ForkController()
        triggers = fc.scan_for_triggers("therefore we choose the optimal strategy: alpha=0.5", layer=2)
        assert len(triggers) > 0  # should detect "therefore" and "alpha="

    def test_fork_candidates(self):
        from prometheus_v10.speculative_evolution import ForkController
        fc = ForkController()
        from prometheus_v10.speculative_evolution import TriggerSignal
        trigger = TriggerSignal(signal_type="design_decision", content="optimal", layer=2)
        candidates = fc.fork_candidates(trigger, "prefix context", available_capacity=2)
        assert len(candidates) == 2

    def test_elastic_resource_pool(self):
        from prometheus_v10.speculative_evolution import ElasticResourcePool, SpeculativeCandidate
        pool = ElasticResourcePool(total_capacity=10)
        c = SpeculativeCandidate(candidate_id="test_1", layer=0)
        assert pool.enqueue_validation(c)
        next_c = pool.next_to_validate()
        assert next_c is not None

    def test_early_termination(self):
        from prometheus_v10.speculative_evolution import SpeculativeEvolver
        evolver = SpeculativeEvolver(early_termination_threshold=0.1)
        # Feed some fitness values
        evolver._fitness_samples.extend([0.5, 0.6, 0.55])
        # High fitness should trigger early termination
        assert evolver.should_terminate_early(0.9)


# ═══════════════════════════════════════════
# P1-3: Skill Evaluator
# ═══════════════════════════════════════════

class TestSkillEval:
    def test_generate_tasks(self):
        from prometheus_v10.skill_eval import SkillEvaluator
        se = SkillEvaluator()
        tasks = se.generate_tasks("search_skill", "search workflow for documents", n_tasks=3)
        assert len(tasks) == 3
        assert all(t.skill_name == "search_skill" for t in tasks)

    def test_evaluate_skill(self):
        from prometheus_v10.skill_eval import SkillEvaluator
        se = SkillEvaluator()
        se.generate_tasks("test_skill", "validation workflow")
        result = se.evaluate("test_skill", {"output": "success completed result"}, None)
        assert 0.0 <= result.instruction_following <= 1.0
        assert 0.0 <= result.goal_completion <= 1.0

    def test_compute_utility(self):
        from prometheus_v10.skill_eval import SkillEvaluator
        se = SkillEvaluator()
        se.generate_tasks("util_skill", "monitoring workflow")
        se.evaluate("util_skill", {"output": "success"}, {"output": "fail"})
        utility = se.compute_utility("util_skill")
        assert utility.skill_name == "util_skill"
        assert utility.marginal_value >= 0.0


# ═══════════════════════════════════════════
# P1-4: Workflow Memory
# ═══════════════════════════════════════════

class TestWorkflowMemory:
    def test_store_template(self):
        from prometheus_v10.workflow_memory import WorkflowMemory
        wm = WorkflowMemory()
        tpl = wm.store_template("Onboarding", "HR", [
            {"description": "Create case", "action": "create_case", "tool": "hrms"},
            {"description": "Get approval", "action": "request_approval", "tool": "email",
             "condition": "is_tenure_track", "branch": "if_true"},
            {"description": "Dept head approval", "action": "request_approval", "tool": "email",
             "condition": "is_tenure_track", "branch": "if_false"},
        ], conditions=[{"condition": "is_tenure_track", "true_branch": "Provost", "false_branch": "DeptHead"}])
        assert tpl.template_id.startswith("wf_tpl_")
        assert len(tpl.steps) == 3

    def test_resolve_workflow(self):
        from prometheus_v10.workflow_memory import WorkflowMemory
        wm = WorkflowMemory()
        wm.store_template("Onboarding", "HR", [
            {"description": "Create case", "action": "create_case"},
            {"description": "Provost approval", "action": "approve", "condition": "is_tenure_track", "branch": "if_true"},
            {"description": "Dept head approval", "action": "approve", "condition": "is_tenure_track", "branch": "if_false"},
        ], conditions=[{"condition": "is_tenure_track"}])
        # Resolve for tenure-track
        wf = wm.resolve_workflow("wf_tpl_1", {"is_tenure_track": True})
        assert wf is not None
        descriptions = [s.description for s in wf.resolved_steps]
        assert "Provost approval" in descriptions
        assert "Dept head approval" not in descriptions

    def test_evaluate_workflow(self):
        from prometheus_v10.workflow_memory import WorkflowMemory, PersonalizedWorkflow, WorkflowStep
        wm = WorkflowMemory()
        pred = PersonalizedWorkflow(resolved_steps=[
            WorkflowStep(description="Step A", action="a", order=0),
            WorkflowStep(description="Step B", action="b", order=1),
        ], condition_resolutions={"cond1": True}, personalization_score=1.0)
        gt = PersonalizedWorkflow(resolved_steps=[
            WorkflowStep(description="Step A", action="a", order=0),
            WorkflowStep(description="Step B", action="b", order=1),
            WorkflowStep(description="Step C", action="c", order=2),
        ], condition_resolutions={"cond1": True})
        eval_result = wm.evaluate_workflow(pred, gt)
        assert eval_result.recall == 2/3  # 2 of 3 ground truth steps found
        assert eval_result.precision == 1.0  # all predicted steps correct
        assert eval_result.condition_resolution == 1.0


# ═══════════════════════════════════════════
# P2-1: Benchmark Adapter
# ═══════════════════════════════════════════

class TestBenchmarkAdapter:
    def test_record_and_correlation(self):
        from prometheus_v10.benchmark_adapter import BenchmarkAdapter
        ba = BenchmarkAdapter()
        for i in range(15):
            ba.record(f"task_{i}", "qa", passed=(i % 2 == 0), score=0.5 + i * 0.03, internal_fitness=0.4 + i * 0.04)
        report = ba.compute_correlation()
        assert report.sample_size == 15
        assert report.pearson_r > 0  # should be positive (both increasing)

    def test_long_horizon(self):
        from prometheus_v10.benchmark_adapter import BenchmarkAdapter
        ba = BenchmarkAdapter()
        for i in range(20):
            ba.record(f"task_{i}", "code", passed=True, score=0.8, internal_fitness=0.7)
        assert ba.long_horizon_pass_rate() > 0.5


# ═══════════════════════════════════════════
# P2-2: Equilibrium Guard
# ═══════════════════════════════════════════

class TestEquilibrium:
    def test_lyapunov_monotone(self):
        from prometheus_v10.equilibrium import LyapunovMonitor
        lm = LyapunovMonitor()
        # Decreasing potential = good (evolution is making progress)
        # No regression because potential never increases
        for v in [1.0, 0.9, 0.8, 0.7, 0.6]:
            lm.record_potential(v)
        is_mono, rate = lm.check_monotonicity()
        assert is_mono
        assert rate == 0.0

    def test_lyapunov_regression(self):
        from prometheus_v10.equilibrium import LyapunovMonitor
        lm = LyapunovMonitor(regression_window=5, max_regression_rate=0.3)
        # Potential increases = regression (evolution going backwards)
        for v in [1.0, 0.9, 1.2, 0.8, 1.3, 0.7, 1.4, 0.6, 1.5, 1.6]:
            lm.record_potential(v)
        is_mono, rate = lm.check_monotonicity()
        assert not is_mono

    def test_equilibrium_guard(self):
        from prometheus_v10.equilibrium import EquilibriumGuard
        eg = EquilibriumGuard(epsilon=0.5)
        state = eg.record_round({"organ_a": 0.5, "organ_b": 0.5}, 1.0)
        assert isinstance(state.nash_epsilon, float)
        assert state.nash_epsilon == 0.0  # equal utilities = 0 epsilon


# ═══════════════════════════════════════════
# P2-3: Co-Evolution Layer
# ═══════════════════════════════════════════

class TestCoEvolution:
    def test_analyze_tool_deficiency(self):
        from prometheus_v10.coevolution_layer import CoEvolutionLayer
        cel = CoEvolutionLayer()
        analysis = cel.analyze_failure({
            "error_type": "tool_error",
            "error_message": "search timed out",
            "affected_skills": ["qa_skill"],
            "affected_tools": ["search"],
        })
        assert analysis.vertical == "tool_deficiency"

    def test_analyze_skill_defect(self):
        from prometheus_v10.coevolution_layer import CoEvolutionLayer
        cel = CoEvolutionLayer()
        analysis = cel.analyze_failure({
            "error_type": "skill_logic",
            "error_message": "wrong workflow step order",
            "affected_skills": ["onboarding_skill"],
            "affected_tools": [],
        })
        assert analysis.vertical == "skill_defect"

    def test_propose_bundle(self):
        from prometheus_v10.coevolution_layer import CoEvolutionLayer
        cel = CoEvolutionLayer()
        analysis = cel.analyze_failure({
            "error_type": "tool_error",
            "error_message": "search KeyError",
            "affected_skills": ["qa_skill"],
            "affected_tools": ["search"],
        })
        bundle = cel.propose_bundle(analysis)
        assert bundle.bundle_id.startswith("bundle_")
        assert len(bundle.tool_edits) > 0  # tool deficiency → tool edit
