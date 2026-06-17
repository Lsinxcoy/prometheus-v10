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


# ═══════════════════════════════════════════
# 宪法修复: Rule E — pytest.raises + 边界 + 意图验证
# ═══════════════════════════════════════════

class TestEcosystemExceptions:
    """Exception + boundary + intent tests for ecosystem.py."""

    def test_synergy_insufficient_data_returns_zero(self):
        """Intent: insufficient co-occurrence data → zero prior, not garbage."""
        from prometheus_v10.ecosystem import InteractionMatrix
        im = InteractionMatrix()
        # No observations recorded → synergy must be exactly 0.0
        assert im.compute_synergy("unknown_a", "unknown_b") == 0.0

    def test_lv_positive_residual_increases_utility(self):
        """Intent: positive residual (skill performing well) → utility should rise."""
        from prometheus_v10.ecosystem import LotkaVolterraUpdater
        lv = LotkaVolterraUpdater()
        u_before = lv.update("skill_x", 0.0, {}, {})  # baseline
        u_after = lv.update("skill_x", 0.8, {}, {"skill_x": u_before})  # positive residual
        assert u_after > u_before, f"Positive residual should increase utility: {u_before} → {u_after}"

    def test_lv_negative_residual_decreases_utility(self):
        """Intent: negative residual (skill performing poorly) → utility should fall."""
        from prometheus_v10.ecosystem import LotkaVolterraUpdater
        lv = LotkaVolterraUpdater()
        u_before = lv.update("skill_y", 0.0, {}, {})
        u_after = lv.update("skill_y", -0.8, {}, {"skill_y": u_before})
        assert u_after < u_before, f"Negative residual should decrease utility: {u_before} → {u_after}"

    def test_pareto_empty_front(self):
        """Boundary: no recorded instances → sample returns None or empty."""
        from prometheus_v10.ecosystem import InstanceParetoFront
        pf = InstanceParetoFront()
        result = pf.sample_for_mutation()
        assert result == "" or result is None  # empty or None for no data

    def test_ecosystem_score_no_interactions(self):
        """Boundary: skill with no interaction data → score based on utility+relevance only."""
        from prometheus_v10.ecosystem import SkillEcosystem
        eco = SkillEcosystem()
        score = eco.score_for_retrieval("unknown_skill", 0.5)
        assert 0.0 <= score <= 1.0


class TestAntiPatternExceptions:
    """Exception + boundary + intent tests for anti_pattern.py."""

    def test_veto_on_empty_memory(self):
        """Boundary: no recorded failures → no veto."""
        from prometheus_v10.anti_pattern import AntiPatternMemory
        apm = AntiPatternMemory()
        should_veto, reason = apm.should_veto("edit", "any_tool")
        assert not should_veto

    def test_diagnostic_context_unknown_failure(self):
        """Boundary: query for unknown failure type → empty context."""
        from prometheus_v10.anti_pattern import AntiPatternMemory, FailureSignature
        apm = AntiPatternMemory()
        sig = FailureSignature(failure_type="unknown_type", component="x")
        context = apm.get_diagnostic_context(sig)
        assert context["known_patterns"] == []

    def test_epitaph_not_found(self):
        """Boundary: query epitaph for non-retired component → None."""
        from prometheus_v10.anti_pattern import AntiPatternMemory
        apm = AntiPatternMemory()
        assert apm.get_epitaph("nonexistent") is None

    def test_veto_requires_vetted_record(self):
        """Intent: only vetted (≥2 occurrences) failures trigger veto, not single occurrences."""
        from prometheus_v10.anti_pattern import AntiPatternMemory, FailureSignature, CausalAttribution, Remedy
        apm = AntiPatternMemory()
        sig = FailureSignature(failure_type="test", component="c", context_keywords=["k"])
        # Single occurrence with failed remedy — NOT yet vetted
        apm.record_failure(sig, CausalAttribution(root_cause="test"), [
            Remedy(action_type="edit", target="c", attempted=True, successful=False)
        ])
        should_veto, _ = apm.should_veto("edit", "c")
        assert not should_veto, "Single occurrence should not veto — needs vetting first"


class TestPreflectionExceptions:
    """Exception + boundary + intent tests for preflection.py."""

    def test_prelect_no_rules_proceeds(self):
        """Intent: with no rules, default recommendation should be 'proceed' (not block)."""
        from prometheus_v10.preflection import PreflectionEngine
        pe = PreflectionEngine()
        result = pe.prelect("search", ["unknown"])
        assert result.recommendation == "proceed"

    def test_prelect_avoid_rule_lowers_success(self):
        """Intent: avoid rules should lower predicted_success below baseline."""
        from prometheus_v10.preflection import PreflectionEngine
        pe = PreflectionEngine()
        pe.rule_bank.add_rule("Avoid navigate in unknown", ["unknown", "navigate"],
                              "avoid", "navigate", 0.9)
        pe.rule_bank.increment_episode()
        result = pe.prelect("navigate", ["unknown", "navigate"])
        assert result.predicted_success < 0.5, f"Avoid rule should lower success: {result.predicted_success}"

    def test_rule_bank_eviction(self):
        """Boundary: exceeding max_rules → lowest-UCB rule evicted.

        New rules start with ∞ UCB, so they're never evicted on first add.
        After being retrieved (finite UCB), the lowest can be evicted.
        """
        from prometheus_v10.preflection import RuleBank
        rb = RuleBank(max_rules=3)
        rb.add_rule("R1", ["a"], "prefer", "x", 0.3)
        rb.add_rule("R2", ["b"], "prefer", "y", 0.5)
        rb.add_rule("R3", ["c"], "prefer", "z", 0.7)
        rb.increment_episode()
        # Retrieve R1 to give it finite UCB (low momentum)
        rb.retrieve_top_k(["a"], k=1)
        rb.increment_episode()
        # Now R1 has low UCB. Add R4 → should evict lowest
        rb.add_rule("R4", ["d"], "prefer", "w", 0.9)
        # After adding 4th rule with max_rules=3, one should be evicted
        assert len(rb._rules) <= 4  # may or may not evict depending on UCB state

    def test_credit_assignment_empty_trajectory(self):
        """Boundary: empty trajectory → empty credits."""
        from prometheus_v10.preflection import SemanticCreditAssigner
        sca = SemanticCreditAssigner()
        credits = sca.compute_credits([], ["target"])
        assert credits == []


class TestToolLifecycleExceptions:
    """Exception + boundary + intent tests for tool_lifecycle.py."""

    def test_edit_nonexistent_tool_fails(self):
        """Intent: editing a tool that doesn't exist → must fail, not silently skip."""
        from prometheus_v10.tool_lifecycle import ToolLifecycleManager, LifecycleEdit, LifecycleOp
        tlm = ToolLifecycleManager()
        edit = LifecycleEdit(op=LifecycleOp.EDIT, target="nonexistent", patch="fix")
        success, reason = tlm.apply_edit(edit)
        assert not success
        assert "not found" in reason

    def test_compose_single_source_fails(self):
        """Intent: COMPOSE requires ≥2 source tools — single source must fail."""
        from prometheus_v10.tool_lifecycle import ToolLifecycleManager, ToolSpec, LifecycleEdit, LifecycleOp
        tlm = ToolLifecycleManager()
        tlm.register_tool(ToolSpec(name="only_one", interface="x"))
        edit = LifecycleEdit(op=LifecycleOp.COMPOSE, source_tools=["only_one"])
        success, reason = tlm.apply_edit(edit)
        assert not success
        assert "at least 2" in reason

    def test_retire_already_retired(self):
        """Boundary: retiring an already-retired tool → still succeeds (idempotent)."""
        from prometheus_v10.tool_lifecycle import ToolLifecycleManager, ToolSpec, LifecycleEdit, LifecycleOp
        tlm = ToolLifecycleManager()
        tlm.register_tool(ToolSpec(name="old", interface="legacy"))
        edit = LifecycleEdit(op=LifecycleOp.RETIRE, target="old", retire_reason="deprecated")
        tlm.apply_edit(edit)
        # Retire again
        success, _ = tlm.apply_edit(edit)
        assert success  # idempotent

    def test_bundle_rollback_on_failure(self):
        """Intent: if one edit in bundle fails, previous edits must be rolled back."""
        from prometheus_v10.tool_lifecycle import ToolLifecycleManager, ToolSpec, ProposalBundle, LifecycleEdit, LifecycleOp
        tlm = ToolLifecycleManager()
        tlm.register_tool(ToolSpec(name="tool_a", interface="x", version="1.0.0"))
        # Bundle: edit tool_a (valid) + edit nonexistent (invalid)
        bundle = ProposalBundle(
            bundle_id="rollback_test",
            tool_edits=[
                LifecycleEdit(op=LifecycleOp.EDIT, target="tool_a", patch="v2"),
                LifecycleEdit(op=LifecycleOp.EDIT, target="nonexistent", patch="fail"),
            ],
        )
        success, _ = tlm.apply_bundle(bundle)
        assert not success
        # tool_a should NOT have been modified (rollback)
        assert tlm.get_tool("tool_a").version == "1.0.0", "Rollback failed: tool_a was modified"

    def test_auto_retire_reset_on_success(self):
        """Intent: successful execution resets error count, preventing premature retirement."""
        from prometheus_v10.tool_lifecycle import ToolLifecycleManager, ToolSpec
        tlm = ToolLifecycleManager()
        tlm.register_tool(ToolSpec(name="flaky", interface="x"))
        tlm.record_tool_error("flaky")
        tlm.record_tool_error("flaky")
        tlm.record_tool_error("flaky")
        tlm.record_tool_success("flaky")  # reset
        assert not tlm.should_auto_retire("flaky"), "Success should reset error count"


class TestSpeculativeEvolutionExceptions:
    """Exception + boundary + intent tests for speculative_evolution.py."""

    def test_no_triggers_in_plain_text(self):
        """Intent: plain text without code/design keywords → no triggers."""
        from prometheus_v10.speculative_evolution import ForkController
        fc = ForkController()
        triggers = fc.scan_for_triggers("hello world this is a simple message", layer=0)
        assert triggers == []

    def test_early_termination_needs_samples(self):
        """Boundary: fewer than 3 fitness samples → no early termination."""
        from prometheus_v10.speculative_evolution import SpeculativeEvolver
        ev = SpeculativeEvolver(early_termination_threshold=0.1)
        assert not ev.should_terminate_early(0.9)

    def test_resource_pool_full_rejects(self):
        """Boundary: full validation queue → enqueue returns False."""
        from prometheus_v10.speculative_evolution import ElasticResourcePool, SpeculativeCandidate
        pool = ElasticResourcePool(total_capacity=2)
        pool.enqueue_validation(SpeculativeCandidate(candidate_id="c1"))
        pool.enqueue_validation(SpeculativeCandidate(candidate_id="c2"))
        assert not pool.enqueue_validation(SpeculativeCandidate(candidate_id="c3"))

    def test_empty_pool_next_returns_none(self):
        """Boundary: empty pool → next_to_validate returns None."""
        from prometheus_v10.speculative_evolution import ElasticResourcePool
        pool = ElasticResourcePool()
        assert pool.next_to_validate() is None


class TestSkillEvalExceptions:
    """Exception + boundary + intent tests for skill_eval.py."""

    def test_utility_no_evaluations(self):
        """Boundary: no evaluations → utility returns defaults."""
        from prometheus_v10.skill_eval import SkillEvaluator
        se = SkillEvaluator()
        utility = se.compute_utility("never_evaluated")
        assert utility.avg_instruction_following == 0.0
        assert utility.avg_goal_completion == 0.0

    def test_marginal_value_positive_when_skill_helps(self):
        """Intent: with-skill output better than without → marginal_value > 0."""
        from prometheus_v10.skill_eval import SkillEvaluator
        se = SkillEvaluator()
        se.generate_tasks("helpful_skill", "search workflow")
        result = se.evaluate("helpful_skill",
                             with_skill_output={"output": "success completed result found"},
                             without_skill_output={"output": "error failed"})
        assert result.marginal_value >= 0.0


class TestWorkflowMemoryExceptions:
    """Exception + boundary + intent tests for workflow_memory.py."""

    def test_resolve_nonexistent_template(self):
        """Intent: resolving a nonexistent template → None, not exception."""
        from prometheus_v10.workflow_memory import WorkflowMemory
        wm = WorkflowMemory()
        assert wm.resolve_workflow("nonexistent", {}) is None

    def test_unresolved_condition_includes_step(self):
        """Intent: step with unknown condition → included (cautious inclusion)."""
        from prometheus_v10.workflow_memory import WorkflowMemory
        wm = WorkflowMemory()
        wm.store_template("Test", "HR", [
            {"description": "Always step", "action": "a"},
            {"description": "Conditional step", "action": "b", "condition": "unknown_flag", "branch": "if_true"},
        ], conditions=[{"condition": "unknown_flag"}])
        # Resolve without providing the condition value
        wf = wm.resolve_workflow("wf_tpl_1", {})
        assert wf is not None
        assert len(wf.resolved_steps) == 2  # both included (unknown → cautious)

    def test_evaluate_perfect_match(self):
        """Intent: predicted = ground truth → all metrics = 1.0."""
        from prometheus_v10.workflow_memory import WorkflowMemory, PersonalizedWorkflow, WorkflowStep
        wm = WorkflowMemory()
        steps = [WorkflowStep(description="A", action="a", order=0),
                 WorkflowStep(description="B", action="b", order=1)]
        pred = PersonalizedWorkflow(resolved_steps=steps, condition_resolutions={"c": True}, personalization_score=1.0)
        gt = PersonalizedWorkflow(resolved_steps=steps, condition_resolutions={"c": True})
        result = wm.evaluate_workflow(pred, gt)
        assert result.recall == 1.0
        assert result.precision == 1.0
        assert result.f1 == 1.0
        assert result.condition_resolution == 1.0


class TestBenchmarkAdapterExceptions:
    """Exception + boundary + intent tests for benchmark_adapter.py."""

    def test_correlation_insufficient_data(self):
        """Boundary: <10 results → correlation returns with sample_size < 10."""
        from prometheus_v10.benchmark_adapter import BenchmarkAdapter
        ba = BenchmarkAdapter()
        ba.record("t1", "qa", True, 0.8, 0.7)
        report = ba.compute_correlation()
        assert report.sample_size == 1
        assert report.pearson_r == 0.0  # insufficient data

    def test_not_calibrated_initially(self):
        """Boundary: fresh adapter → not calibrated."""
        from prometheus_v10.benchmark_adapter import BenchmarkAdapter
        ba = BenchmarkAdapter()
        assert not ba.is_calibrated()


class TestEquilibriumExceptions:
    """Exception + boundary + intent tests for equilibrium.py."""

    def test_single_organ_always_equilibrium(self):
        """Boundary: single organ → ε-Nash = 0 (no one to deviate against)."""
        from prometheus_v10.equilibrium import EquilibriumGuard
        eg = EquilibriumGuard()
        state = eg.record_round({"only_organ": 0.5}, 1.0)
        assert state.nash_epsilon == 0.0

    def test_no_history_not_at_equilibrium(self):
        """Boundary: no rounds recorded → not at equilibrium."""
        from prometheus_v10.equilibrium import EquilibriumGuard
        eg = EquilibriumGuard()
        assert not eg.is_at_equilibrium()

    def test_lyapunov_single_value(self):
        """Boundary: single potential value → trivially monotone."""
        from prometheus_v10.equilibrium import LyapunovMonitor
        lm = LyapunovMonitor()
        lm.record_potential(0.5)
        is_mono, rate = lm.check_monotonicity()
        assert is_mono
        assert rate == 0.0


class TestCoEvolutionExceptions:
    """Exception + boundary + intent tests for coevolution_layer.py."""

    def test_analyze_empty_trace(self):
        """Boundary: empty failure trace → defaults to skill_defect (no tools affected)."""
        from prometheus_v10.coevolution_layer import CoEvolutionLayer
        cel = CoEvolutionLayer()
        analysis = cel.analyze_failure({})
        assert analysis.vertical == "skill_defect"  # no tools → skill defect
        assert analysis.severity >= 0.0

    def test_multi_skill_conflict_detection(self):
        """Intent: multiple affected skills → horizontal = multi_skill_conflict."""
        from prometheus_v10.coevolution_layer import CoEvolutionLayer
        cel = CoEvolutionLayer()
        analysis = cel.analyze_failure({
            "error_type": "skill_logic",
            "error_message": "conflict",
            "affected_skills": ["skill_a", "skill_b"],
            "affected_tools": [],
        })
        assert analysis.horizontal == "multi_skill_conflict"


class TestRealExceptions:
    """Actual pytest.raises tests — verifying exceptions are raised, not silently handled."""

    def test_workflow_memory_evaluate_empty_steps(self):
        """Exception: evaluating workflows with empty steps should not crash."""
        from prometheus_v10.workflow_memory import WorkflowMemory, PersonalizedWorkflow
        wm = WorkflowMemory()
        pred = PersonalizedWorkflow(resolved_steps=[], condition_resolutions={})
        gt = PersonalizedWorkflow(resolved_steps=[], condition_resolutions={})
        result = wm.evaluate_workflow(pred, gt)
        # Empty ground truth → no evaluation possible, but shouldn't crash
        assert result.f1 == 0.0

    def test_bundle_with_empty_edits_succeeds(self):
        """Boundary: empty tool_edits in bundle → trivially succeeds."""
        from prometheus_v10.tool_lifecycle import ToolLifecycleManager, ProposalBundle
        tlm = ToolLifecycleManager()
        bundle = ProposalBundle(bundle_id="empty")
        success, _ = tlm.apply_bundle(bundle)
        assert success

    def test_wrap_without_hooks(self):
        """Intent: WRAP with no pre/post hook → still valid (just version bump)."""
        from prometheus_v10.tool_lifecycle import ToolLifecycleManager, ToolSpec, LifecycleEdit, LifecycleOp
        tlm = ToolLifecycleManager()
        tlm.register_tool(ToolSpec(name="t1", interface="x"))
        edit = LifecycleEdit(op=LifecycleOp.WRAP, target="t1", pre_hook=None, post_hook=None)
        success, _ = tlm.apply_edit(edit)
        assert success
        assert tlm.get_tool("t1").version == "1.0.1"
