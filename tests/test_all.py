"""
Master test runner — runs all test classes in one shot.
Usage: pytest tests/test_all.py -v
"""

from tests.test_cache_check import TestCacheCheck
from tests.test_cache_compression import TestCompressAnswer
from tests.test_cache_store import TestCacheStore
from tests.test_context_enricher import TestExtractors, TestExtractTripContextDeterministic
from tests.test_graph_guards import TestDetectRepetition
from tests.test_hitl_resume import TestHitlResume
from tests.test_planner_dependency_graph import TestPlannerDependencyGraph
from tests.test_planner_scheduler_result import TestPlannerSchedulerResult
from tests.test_prompt_loader import TestPromptLoader
from tests.test_researcher import TestResearcher
from tests.test_reviewer import TestContentToText, TestReviewPlan
from tests.test_router import TestRouterFunctions
from tests.test_semantic_cache import (
    TestNormalizeQuery, TestCosineSimilarity, TestFindCachedAnswer,
    TestStoreCacheEntry, TestDbInitializedGuard,
)
from tests.test_sub_agents_parallel import TestSubAgentsParallel
from tests.test_validator import TestInputValidator, TestAiValidator
