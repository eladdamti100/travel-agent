"""
Master test runner — runs all test classes in one shot.
Usage: pytest tests/Test_all.py -v
"""

from tests.Test_cache_check import TestCacheCheck
from tests.Test_cache_compression import TestCompressAnswer
from tests.Test_cache_store import TestCacheStore
from tests.Test_graph_guards import TestDetectRepetition
from tests.Test_hitl_resume import TestHitlResume
from tests.Test_planner_dependency_graph import TestPlannerDependencyGraph
from tests.Test_planner_scheduler_result import TestPlannerSchedulerResult
from tests.Test_prompt_loader import TestPromptLoader
from tests.Test_researcher import TestResearcher
from tests.Test_reviewer import TestContentToText, TestReviewPlan
from tests.Test_router import TestRouterFunctions
from tests.Test_semantic_cache import (
    TestNormalizeQuery, TestCosineSimilarity, TestFindCachedAnswer,
    TestStoreCacheEntry, TestDbInitializedGuard,
)
from tests.Test_sub_agents_parallel import TestSubAgentsParallel
from tests.Test_context_enricher import TestExtractors, TestExtractTripContextDeterministic
from tests.Test_validator import TestInputValidator, TestAiValidator
