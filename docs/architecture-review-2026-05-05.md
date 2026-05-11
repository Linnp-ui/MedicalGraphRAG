# GraphRAG Architecture Review

Date: 2026-05-05

## Scope

This review covered:

- Backend architecture under `backend/src`
- Frontend architecture under `frontend/src`
- API contract and validation behavior
- Existing automated tests
- Practical resilience checks for invalid input, concurrency, and toolchain stability

## Executive Summary

The project has a clear layered split between API, ingestion, retrieval, workflow, and graph persistence. The overall stack choice is reasonable for a GraphRAG system: FastAPI for transport, Neo4j for graph storage, LangChain/LangGraph for orchestration, and React/Vite for the UI.

The main architectural weakness is not the coarse-grained module layout. It is the thin safety boundary around graph-facing parameters and the partial drift between implementation, tests, and documentation. Before this review, graph label and relationship inputs were interpolated into Cypher paths with weak validation, node id failures surfaced as generic server errors, and frontend test/tooling drift had already broken type checks and test execution.

The codebase is in a recoverable state. The highest-value short-term work is input hardening, consistency cleanup, and better isolation of database-bound behavior. Those first steps were implemented during this review.

## Current Architecture

### Backend

- `api`: FastAPI routes, schemas, and middleware
- `core`: configuration, metrics, circuit breaker, Neo4j client
- `ingestion`: document loading, chunking, embeddings, graph building
- `retrieval`: vector, graph, and hybrid retrieval
- `chains`: Cypher generation and QA chaining
- `workflow`: LangGraph routing and execution

Assessment:

- Layering is understandable and mostly aligned with responsibilities.
- `api.routes` still carries too much orchestration logic and direct dependency wiring.
- Many route handlers are `async`, but most heavy work remains synchronous and database-bound.
- The Neo4j client is the de facto integration hub; this is useful, but it also centralizes risk.

### Frontend

- `components`: page-level views and graph canvas
- `lib`: API client and graph helpers
- `types`: graph models
- `test`: Vitest and Testing Library coverage

Assessment:

- UI structure is straightforward and serviceable.
- The API layer is thin and easy to follow.
- Graph rendering is strongly coupled to the vis-network runtime API.
- Tests were present but had drifted from current UI output and fetch error behavior.

## Dependency and Data Flow Review

### Primary backend flow

1. Request enters `FastAPI`
2. Route validates payload/query params
3. Route delegates to retrieval, workflow, or ingestion code
4. Retrieval/workflow call Neo4j and LLM-related components
5. Route reshapes result into API schema

Assessment:

- The flow is simple and readable.
- There is no strong application-service layer between routes and infrastructure.
- Graph query construction depends on runtime-generated labels and relationship types, which requires strict validation at the boundary.

### Primary frontend flow

1. View triggers `api` client call
2. Response is stored in component state
3. `GraphCanvas` converts graph data to vis-network datasets
4. User interactions trigger follow-up fetches for neighbors or search results

Assessment:

- The flow is fine for the current app size.
- As the UI grows, stateful graph interactions will become harder to reason about without a dedicated state model.

## Interface Review

### Strengths

- Request and response models are already defined with Pydantic.
- Graph endpoints expose predictable JSON structures for nodes, edges, and stats.
- Frontend API helpers broadly match backend endpoints.

### Weaknesses found

- Some query/path parameter validation lived outside schemas and was inconsistent.
- Invalid node ids returned `500` instead of `400`.
- Graph label and relationship inputs were too close to dynamic Cypher interpolation.
- README structure and actual repository structure are out of sync.

## Technology Stack Review

### Reasonable choices

- `FastAPI`: good fit for typed API development
- `Neo4j`: appropriate for entity/relation-centric retrieval
- `LangChain` and `LangGraph`: acceptable for orchestration, assuming dependency drift is controlled
- `React` + `Vite`: good fit for a graph-heavy internal UI

### Risks

- LangChain/LangGraph upgrade churn can destabilize orchestration code.
- Synchronous graph access in `async` routes will cap concurrency under load.
- The current in-memory rate limiter is process-local and not durable across instances.
- The frontend graph layer is coupled to a specific visualization library surface.

## Testing Review

### Existing tests

- Backend: pytest-based API and ingestion tests
- Frontend: Vitest tests for API client and `GraphView`

### Tests executed in this review

- `backend`: `python -m pytest -q`
  Result: `31 passed`
- `frontend`: `npm run lint`
  Result: passed
- `frontend`: `npm run test -- --run`
  Result: `27 passed`

### Resilience and boundary coverage verified

- Invalid graph node id handling
- Invalid node label handling
- Invalid relationship type handling
- Oversized graph page size handling
- Query-result node id validation
- Existing concurrent request test path

### Coverage gaps that still remain

- No real network fault injection against Neo4j or LLM providers
- No production-like pressure test against a running server process
- No long-running ingestion benchmark with large corpora
- No frontend browser-level interaction tests beyond component tests

## Implemented Improvements

The following fixes were applied during this review:

1. Hardened graph parameter handling.
   - Added identifier validation for `node_label` and `relationship_type`
   - Added numeric validation for node ids
   - Rejected invalid graph query inputs with `400/422` instead of leaking into generic `500`

2. Reduced Pydantic v2 drift.
   - Replaced deprecated class-based config with `ConfigDict` / `SettingsConfigDict`
   - Removed the `schema` field shadowing warning from the response model while preserving the API field name

3. Restored frontend type health.
   - Fixed vis-network typing issues in `GraphCanvas`
   - Normalized network ids before passing them through the UI

4. Repaired and verified automated tests.
   - Updated backend tests for new boundary behavior
   - Fixed frontend tests that were asserting stale UI assumptions
   - Re-ran backend and frontend suites successfully

## Findings and Recommendations

### High priority

- Introduce a thin service layer between `api.routes` and infrastructure-heavy modules.
  Reason: routes currently mix HTTP, orchestration, and graph access concerns.

- Move blocking retrieval and ingestion work off direct synchronous route paths.
  Reason: the current `async` surface does not provide true non-blocking behavior.

- Formalize graph query safety rules in one place.
  Reason: dynamic graph identifiers are an enduring risk surface.

### Medium priority

- Split the Neo4j client into read/query helpers and graph exploration helpers.
  Reason: `Neo4jClient` is becoming a broad utility class with mixed responsibilities.

- Add contract tests around API schemas shared with the frontend.
  Reason: backend/frontend drift already surfaced in tests and typing.

- Replace the in-memory rate limiter with a shared store if multi-instance deployment is planned.

### Lower priority

- Update repository documentation to match actual layout and commands.
- Reduce use of broad `except Exception` blocks where failure modes are known.
- Add lightweight observability around Neo4j query latency percentiles and ingestion phases.

## Optimization Plan

### Phase 1: Completed now

- Input validation hardening
- Pydantic compatibility cleanup
- Frontend type repair
- Test suite stabilization

### Phase 2: Next implementation slice

- Extract route orchestration into application services
- Introduce explicit DTO mapping between route layer and graph client
- Add retry and timeout policy tests for Neo4j-facing code
- Add a small benchmark harness for graph read endpoints

### Phase 3: Reliability and performance

- Convert heavy graph and ingestion flows to properly isolated async or worker-based execution
- Add load tests against a live backend instance
- Add network fault simulation for Neo4j and LLM upstreams
- Add cache effectiveness metrics and query hot-path profiling

## Residual Risk

The project is healthier than the initial baseline, but it is not yet production-hardened. The main remaining risks are blocking backend execution paths, limited fault-injection coverage, and service-boundary weakness between HTTP handlers and graph infrastructure.
