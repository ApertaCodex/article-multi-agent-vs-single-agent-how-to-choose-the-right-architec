# Multi-Agent vs Single-Agent: How to Choose the Right Architecture

> Originally published on [omnithium.ai](https://omnithium.ai/blog/multi-agent-vs-single-agent)

The most common architectural mistake teams make when building AI agents isn't choosing the wrong model or writing bad prompts. It's choosing the wrong number of agents.

Multi-agent systems have an almost gravitational pull for engineers. They feel architecturally correct, clean separation of concerns, specialized roles, modular composition. The instinct to reach for them is understandable. But multi-agent systems carry real costs: higher [orchestration complexity](https://omnithium.ai/blog/enterprise-ai-agent-orchestration-patterns.html), harder-to-trace failure modes, more surface area for things to go wrong, and substantially more infrastructure to [govern and monitor](https://omnithium.ai/blog/why-multi-agent-systems-need-governance.html).

A single, well-designed agent handling a well-scoped task will outperform a fragmented multi-agent pipeline on nearly every operational metric, until it doesn't. The question is knowing where that line is.

This post gives you a decision framework for making that call, grounded in task structure rather than architectural preference. We'll cover the five dimensions that actually matter: task complexity and decomposability, parallelism requirements, state management overhead, failure blast radius, and total cost implications. We'll end with a concrete decision tree.

---

## Why This Decision Is Harder Than It Looks

Single-agent and multi-agent aren't points on a spectrum, they represent fundamentally different operational models. A single agent is a loop: perceive inputs, reason, call tools, emit outputs. A multi-agent system is a distributed system: multiple reasoning processes coordinating over shared or passed state, with all the failure modes that implies.

The confusion arises because the initial implementation cost of multi-agent systems has dropped dramatically. Frameworks make spinning up a coordinator agent with three subagents feel like a ten-minute task. What they don't surface is the operational cost that accrues over weeks: debugging a failed run where three agents each produced partial outputs and none of them failed loudly, tracing a billing spike across a workflow where token consumption is split across five model calls with different context windows, or explaining to a [compliance team](https://omnithium.ai/blog/eu-ai-act-agent-compliance.html) which agent made the decision that ended up in the audit log.

Neither architecture is inherently superior. The right choice depends on what your task actually requires.

---

## Dimension 1: Task Complexity and Decomposability

The first question to ask about any task is whether it genuinely decomposes into independent subtasks.

**Tasks that decompose well** have clear boundaries between sub-problems. The output of one step doesn't need to be in the context window of another step to be evaluated. The subtasks can be specified independently and validated independently. A common example: a research pipeline that (1) retrieves documents from five different sources, (2) summarizes each, and (3) synthesizes the summaries into a final report. Steps 1 and 2 are embarrassingly parallel, each document summary doesn't need to know about the others until synthesis.

**Tasks that decompose poorly** have high inter-step dependency. Each step's output shapes how the next step should reason. Code generation is a canonical case: generating a function, then generating tests for it, then refactoring based on test output requires shared context at every step. Breaking this into three agents doesn't help, you're just passing a large shared context over message boundaries instead of keeping it in one context window.

A practical test: if you mapped your task as a directed graph of subtasks, how many edges would there be relative to nodes? Sparse graphs (few dependencies) favor multi-agent. Dense graphs favor single-agent with tool calls.

```python
# Single-agent approach: high inter-step dependency
# All context stays in one reasoning loop

def run_code_review_agent(pr_diff: str, codebase_context: str) -> ReviewResult:
 agent = Agent(
 model="claude-opus-4",
 system_prompt=CODE_REVIEW_SYSTEM_PROMPT,
 tools=[
 search_codebase_tool,
 check_test_coverage_tool,
 lookup_style_guide_tool,
 post_review_comment_tool,
 ]
 )
 # All reasoning happens in one context window,
 # the agent can refer back to earlier findings
 # without serializing/deserializing state
 return agent.run(
 f"Review this PR diff:\n{pr_diff}\n\nCodebase context:\n{codebase_context}"
 )
```

```python
# Multi-agent approach: works when subtasks are genuinely independent
# Each subagent gets only the context it needs

def run_market_research_pipeline(query: str, sources: list[str]) -> ResearchReport:
 # Step 1: Retrieve and summarize in parallel (no inter-dependency)
 with AgentPool(model="gpt-4o-mini", max_workers=len(sources)) as pool:
 summaries = pool.map(
 task=summarize_source_task,
 inputs=[{"source": s, "query": query} for s in sources]
 )

 # Step 2: Synthesize (single agent, now has all summaries)
 synthesis_agent = Agent(
 model="claude-opus-4",
 system_prompt=SYNTHESIS_SYSTEM_PROMPT,
 )
 return synthesis_agent.run(
 f"Synthesize these research summaries for: {query}\n\n"
 + "\n\n".join(summaries)
 )
```

A common decomposability mistake: treating sequential steps as parallelizable. If step 2 needs step 1's output to determine what to do (not just to use as input), they're not parallelizable, they're a chain. Chains inside a single agent are usually cheaper and more reliable than chains across multiple agents.

---

## Dimension 2: Parallelism Requirements

Parallelism is the clearest case for multi-agent systems, and the one most often overapplied.

Real parallelism needs arise when:

- You have N independent work items (documents, tickets, records) that need the same processing
- You need to run independent sub-investigations and combine results
- You're hitting latency ceilings on sequential processing that actually affect user-facing SLAs
- Different subtasks benefit from different models (e.g., one subtask needs a reasoning model, another just needs a fast summarizer)

The critical check: will users actually notice the latency difference? Async background workflows often don't benefit from parallelism in any user-perceptible way. If a batch job runs in 40 minutes single-agent versus 12 minutes multi-agent, but it runs overnight either way, you've added orchestration complexity for no practical gain.

When parallelism genuinely matters, it delivers real throughput improvements. A document processing pipeline handling 500 PDFs sequentially at ~8 seconds each takes over an hour. The same pipeline with 20 parallel subagents completes in roughly 3-4 minutes, a 20x improvement that's operationally meaningful.

But measure first:

```python
import asyncio
import time
from typing import Callable

async def benchmark_parallelism(
 task_fn: Callable,
 inputs: list,
 concurrency_levels: list[int] = [1, 5, 10, 20]
) -> dict[int, float]:
 """
 Benchmark actual latency gains at different concurrency levels
 before committing to a multi-agent architecture.
 """
 results = {}

 for concurrency in concurrency_levels:
 semaphore = asyncio.Semaphore(concurrency)

 async def bounded_task(input_item):
 async with semaphore:
 return await task_fn(input_item)

 start = time.perf_counter()
 await asyncio.gather(*[bounded_task(inp) for inp in inputs])
 elapsed = time.perf_counter() - start

 results[concurrency] = elapsed
 print(f"Concurrency {concurrency}: {elapsed:.2f}s")

 return results
```

Run this before you architect a multi-agent system for latency reasons. You'll often find that concurrency=5 captures 80% of the latency gains of concurrency=20, with a fraction of the coordination overhead.

---

## Dimension 3: State Management Overhead

This is where multi-agent systems quietly accumulate their most insidious costs.

A single agent's state is its context window. It's immediately available to every reasoning step, consistent by definition, and disposed of cleanly when the run ends. The agent doesn't need to worry about whether another agent has modified shared state since its last read.

A multi-agent system needs a strategy for shared state, and every strategy has a cost:

**Message passing** (agents pass outputs directly to the next agent) is the simplest approach but creates brittle pipelines. If the format of agent A's output changes slightly, agent B may fail silently or hallucinate corrections.

**Shared memory stores** (agents read/write to a common key-value store or database) introduce race conditions if agents run in parallel. You need locking or conflict resolution strategies.

**Coordinator-managed state** (a coordinator agent holds all state and delegates with explicit context) is the safest pattern but pushes a large context window load onto the coordinator and creates a single point of bottleneck.

```yaml
# Example: State passing configuration in a multi-agent workflow
# This looks clean but becomes complex as agents multiply

workflow:
 name: customer_support_pipeline
 state_strategy: message_passing

 agents:
 - id: classifier
 outputs:
 - key: intent
 schema: { type: string, enum: [billing, technical, general] }
 - key: sentiment
 schema: { type: string, enum: [positive, neutral, negative] }
 - key: priority
 schema: { type: integer, minimum: 1, maximum: 5 }

 - id: resolver
 inputs:
 # Explicit dependency: resolver needs ALL classifier outputs
 # If classifier output schema changes, resolver breaks
 from: classifier
 keys: [intent, sentiment, priority, original_message]

 - id: quality_checker
 inputs:
 from: [classifier, resolver]
 keys: [intent, resolution, original_message]
 # Now quality_checker needs to merge two state sources
 # Conflict resolution: what if classifier and resolver
 # have different views of 'intent'?
```

A diagnostic question: how many agents need to read or write shared state? If the answer is more than two, you almost certainly need an explicit state management layer, and you should account for that complexity in your build estimate. A single agent sidesteps this entirely.

---

## Dimension 4: Failure Blast Radius

This is the dimension most teams underweight during architecture decisions and overweight after their first production incident.

**Single-agent failure modes** are generally loud and self-contained. The agent fails, returns an error, and the run terminates. You have a single stack trace, a single set of tool call logs, and a clear accountability boundary.

**Multi-agent failure modes** are frequently silent, partial, and ambiguous. Common patterns:

- Agent A completes successfully but produces subtly wrong output. Agent B accepts it without validation. The final output is wrong, but both agents report success.
- Agent A times out. The coordinator retries Agent A. Agent B has already started and is now working with stale state. Both complete, producing conflicting outputs.
- One agent in a parallel pool hits a rate limit. The pool partial-succeeds. The aggregation step synthesizes 8 of 10 expected inputs without noticing the gap.

These failure modes aren't hypothetical, they're the standard failure taxonomy of distributed systems. Multi-agent architectures inherit all of them.

Quantifying blast radius matters for systems where failures have downstream consequences. A billing automation agent that erroneously generates invoices has a much higher blast radius than a content summarization agent that produces a mediocre summary. Higher blast radius tasks require more explicit failure handling, which is easier to implement and verify in a single-agent system.

```python
# Pattern: Explicit failure surface reduction in multi-agent pipelines
# Validate subagent outputs before passing downstream

from pydantic import BaseModel, ValidationError
from typing import Optional

class SubagentOutput(BaseModel):
 intent: str
 confidence: float
 extracted_entities: list[str]
 reasoning_trace: str

 class Config:
 # Reject unexpected fields, don't silently drop them
 extra = "forbid"

async def safe_subagent_call(
 agent: Agent,
 input_data: dict,
 output_schema: type[BaseModel],
 max_retries: int = 2
) -> Optional[BaseModel]:
 """
 Validates subagent output against schema before passing downstream.
 Surfaces schema violations as explicit failures rather than
 allowing malformed data to propagate.
 """
 for attempt in range(max_retries + 1):
 try:
 raw_output = await agent.run_async(input_data)
 validated = output_schema.model_validate(raw_output)
 return validated
 except ValidationError as e:
 if attempt == max_retries:
 # Fail loudly, don't silently skip
 raise AgentOutputValidationError(
 agent_id=agent.id,
 attempt=attempt,
 validation_errors=e.errors(),
 raw_output=raw_output
 )
 # Log and retry with explicit error context
 await log_validation_failure(agent.id, attempt, e)
 return None
```

The key principle: multi-agent systems need explicit output contracts between agents. If you're not validating what one agent passes to another, you're accumulating silent failure risk with every agent you add.

---

## Dimension 5: Cost Implications

Cost is rarely the deciding factor in architecture decisions, until it's the only thing anyone talks about.

The cost equation for multi-agent systems has three components that single-agent systems mostly avoid:

**1. Coordination overhead tokens.** Every agent needs system prompts, output formatting instructions, and usually some context about where it sits in the workflow. A multi-agent system with five agents might spend 2,000-5,000 tokens per agent just on coordination scaffolding, before doing any actual task work. At scale, this overhead compounds.

**2. Context duplication.** When agents need shared context (the original user request, background information, previous decisions), that context gets repeated in each agent's context window. A 3,000-token background document included in five agents' contexts costs 15,000 tokens of input per run.

**3. Model tier choices.** Multi-agent architectures do enable model tiering, using a cheaper model for simple subtasks and a more expensive model for complex synthesis. This can reduce costs when applied deliberately. A realistic estimate: routing 60% of subtasks to a fast, inexpensive model and 40% to a capable frontier model, with coordination overhead factored in, typically saves 30-45% on raw model costs versus running everything through a frontier model. But this benefit only materializes if you've instrumented cost attribution per agent and actively tune the routing over time.

```python
# Cost attribution per agent, essential for optimization
# Without this, you're flying blind

from dataclasses import dataclass, field
from decimal import Decimal

@dataclass
class AgentRunCost:
 agent_id: str
 model: str
 input_tokens: int
 output_tokens: int
 tool_calls: int

 # Model pricing (per 1M tokens, adjust to current rates)
 INPUT_PRICES = {
 "gpt-4o": Decimal("2.50"),
 "gpt-4o-mini": Decimal("0.15"),
 "claude-opus-4": Decimal("15.00"),
 "claude-haiku-3-5": Decimal("0.80"),
 }
 OUTPUT_PRICES = {
 "gpt-4o": Decimal("10.00"),
 "gpt-4o-mini": Decimal("0.60"),
 "claude-opus-4": Decimal("75.00"),
 "claude-haiku-3-5": Decimal("4.00"),
 }

 @property
 def total_cost(self) -> Decimal:
 input_cost = (
 Decimal(self.input_tokens) / Decimal(1_000_000)
 * self.INPUT_PRICES[self.model]
 )
 output_cost = (
 Decimal(self.output_tokens) / Decimal(1_000_000)
 * self.OUTPUT_PRICES[self.model]
 )
 return input_cost + output_cost

def aggregate_workflow_cost(agent_costs: list[AgentRunCost]) -> dict:
 total = sum(c.total_cost for c in agent_costs)
 by_agent = {c.agent_id: c.total_cost for c in agent_costs}
 by_model = {}
 for c in agent_costs:
 by_model[c.model] = by_model.get(c.model, Decimal(0)) + c.total_cost

 return {
 "total_cost": total,
 "cost_by_agent": by_agent,
 "cost_by_model": by_model,
 "coordination_overhead_pct": _estimate_overhead(agent_costs)
 }
```

The honest cost conclusion: for tasks that run infrequently or have low volume, multi-agent overhead is rarely worth optimizing. For high-volume, high-frequency workloads, the [cost implications](https://omnithium.ai/blog/llm-cost-optimization-agents.html) of your architecture choice can determine whether the product is viable.

---

## The Decision Tree

Here's the framework distilled into a decision process. Work through the questions in order and stop at the first definitive branch.

```
START: Does your task have clear, independently executable subtasks?
│
├── NO → Single-agent with tools.
│ Your task has high inter-step dependency. Keeping all
│ reasoning in one context window will outperform
│ serializing state across agent boundaries.
│
└── YES → Does parallel execution actually reduce meaningful latency
 or enable necessary throughput?
 │
 ├── NO → Single-agent with sequential tool calls.
 │ The task decomposes, but serialized execution
 │ is sufficient. Avoid orchestration overhead.
 │
 └── YES → Do subtasks require genuinely different
 capabilities (different models, different
 tool sets, different system prompts)?
 │
 ├── NO → Parallel single-agent with batching.
 │ Run the same agent concurrently
 │ over independent inputs. Simpler
 │ than multi-agent, same throughput.
 │
 └── YES → What is the blast radius of a
 partial failure?
 │
 ├── LOW → Multi-agent with
 │ lightweight coordination.
 │ Choreography or event-
 │ driven patterns work well.
 │
 └── HIGH → Multi-agent with explicit
 output contracts, validation
 at every handoff, and
 human-in-the-loop gates
 at critical junctions.
 Budget for observability
 infrastructure.
```

Two additional overrides that should redirect you to single-agent regardless of where the tree leads:

**Override 1: Debugging velocity matters.** If your team is still in early development, the debugging speed advantage of single-agent systems is significant. Multi-agent systems are substantially harder to trace and reproduce. Build single-agent first, refactor to multi-agent if you hit a genuine ceiling.

**Override 2: Compliance requires clear attribution.** If you need to answer "which system made this decision?" at the individual-step level, a single agent with a detailed tool call audit log is far easier to reason about than a multi-agent workflow where a decision emerged from the interaction of three agents. This matters acutely in regulated industries.

---

## Practical Refactoring: When to Migrate and How

Teams that start single-agent and need to migrate to multi-agent generally have one of three trigger signals:

1. **Latency ceiling:** Sequential processing is too slow for production SLAs, and profiling confirms the bottleneck is LLM inference time (not I/O or tool calls).
2. **Context window exhaustion:** The task requires more information than fits in a single context window, and retrieval-augmented approaches aren't sufficient.
3. **Model specialization:** Different subtasks have dramatically different performance profiles across models, a reasoning-heavy analysis step benefits from a frontier model, while a formatting step works fine with a smaller model.

When any of these signals appears, migrate the smallest possible scope. Extract one subtask into a separate agent. Instrument it. Validate that the performance improvement materializes and that failure handling is correct. Then expand.

```python
# Migration pattern: extract one subtask at a time
# Before: single agent handles everything

class SingleAgentPipeline:
 def run(self, document: str) -> AnalysisResult:
 agent = Agent(model="claude-opus-4", tools=ALL_TOOLS)
 return agent.run(FULL_ANALYSIS_PROMPT.format(document=document))

# After step 1: extract the classification subtask
# Everything else still runs in the original agent

class PartiallyMigratedPipeline:
 def __init__(self):
 self.classifier = Agent(
 model="gpt-4o-mini", # cheaper model, simpler task
 tools=[],
 system_prompt=CLASSIFICATION_ONLY_PROMPT
 )
 self.main_agent = Agent(
 model="claude-opus-4",
 tools=ALL_TOOLS,
 system_prompt=ANALYSIS_WITH_CLASSIFICATION_PROMPT
 )

 def run(self, document: str) -> AnalysisResult:
 # Step 1: classify (now a separate agent)
 classification = self.classifier.run(document)

 # Validate before passing downstream
 assert classification.category in VALID_CATEGORIES

 # Step 2: full analysis with classification context
 return self.main_agent.run(
 ANALYSIS_PROMPT.format(
 document=document,
 category=classification.category
 )
 )
```

This incremental approach lets you measure the actual impact of each extraction, latency, cost, failure rate, before committing to a fully decomposed multi-agent design.

---

## Where Teams Go Wrong

A few failure patterns worth naming explicitly:

**Premature decomposition.** Splitting tasks into agents before understanding the inter-step dependency structure. The result is agents that need to pass large context windows between each other, eliminating the parallelism benefit while adding orchestration overhead.

**Agent proliferation.** Each new capability gets its own agent. After six months, a system has 15 agents where 5 would have been sufficient. [Governance](https://omnithium.ai/blog/ai-agent-governance-enterprise-guide.html), monitoring, and debugging costs scale with agent count, linearly if you're lucky, super-linearly in practice.

**Ignoring partial failure.** Designing the happy path of a multi-agent system without designing its failure modes. Production systems experience partial failures constantly. If your multi-agent pipeline doesn't have explicit handling for the case where one agent in a parallel pool returns garbage, you'll find out under pressure.

**Treating multi-agent as inherently more scalable.** A well-designed single agent with async tool calls and caching will outscale a poorly designed multi-agent system on most workloads. Scalability comes from implementation quality, not agent count.

---

## Conclusion

The decision between single-agent and multi-agent architecture isn't about sophistication, it's about fit. Multi-agent systems solve real problems: genuine parallelism requirements, tasks that exceed context window limits, and workflows that benefit from model specialization. They introduce real costs: orchestration complexity, harder failure modes, state management overhead, and higher infrastructure requirements for [observability](https://omnithium.ai/blog/agent-observability-beyond-uptime.html) and [governance](https://omnithium.ai/blog/why-multi-agent-systems-need-governance.html).

For most tasks, especially early in development, a single agent with well-designed tools is the right starting point. It's faster to build, easier to debug, simpler to govern, and often cheaper to operate. The signal to migrate to multi-agent is specific and measurable, not a feeling that the architecture should be more modular.

Run the task through the five dimensions: decomposability, parallelism needs, state management complexity, failure blast radius, and cost. Apply the decision tree. If multi-agent is the answer, scope the smallest possible implementation that solves the actual bottleneck. Instrument everything from day one, you can't optimize what you can't measure.

The best architecture is the simplest one that meets your production requirements. For most teams, that bar is lower than the architecture they initially reach for.

No matter which architecture you choose, the platform you run it on matters. [Omnithium](https://omnithium.ai) gives teams a single control plane for both single-agent and multi-agent workflows, with built-in observability, governance, and cost tracking. [Explore Omnithium’s pricing](https://omnithium.ai/pricing) to find a plan that fits your scale.