"""End-to-end test: runs a full conversation through the graph with real LLM."""

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings
from app.graph.builder import build_graph
from app.models.scenario import load_scenario


async def main():
    print(f"Model: {settings.llm_model}")
    print(f"API Base: {settings.llm_api_base}")
    print(f"API Key: {settings.llm_api_key[:10]}...")
    print()

    scenario = load_scenario("diabetes_drug")
    graph = build_graph()
    session_id = "e2e-test-002"
    config = {"configurable": {"thread_id": session_id}}

    # Initialize state
    initial_state = {
        "session_id": session_id,
        "scenario_id": "diabetes_drug",
        "messages": [AIMessage(content=scenario.opening_message)],
        "semantic_coverage": scenario.get_coverage_init(),
        "semantic_evidence": {},
        "current_analysis": None,
        "current_strategy": None,
        "phase": "active",
        "turn_count": 0,
        "customer_attitude": "cautious",
        "stagnation_count": 0,
        "max_turns": 10,
        "difficulty": "normal",
    }

    await graph.aupdate_state(config, initial_state)
    print(f"[客户开场] {scenario.opening_message}")
    print()

    # Simulate sales messages
    sales_messages = [
        "张主任您好！我们这款GLP-1受体激动剂在III期临床试验中表现非常出色，患者的HbA1c平均降幅达到了1.5%，糖化血红蛋白达标率显著优于对照组。",
        "关于安全性方面，这款药在老年亚组中的低血糖发生率仅为1.8%，显著低于传统的磺脲类药物，对您科室那些年纪较大的患者非常友好。",
        "而且这款药的一大优势是每周只需皮下注射一次，大大减少了患者的注射负担，对提升用药依从性帮助很大。",
    ]

    for i, msg in enumerate(sales_messages, 1):
        print(f"{'='*60}")
        print(f"第 {i} 轮")
        print(f"{'='*60}")
        print(f"\n[销售] {msg}\n")

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=msg)]},
            config,
        )

        # Show analysis
        analysis = result.get("current_analysis", {})
        print("[Evaluator 语义分析]")
        for p in analysis.get("analysis", []):
            status = "✓" if p.get("newly_matched") and p.get("match_level") == "full" else "✗"
            print(f"  {status} {p.get('point_id')}: confidence={p.get('confidence', 0):.2f}, level={p.get('match_level', 'none')}")

        # Show multi-dimensional scoring
        eq = analysis.get("expression_quality", {})
        if isinstance(eq, dict) and "data_citation" in eq:
            print("\n[表达质量多维评分]")
            for dim in ["data_citation", "customer_relevance", "fab_structure", "interaction"]:
                dim_data = eq.get(dim, {})
                if isinstance(dim_data, dict):
                    score = dim_data.get("score", "N/A")
                    note = dim_data.get("note", "")
                    dim_label = {
                        "data_citation": "数据引用",
                        "customer_relevance": "客户关联",
                        "fab_structure": "FAB结构",
                        "interaction": "互动技巧",
                    }.get(dim, dim)
                    print(f"  {dim_label}: {score}/5 - {note}")
            overall = eq.get("overall_score", "N/A")
            print(f"  综合评分: {overall}/5")
            if eq.get("quality_note"):
                print(f"  总评: {eq['quality_note']}")
        else:
            print(f"\n[表达质量] {eq}")

        # Show coverage
        coverage = result["semantic_coverage"]
        covered = sum(coverage.values())
        total = len(coverage)
        print(f"\n[覆盖状态] {covered}/{total} | attitude={result['customer_attitude']} | phase={result['phase']}")
        for pid, is_covered in coverage.items():
            print(f"  {'✓' if is_covered else '○'} {pid}")

        # Show AI response
        for m in reversed(result["messages"]):
            if isinstance(m, AIMessage):
                print(f"\n[客户回复] {m.content}")
                break

        # Show strategy
        strategy = result.get("current_strategy", {})
        if strategy:
            print(f"\n[策略] target={strategy.get('target_point')}, strategy={strategy.get('strategy')}")
        print()

        if result["phase"] == "completed":
            print("=== 对话结束 ===")
            break

    print("=== E2E 测试完成 ===")


if __name__ == "__main__":
    asyncio.run(main())
