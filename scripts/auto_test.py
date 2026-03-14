#!/usr/bin/env python3
"""
AI Sales Trainer 自动化测试脚本

通过 HTTP API 自动执行一次完整的训练对话，导出对话记录和报告。
用于 Claude 自驱动系统质量迭代。

用法:
  uv run python scripts/auto_test.py [选项]

选项:
  --scenario    场景ID（默认 diabetes_drug）
  --difficulty  难度 easy/normal/hard（默认 normal）
  --output      输出目录（默认 test_results/）
  --messages    自定义消息JSON文件路径（可选）
  --base-url    API地址（默认 http://localhost:8000）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

# ── 内置测试话术（diabetes_drug 场景） ──────────────────────────────

DEFAULT_MESSAGES = [
    {
        "target": "SP1 (HbA1c改善)",
        "content": (
            "张主任您好！我们这款GLP-1受体激动剂在III期临床试验中表现非常出色。"
            "以SUSTAIN系列研究为例，患者使用后HbA1c平均降幅达到1.5%，"
            "糖化血红蛋白达标率（<7%）显著优于对照组。"
            "对您科室里那些糖化控制不达标的患者，这个数据应该很有参考价值。"
            "您目前患者的糖化达标情况怎么样？"
        ),
    },
    {
        "target": "SP2 (低血糖安全)",
        "content": (
            "您提到的这个顾虑非常重要。安全性方面，这款药的一大优势就是低血糖风险极低。"
            "在临床试验中，低血糖事件发生率仅1.8%，特别是在老年亚组中，"
            "低血糖风险显著低于磺脲类药物。对于您管理的高龄糖尿病患者来说，"
            "这意味着可以放心降糖而不用过分担心低血糖事件。"
            "张主任您觉得这个安全性数据对您的临床决策有帮助吗？"
        ),
    },
    {
        "target": "SP3 (用药便利)",
        "content": (
            "除了疗效和安全性，这款药在用药便利性上也是一大亮点。"
            "它只需要每周皮下注射一次，相比每天注射的方案，大大减少了注射负担。"
            "我们的临床数据显示，这种一周一针的方案显著提升了患者的用药依从性，"
            "对长期血糖管理非常有利。特别是对于那些因为怕打针而不愿意用注射剂的患者，"
            "接受度会高很多。您觉得在您的科室推广一周一次的方案，患者会更容易接受吗？"
        ),
    },
]


def api(client: httpx.Client, method: str, path: str, **kwargs) -> dict:
    resp = client.request(method, path, **kwargs)
    if resp.status_code in (200, 201):
        return resp.json()
    if resp.status_code == 204:
        return {}
    print(f"  ❌ API 错误: {method} {path} → {resp.status_code}")
    print(f"     {resp.text[:500]}")
    sys.exit(1)


def format_coverage(coverage: dict) -> str:
    parts = []
    for pid in sorted(coverage.keys()):
        mark = "✓" if coverage[pid] else "✗"
        parts.append(f"{pid} {mark}")
    return " | ".join(parts)


def format_quality(analysis: dict) -> str:
    eq = analysis.get("expression_quality") if analysis else None
    if not eq:
        return "无评分"
    dims = {
        "data_citation": "数据引用",
        "customer_relevance": "客户相关",
        "fab_structure": "FAB结构",
        "interaction": "互动技巧",
    }
    parts = []
    for key, label in dims.items():
        dim = eq.get(key)
        if dim:
            parts.append(f"{label} {dim.get('score', '?')}/5")
    return ", ".join(parts) if parts else "无评分"


def run_test(
    base_url: str,
    scenario_id: str,
    difficulty: str,
    messages: list[dict],
    output_dir: Path,
):
    client = httpx.Client(base_url=base_url, timeout=300)

    # ── 1. 验证场景 ──
    print(f"\n{'='*60}")
    print(f"  AI Sales Trainer 自动化测试")
    print(f"  场景: {scenario_id} | 难度: {difficulty}")
    print(f"{'='*60}\n")

    scenarios = api(client, "GET", "/api/v1/scenarios")["scenarios"]
    scenario_names = {s["id"]: s["name"] for s in scenarios}
    if scenario_id not in scenario_names:
        print(f"❌ 场景 '{scenario_id}' 不存在。可用: {list(scenario_names.keys())}")
        sys.exit(1)

    scenario_name = scenario_names[scenario_id]
    print(f"✓ 场景确认: {scenario_name}\n")

    # ── 2. 创建会话 ──
    session_data = api(
        client, "POST", "/api/v1/sessions",
        json={"scenario_id": scenario_id, "difficulty": difficulty},
    )
    session_id = session_data["session_id"]
    opening = session_data["opening_message"]["content"]
    print(f"✓ 会话创建: {session_id}\n")
    print(f"  👤 客户(开场白): {opening}\n")

    # ── 3. 逐轮发送消息 ──
    rounds = []
    for i, msg_info in enumerate(messages, 1):
        target = msg_info.get("target", f"第{i}轮")
        content = msg_info["content"]

        print(f"{'─'*60}")
        print(f"  第 {i} 轮 — 目标: {target}")
        print(f"{'─'*60}")
        print(f"  🧑‍💼 销售: {content[:80]}...")

        t0 = time.time()
        result = api(
            client, "POST", f"/api/v1/sessions/{session_id}/messages",
            json={"content": content},
        )
        elapsed = time.time() - t0

        ai_reply = result["assistant_message"]["content"]
        state = result["state"]
        analysis = result["user_message"].get("analysis", {})

        print(f"  👤 客户: {ai_reply}")
        print(f"  ⏱  耗时: {elapsed:.1f}s")
        print(f"  📊 覆盖: {format_coverage(state['semantic_coverage'])}")
        print(f"  😀 态度: {state['customer_attitude']}")
        print(f"  📝 评分: {format_quality(analysis)}")
        print(f"  🎯 阶段: {state['phase']} (第{state['turn_count']}轮)")
        print()

        rounds.append({
            "turn": i,
            "target": target,
            "user_content": content,
            "ai_reply": ai_reply,
            "state": state,
            "analysis": analysis,
            "elapsed_seconds": round(elapsed, 1),
        })

        if state["phase"] == "completed":
            print("  ✅ 对话已完成！\n")
            break

    # ── 4. 获取报告 ──
    print(f"{'─'*60}")
    print("  获取训练报告...")

    try:
        report = api(client, "GET", f"/api/v1/sessions/{session_id}/report")
        has_report = True
    except SystemExit:
        print("  ⚠ 报告生成失败（可能对话未完成）")
        report = None
        has_report = False

    # ── 5. 导出对话记录 ──
    conversation = api(client, "GET", f"/api/v1/sessions/{session_id}/messages")

    # ── 6. 保存结果 ──
    result_dir = output_dir / session_id
    result_dir.mkdir(parents=True, exist_ok=True)

    (result_dir / "conversation.json").write_text(
        json.dumps(conversation, ensure_ascii=False, indent=2)
    )

    if has_report:
        (result_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )

    # ── 7. 生成 summary.md ──
    md = generate_summary_md(
        scenario_name=scenario_name,
        scenario_id=scenario_id,
        difficulty=difficulty,
        session_id=session_id,
        opening=opening,
        rounds=rounds,
        report=report,
    )
    (result_dir / "summary.md").write_text(md)

    print(f"\n{'='*60}")
    print(f"  测试完成!")
    print(f"  结果目录: {result_dir}")
    print(f"  - conversation.json  (完整对话记录)")
    if has_report:
        print(f"  - report.json        (训练报告)")
    print(f"  - summary.md         (可读摘要)")
    print(f"{'='*60}\n")

    # 打印 summary.md 到控制台方便 Claude 直接读取
    print(md)

    return str(result_dir)


def generate_summary_md(
    *,
    scenario_name: str,
    scenario_id: str,
    difficulty: str,
    session_id: str,
    opening: str,
    rounds: list[dict],
    report: dict | None,
) -> str:
    lines = [
        f"# 测试报告: {scenario_name}",
        f"- 场景: {scenario_id} | 难度: {difficulty}",
        f"- 会话: {session_id}",
        f"- 总轮次: {len(rounds)}",
        "",
        "## 对话记录",
        "",
        f"**客户(开场白)**: {opening}",
        "",
    ]

    for r in rounds:
        lines.append(f"### 第 {r['turn']} 轮 — {r['target']}")
        lines.append(f"**销售**: {r['user_content']}")
        lines.append("")
        lines.append(f"**客户**: {r['ai_reply']}")
        lines.append("")
        cov = format_coverage(r["state"]["semantic_coverage"])
        att = r["state"]["customer_attitude"]
        score = format_quality(r["analysis"])
        lines.append(f"> 覆盖: {cov} | 态度: {att} | 评分: {score} | 耗时: {r['elapsed_seconds']}s")
        lines.append("")

    if report:
        lines.append("## 训练报告")
        s = report.get("summary", {})
        lines.append(
            f"- 覆盖率: {s.get('coverage_rate', 0) * 100:.0f}% "
            f"| 效率分: {s.get('efficiency_score', 0):.1%} "
            f"| 总分: {s.get('overall_score', 0):.0f}"
        )
        lines.append(f"- 总轮次: {s.get('total_turns', '?')}")
        lines.append("")

        radar = report.get("skill_radar", {})
        if radar:
            dims = {
                "data_citation": "数据引用",
                "customer_relevance": "客户相关性",
                "fab_structure": "FAB结构",
                "interaction": "互动技巧",
            }
            parts = [f"{label} {radar.get(k, 0):.1f}" for k, label in dims.items()]
            lines.append(f"- 技能雷达: {', '.join(parts)}")
            lines.append("")

        fb = report.get("feedback", {})
        if fb.get("strengths"):
            lines.append("### 优势")
            for item in fb["strengths"]:
                lines.append(f"- {item}")
            lines.append("")
        if fb.get("improvements"):
            lines.append("### 改进建议")
            for item in fb["improvements"]:
                lines.append(f"- {item}")
            lines.append("")
        if fb.get("overall"):
            lines.append(f"### 总评\n{fb['overall']}")
            lines.append("")

        # 语义点明细
        details = report.get("semantic_detail", [])
        if details:
            lines.append("### 语义点覆盖明细")
            for d in details:
                mark = "✅" if d.get("covered") else "❌"
                turn_info = f"第{d['covered_at_turn']}轮" if d.get("covered_at_turn") else "未覆盖"
                conf = f"置信度 {d['confidence']:.0%}" if d.get("confidence") else ""
                lines.append(f"- {mark} **{d.get('name', d.get('point_id', '?'))}** — {turn_info} {conf}")
                if d.get("evidence"):
                    lines.append(f"  > 证据: {d['evidence']}")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="AI Sales Trainer 自动化测试")
    parser.add_argument("--scenario", default="diabetes_drug", help="场景ID")
    parser.add_argument("--difficulty", default="normal", choices=["easy", "normal", "hard"])
    parser.add_argument("--output", default="test_results", help="输出目录")
    parser.add_argument("--messages", default=None, help="自定义消息JSON文件路径")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API地址")
    args = parser.parse_args()

    if args.messages:
        with open(args.messages) as f:
            messages = json.load(f)
    else:
        messages = DEFAULT_MESSAGES

    run_test(
        base_url=args.base_url,
        scenario_id=args.scenario,
        difficulty=args.difficulty,
        messages=messages,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
