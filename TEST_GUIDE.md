# AI Sales Trainer 测试指南

## 环境

- 后端地址：`http://localhost:8000`
- 健康检查：`GET http://localhost:8000/health`

## 测试流程

按顺序执行以下步骤，每一步都用 curl 调用。

---

### 1. 注册用户

```bash
curl -s http://localhost:8000/api/v1/auth/register \
  -X POST -H 'Content-Type: application/json' \
  -d '{"username":"tester","password":"test1234","display_name":"测试员"}'
```

返回 `{ "token": "eyJ...", "user": {...} }`，**保存 token 值**，后续请求需要带上。

### 2. 登录（如已注册）

```bash
curl -s http://localhost:8000/api/v1/auth/login \
  -X POST -H 'Content-Type: application/json' \
  -d '{"username":"tester","password":"test1234"}'
```

同样返回 token。

### 3. 查看可用场景

```bash
curl -s http://localhost:8000/api/v1/scenarios
```

返回 `{ "scenarios": [...] }`。当前有两个场景：
- `diabetes_drug` — 降糖药，3 个语义点，简单
- `oncology_immunotherapy` — PD-1 肿瘤推广，5 个语义点，复杂

### 4. 创建训练会话

```bash
curl -s http://localhost:8000/api/v1/sessions \
  -X POST -H 'Content-Type: application/json' \
  -d '{"scenario_id":"oncology_immunotherapy","difficulty":"normal"}'
```

返回中包含：
- `session_id` — **保存这个值**，后续所有对话需要
- `opening_message.content` — 客户的开场白
- `semantic_coverage` — 初始覆盖状态（全 false）

### 5. 发送消息（非流式，推荐测试用）

```bash
curl -s http://localhost:8000/api/v1/sessions/{session_id}/messages \
  -X POST -H 'Content-Type: application/json' \
  -d '{"content":"您好陈教授，我是XX药业的销售代表，今天想跟您交流一下我们的国产PD-1单抗。"}'
```

返回 JSON 包含：
- `assistant_message.content` — 客户回复
- `state.semantic_coverage` — 语义覆盖更新
- `state.customer_attitude` — 客户态度（cautious/interested/convinced）
- `state.phase` — 阶段（active/wrapping_up/completed）
- `user_message.analysis` — 销售表达的评估结果

### 6. 多轮对话

重复步骤 5，每次发不同的销售话术。建议按以下顺序逐步覆盖 5 个语义点：

**第 1 轮**（自我介绍 + 引出产品）：
```json
{"content":"您好陈教授，我是XX药业的销售代表，今天想跟您交流一下我们最新的PD-1单抗在NSCLC领域的临床数据。"}
```

**第 2 轮**（SP1 疗效数据）：
```json
{"content":"在我们的头对头III期注册研究中，针对晚期NSCLC，ORR达到39.2%，非劣效于对照组的37.8%。中位PFS 11.3个月，HR=0.93，95%CI在1.0以内，达到了预设的非劣效终点。"}
```

**第 3 轮**（SP2 安全性）：
```json
{"content":"安全性方面，得益于我们独特的Fc段改造技术，3-5级irAE发生率为12.3%，显著低于对照组的18.1%。免疫性肺炎发生率仅3.2%，因不良反应导致的停药率只有5.8%。我们还配套提供完整的irAE分级管理手册。"}
```

**第 4 轮**（SP3 真实世界数据）：
```json
{"content":"关于真实世界证据，我们已完成覆盖全国78家中心、3200例中国NSCLC患者的RWS研究。其中35%是70岁以上老年患者，22%是PS评分2分的患者，这些都是注册临床中较少纳入的人群。真实世界的ORR为34.6%，中位PFS 9.8个月，数据已在ASCO和ESMO发表。"}
```

**第 5 轮**（SP4 医保经济学）：
```json
{"content":"费用方面，我们已纳入国家医保目录，患者年治疗费用降至约3.8万元，相比进口PD-1降低约60%。药经分析显示ICER值远低于3倍GDP/QALY阈值。医保报销后患者月均自付不到1000元。"}
```

**第 6 轮**（SP5 联合用药）：
```json
{"content":"联合用药方面，我们联合培美曲塞+铂类一线治疗非鳞NSCLC的III期研究ORR达56.7%，中位PFS 13.2个月。目前已获批联合化疗一线治疗鳞状和非鳞状NSCLC两个适应症。联合贝伐和化疗的四药方案I/II期数据ORR高达63%。"}
```

### 7. 检查会话状态

```bash
curl -s http://localhost:8000/api/v1/sessions/{session_id}
```

关注：
- `semantic_coverage` — 5 个 SP 是否都变成 true
- `phase` — 是否变为 `completed`
- `customer_attitude` — 是否演进到 `convinced`

### 8. 获取训练报告

```bash
curl -s http://localhost:8000/api/v1/sessions/{session_id}/report
```

返回完整报告：
- `summary.overall_score` — 总分（0-100）
- `summary.coverage_rate` — 覆盖率
- `semantic_detail` — 每个点的覆盖证据
- `skill_radar` — 四维评分
- `feedback` — 优势、改进建议、总结

### 9. 查看历史会话列表

```bash
curl -s http://localhost:8000/api/v1/sessions
```

### 10. 查看某会话的全部消息

```bash
curl -s http://localhost:8000/api/v1/sessions/{session_id}/messages
```

---

## 验证要点

1. **信息对称性**：第 1 轮只打招呼时，客户不应提及具体药物机制或数据
2. **语义覆盖递增**：每轮提供具体数据后，对应 SP 应变为 true
3. **态度渐变**：从 cautious → interested → convinced，不会跳跃或回退
4. **表达质量评分**：模糊表述（"效果不错"）应得低分，带数据的表述应得高分
5. **阶段流转**：全部覆盖后 phase 变为 wrapping_up，客户回复后变为 completed
6. **报告生成**：completed 后能生成包含具体证据的训练报告

## 流式接口（可选）

如果要测试 SSE 流式响应：

```bash
curl -N http://localhost:8000/api/v1/sessions/{session_id}/chat/stream \
  -X POST -H 'Content-Type: application/json' \
  -d '{"content":"您好陈教授"}'
```

会返回 `event: thinking / delta / analysis / metadata / done` 等 SSE 事件。

## 快速一键测试脚本

```bash
# 1. 创建会话
SESSION=$(curl -s http://localhost:8000/api/v1/sessions -X POST \
  -H 'Content-Type: application/json' \
  -d '{"scenario_id":"oncology_immunotherapy","difficulty":"normal"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "Session: $SESSION"

# 2. 依次发送 6 轮消息
MSGS=(
  '您好陈教授，我是XX药业的销售代表，今天想跟您交流一下我们的国产PD-1单抗。'
  '在头对头III期研究中，ORR达39.2%，非劣效于对照组37.8%。中位PFS 11.3个月，HR=0.93。'
  '安全性方面，3-5级irAE发生率12.3%，低于对照组18.1%。Fc段改造使免疫性肺炎仅3.2%。'
  '真实世界研究覆盖全国78家中心3200例患者，ORR 34.6%，中位PFS 9.8个月，已在ASCO发表。'
  '已纳入医保，年治疗费用约3.8万元，比进口降低60%，患者月均自付不到1000元。'
  '联合培美曲塞+铂类一线治疗ORR达56.7%，PFS 13.2个月，已获批两个联合适应症。'
)

for i in "${!MSGS[@]}"; do
  echo ""
  echo "=== 第$((i+1))轮 ==="
  RESULT=$(curl -s http://localhost:8000/api/v1/sessions/$SESSION/messages \
    -X POST -H 'Content-Type: application/json' \
    -d "{\"content\":\"${MSGS[$i]}\"}")
  echo "客户回复: $(echo $RESULT | python3 -c "import sys,json; print(json.load(sys.stdin)['assistant_message']['content'][:100])")"
  echo "覆盖状态: $(echo $RESULT | python3 -c "import sys,json; print(json.load(sys.stdin)['state']['semantic_coverage'])")"
  echo "客户态度: $(echo $RESULT | python3 -c "import sys,json; print(json.load(sys.stdin)['state']['customer_attitude'])")"
  echo "阶段: $(echo $RESULT | python3 -c "import sys,json; print(json.load(sys.stdin)['state']['phase'])")"
done

# 3. 获取报告
echo ""
echo "=== 训练报告 ==="
curl -s http://localhost:8000/api/v1/sessions/$SESSION/report | python3 -m json.tool
```
