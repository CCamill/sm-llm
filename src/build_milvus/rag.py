"""
RAG 生成函数 + 可信度解释（简化版，一文件实现）
- 说明：尽量用学生风格、简单函数名、少封装。
- 依赖：langchain, langchain-openai, pydantic, tiktoken（可选），你的向量库已就绪，并能提供 retriever。
- 启动：在 __main__ 里看示例，把 my_retriever 换成你已有的 retriever。
"""

from typing import List, Dict, Any
from dataclasses import dataclass

# LLM & LangChain 组件
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# ========== 配置 ==========
# 请先把 OPENAI_API_KEY 配成环境变量；或把 ChatOpenAI(..., api_key="...")
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.2
TOP_K_DOCS = 5

# ========== 数据结构 ==========
@dataclass
class GenResult:
    function_code: str
    explanation: str
    citations: List[Dict[str, Any]]
    judge_score: int
    judge_reason: str
    judge_is_grounded: bool

# ========== 小工具 ==========

def join_docs(docs: List[Document]) -> str:
    """把检索到的文档拼成一个简单的上下文，附上 source/id 方便引用。"""
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source") or d.metadata.get("id") or f"doc_{i}"
        parts.append(f"[SOURCE:{src}]\n{d.page_content}")
    return "\n\n".join(parts)

# ========== 提示词 ==========

# 生成函数的提示：
make_fn_prompt = ChatPromptTemplate.from_template(
    """
你是一个写 Python 小函数的助教。根据用户问题和给定的资料，写出**一个**可运行的 Python 函数，并给出简短解释与引用。
要求：
1) 只输出一个函数定义，函数名尽量短（如 calc, run, solve）。
2) 不要依赖网络或外部服务；只可用标准库，尽量少 import。
3) 函数应与用户意图直接相关；如果资料不支持，请做最小可行实现并标注假设。
4) 同时给出 citations（用到的 SOURCE 标识），并解释为什么这些资料能支撑实现。

返回 JSON：
{
  "function_code": "...python 函数代码...",
  "explanation": "用中文，100 字内",
  "citations": [{"source_id": "...", "why": "..."}]
}

# 用户问题
{question}

# 资料（带 SOURCE 标签）：
{context}
    """
)

# 评审/打分提示：
judge_prompt = ChatPromptTemplate.from_template(
    """
你是严格的代码评审。请根据**用户问题**与**资料**，对候选函数进行可信度评估：
- 关注：是否与问题匹配、是否被资料支撑、是否有明显漏洞/不当假设。
- 分数 0~100：80 以上表示较可信且与资料一致。

请返回 JSON：
{
  "score": 0-100,
  "is_grounded": true/false,
  "reason": "用中文，80 字内"
}

# 用户问题
{question}

# 资料
{context}

# 候选函数
```python
{function_code}
```
    """
)

# ========== 构建链 ==========

def build_chain(my_retriever) -> RunnableLambda:
    """把检索、生成、评审串起来。my_retriever 需实现 .get_relevant_documents(query) 或为 LangChain retriever。"""

    # 1) 检索
    def fetch_docs(q: str) -> Dict[str, Any]:
        docs = my_retriever.get_relevant_documents(q) if hasattr(my_retriever, "get_relevant_documents") else my_retriever.invoke(q)
        if not isinstance(docs, list):
            docs = []
        docs = docs[:TOP_K_DOCS]
        ctx = join_docs(docs) if docs else "(无检索结果)"
        return {"question": q, "context": ctx}

    # 2) 让 LLM 生成函数(JSON)
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    gen_parser = JsonOutputParser()

    gen_chain = (
        RunnableLambda(fetch_docs)
        | make_fn_prompt
        | llm
        | gen_parser
    )

    # 3) 评审打分(JSON)
    judge_parser = JsonOutputParser()

    def judge_input(x: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "question": x["question"],
            "context": x["context"],
            "function_code": x["function_code"],
        }

    judge_chain = (
        RunnablePassthrough()
        | RunnableLambda(judge_input)
        | judge_prompt
        | llm
        | judge_parser
    )

    # 4) 合并输出为 GenResult
    def pack(x: Dict[str, Any]) -> GenResult:
        score = int(x.get("score", 0)) if isinstance(x, dict) else 0
        return GenResult(
            function_code=x.get("function_code", ""),
            explanation=x.get("explanation", ""),
            citations=x.get("citations", []),
            judge_score=score,
            judge_reason=x.get("reason", ""),
            judge_is_grounded=bool(x.get("is_grounded", False)),
        )

    # 总链：先生成，再把生成结果 + 原始 question/context 喂给评审，再打包
    def run_all(q: str) -> GenResult:
        base = gen_chain.invoke(q)
        # 将 question/context 合并进去，供评审
        # 为简单起见，再算一次 fetch_docs（避免在不同 Runnable 之间传复杂对象）
        qc = fetch_docs(q)
        gen_with_qc = {**base, **qc}
        judge_out = judge_chain.invoke(gen_with_qc)
        final = {**base, **judge_out}
        return pack(final)

    return RunnableLambda(run_all)

# ========== 简易打印 ==========

def pretty_print(res: GenResult):
    print("\n===== 生成的函数 =====\n")
    print(res.function_code)
    print("\n===== 解释 =====\n")
    print(res.explanation)
    if res.citations:
        print("\n===== 引用（来自检索 SOURCE） =====")
        for c in res.citations:
            print(f"- {c.get('source_id')}: {c.get('why')}")
    print("\n===== 可信度评审 =====")
    print(f"分数: {res.judge_score} / 100  |  Grounded: {res.judge_is_grounded}")
    print(f"理由: {res.judge_reason}")

# ========== 示例入口 ==========
if __name__ == "__main__":
    """
    使用说明：
    1) 准备好你的 retriever（如 FAISS/Chroma 等），命名为 my_retriever。
    2) 运行本脚本，输入自然语言需求，它会：检索 -> 生成函数 -> 可信度评审 -> 打印结果。
    3) 生成的函数仅作为起点，请在你项目中根据需要保存到文件并加测例。
    """
    import os

    # ======== 示例占位 ========
    # 这里用一个极简的假 retriever 以便单文件可跑；实际请换成你的向量检索器。
    class DummyRetriever:
        def __init__(self):
            self.docs = [
                Document(page_content="给定名义人力资本存量 H_t，可用折旧率 δ 调整为有效存量：H_t_eff = (1-δ) * H_{t-1} + I_t。", metadata={"source": "note_hc_depr"}),
                Document(page_content="常见 δ 在 1%~5%/年之间，经验研究多取 2%~3%。", metadata={"source": "paper_QXY_2019"}),
            ]
        def invoke(self, q):
            return self.docs

    my_retriever = DummyRetriever()

    chain = build_chain(my_retriever)

    print("输入一个需求，例如：'写个函数，用折旧率把上一期人力资本和本期投资合成为本期有效存量'\n")
    try:
        user_q = input("你的需求: ")
    except EOFError:
        user_q = "写个函数，用折旧率把上一期人力资本和本期投资合成为本期有效存量"

    result: GenResult = chain.invoke(user_q)
    pretty_print(result)
