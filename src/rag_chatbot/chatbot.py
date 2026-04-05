from __future__ import annotations

from typing import Any

import gradio as gr
from litellm import completion

from src.rag_chatbot.config import settings
from src.rag_chatbot.retrieval import RetrievedChunk, build_context_block, retrieve_context


def format_citations(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "\n\nSources: none"
    unique_sources: list[str] = []
    for chunk in chunks:
        citation = f"{chunk.source} (offset {chunk.start_index})"
        if citation not in unique_sources:
            unique_sources.append(citation)
    lines = "\n".join(f"- {source}" for source in unique_sources)
    return f"\n\nSources:\n{lines}"


def generate_answer(question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    retrieved = retrieve_context(question)
    context = build_context_block(retrieved)

    if not retrieved:
        answer = (
            "I cannot find a supported answer in the knowledge base for that question."
        )
        return {
            "answer": answer + format_citations(retrieved),
            "citations": retrieved,
            "used_context": context,
        }

    messages: list[dict[str, str]] = [{"role": "system", "content": settings.system_prompt}]
    for turn in history or []:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append(
        {
            "role": "user",
            "content": (
                "Answer the question using only the context below.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\n\n"
                "If the context is insufficient, say that you cannot find the answer "
                "in the knowledge base."
            ),
        }
    )

    try:
        response = completion(
            model=settings.litellm_model,
            api_key=settings.litellm_api_key or None,
            messages=messages,
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as exc:
        answer = (
            "The retrieval step succeeded, but the language model call failed. "
            f"Error: {exc}"
        )
    return {
        "answer": answer + format_citations(retrieved),
        "citations": retrieved,
        "used_context": context,
    }


def chat(message: str, history: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    result = generate_answer(message, history)
    history = history + [{"user": message, "assistant": result["answer"]}]
    return history, result["used_context"]


def to_chatbot_messages(history: list[dict[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    return messages


def clear_chat() -> tuple[list[dict[str, str]], str]:
    return [], ""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RAG Knowledge Base Chatbot") as demo:
        gr.Markdown(
            """
            # RAG Knowledge Base Chatbot
            Ask questions about the GitLab Handbook knowledge base. The app retrieves
            relevant Markdown chunks from ChromaDB and answers with source citations.
            """
        )
        chatbot = gr.Chatbot(type="messages", height=500)
        state = gr.State([])
        context_box = gr.Textbox(label="Retrieved Context", lines=14)
        msg = gr.Textbox(label="Your Question", placeholder="Ask a handbook question...")
        submit = gr.Button("Send")
        clear = gr.Button("Clear")

        def _submit(user_message: str, history: list[dict[str, str]]):
            if not user_message.strip():
                return to_chatbot_messages(history), history, "", ""
            updated_history, context = chat(user_message, history)
            return to_chatbot_messages(updated_history), updated_history, context, ""

        submit.click(
            _submit,
            inputs=[msg, state],
            outputs=[chatbot, state, context_box, msg],
        )
        msg.submit(
            _submit,
            inputs=[msg, state],
            outputs=[chatbot, state, context_box, msg],
        )
        clear.click(clear_chat, outputs=[state, context_box]).then(
            lambda: [], outputs=[chatbot]
        )

    return demo
