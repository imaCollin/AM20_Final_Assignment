import os

import gradio_client.utils as gradio_client_utils

from src.rag_chatbot.chatbot import build_app


_original_get_type = gradio_client_utils.get_type
_original_json_schema_to_python_type = gradio_client_utils._json_schema_to_python_type


def _safe_get_type(schema):
    # Render hits a Gradio schema parsing edge case where a JSON schema node
    # can be the boolean literal True/False instead of a dict.
    if isinstance(schema, bool):
        return {}
    return _original_get_type(schema)


def _safe_json_schema_to_python_type(schema, defs):
    if isinstance(schema, bool):
        return "Any"
    return _original_json_schema_to_python_type(schema, defs)


gradio_client_utils.get_type = _safe_get_type
gradio_client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type


demo = build_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    host = os.environ.get("HOST", "0.0.0.0")
    on_render = bool(os.environ.get("RENDER"))
    demo.launch(
        server_name=host,
        server_port=port,
        share=on_render,
        show_api=False,
    )
