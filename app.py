# from src.rag_chatbot.chatbot import build_app


# demo = build_app()


# if __name__ == "__main__":
#     demo.launch()
# import os

# from src.rag_chatbot.chatbot import build_app


# demo = build_app()


# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 7860))
#     demo.launch(server_name="0.0.0.0", server_port=port)
import os

from src.rag_chatbot.chatbot import build_app

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
