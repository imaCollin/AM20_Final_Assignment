# from src.rag_chatbot.chatbot import build_app


# demo = build_app()


# if __name__ == "__main__":
#     demo.launch()
import os

from src.rag_chatbot.chatbot import build_app


demo = build_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
