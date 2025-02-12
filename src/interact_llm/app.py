"""
Initial inspiration:https://textual.textualize.io/blog/2024/09/15/anatomy-of-a-textual-user-interface/#were-in-the-pipe-five-by-five
"""

from pathlib import Path

from interfaces.base_hf import ChatHF
from interfaces.chat import ChatHistory, ChatMessage
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Markdown
from transformers.utils.logging import disable_progress_bar

disable_progress_bar()

# temp
MODEL_ID = "BSC-LT/salamandra-2b-instruct"
CACHE_DIR = Path(__file__).parents[3] / "models"  # consider making it an argparse
DEVICE = None # "mps" 
WORDS = ["rojo", "azul"] 
BIAS = 7.0

# classes for formatting
class UserMessage(Markdown):
    pass


class Response(Markdown):
    BORDER_TITLE = "Interact-LLM"


class ChatApp(App):
    """
    Texttual app for chatting with llm
    """

    AUTO_FOCUS = "INPUT"

    CSS = """
    UserMessage {
        background: $primary 10%;
        color: $text;
        margin: 1;        
        margin-right: 8;
        padding: 1 2 0 2;
    }

    Response {
        border: wide $success;
        background: $success 10%;   
        color: $text;             
        margin: 1;      
        margin-left: 8; 
        padding: 1 2 0 2;
    }
    """

    def __init__(self):
        super().__init__()
        self.chat_history = ChatHistory(messages=[])

    def on_mount(self) -> None:
        self.model = ChatHF(model_id=MODEL_ID, cache_dir=CACHE_DIR, device=DEVICE)
        self.model.load()

    def update_chat_history(self, chat_message: ChatMessage) -> None:
        """Update chat history with a single new message."""
        self.chat_history.messages.append(chat_message)

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="chat-view"):
            yield Response("¿Hola quieres practicar conmigo?")
        yield Input(placeholder="Escribe tu mensaje aquí")
        yield Footer()

    @on(Input.Submitted)
    async def on_input(self, user_message: Input.Submitted) -> None:
        chat_view = self.query_one("#chat-view")
        user_message.input.clear()
        await chat_view.mount(UserMessage(user_message.value))
        await chat_view.mount(response := Response())
        response.anchor()

        self.get_model_response(user_message.value, response)

    @work(thread=True)
    def get_model_response(self, user_message: str, response: Response) -> None:
        """
        Displays model response to user message, updating chat history
        """
        self.update_chat_history(ChatMessage(role="user", content=user_message))

        model_response = self.model.generate(self.chat_history, words=WORDS, bias=BIAS)

        # replace weird <|im_end|>
        model_response.content = model_response.content.replace("<|im_end|>", "")

        # display in APP
        response_content = ""
        for chunk in model_response.content:
            response_content += chunk  # add words in a "stream-like" way
            self.call_from_thread(response.update, response_content)

        # update history again with model response
        self.update_chat_history(model_response)


def main():
    app = ChatApp()
    app.run()


if __name__ == "__main__":
    main()
