import sys
import torch
from PyQt5 import QtWidgets, QtCore
from transformers import T5Tokenizer, T5ForConditionalGeneration
from markdown_it import MarkdownIt
import structlog
import logging
from rich.logging import RichHandler

# Configure rich logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(),  # Human-readable logs
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class TranslateMadlad(QtCore.QObject):
    def __init__(self, model_name="jbochi/madlad400-3b-mt"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        try:
            logger.info("Loading translation model...", model_name=self.model_name, device=self.device)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error("Failed to load model or tokenizer", error=str(e))
            raise

    def translate(self, text):
        if not text.strip():
            return ""
        try:
            if self.model is None or self.tokenizer is None:
                logger.error("Model or tokenizer is not loaded")
                return "[Translation Error: Model not loaded]"

            logger.debug("Translating text", text_preview=text[:100])
            input_ids = self.tokenizer(f"<2en> {text}", return_tensors="pt", max_length=512, truncation=True).input_ids.to(self.device)
            outputs = self.model.generate(input_ids, max_length=512)
            translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug("Translation complete", translated_text_preview=translated_text[:100])
            return translated_text
        except Exception as e:
            logger.error("Translation error", error=str(e))
            return "[Translation Error]"

class MarkdownUtils:
    @staticmethod
    def extract_markdown_text(tokens):
        """Extract text content from markdown tokens."""
        text_blocks = []
        for token in tokens:
            if token.type == "inline" and token.children:
                text_blocks.append(" ".join(child.content for child in token.children if child.type == "text"))
            elif token.type == "text":
                text_blocks.append(token.content)
        return text_blocks

    @staticmethod
    def reassemble_markdown(tokens, translated_texts):
        """Reassemble markdown with improved handling for lists, tables, and mermaid blocks."""
        text_index = 0
        markdown_lines = []
        ordered_list_index = 1
        list_stack = []

        logger.info("Starting markdown reassembly.")

        for token in tokens:
            try:
                if token.type == "heading":
                    markdown_lines.append(f"{'#' * token.level} {translated_texts[text_index]}")
                    text_index += 1
                    logger.debug("Formatted heading", heading=markdown_lines[-1])

                elif token.type == "paragraph":
                    markdown_lines.append(translated_texts[text_index])
                    text_index += 1
                    logger.debug("Formatted paragraph", paragraph=markdown_lines[-1])

                elif token.type == "list_item":
                    # Process list items (ordered or unordered)
                    if token.info:
                        formatted_item = f"{ordered_list_index}. {translated_texts[text_index]}"
                        ordered_list_index += 1
                    else:
                        formatted_item = f"- {translated_texts[text_index]}"
                    markdown_lines.append(formatted_item)
                    text_index += 1
                    logger.debug("Formatted list item", list_item=formatted_item)

                elif token.type == "table":
                    # Process tables
                    headers = "|".join(translated_texts[text_index:text_index + len(token.header.children)])
                    separator = "|".join(["---"] * len(token.header.children))
                    markdown_lines.append(f"|{headers}|")
                    markdown_lines.append(f"|{separator}|")
                    text_index += len(token.header.children)

                    for row in token.children:
                        row_content = "|".join([translated_texts[text_index + i] for i in range(len(row.children))])
                        markdown_lines.append(f"|{row_content}|")
                        text_index += len(row.children)
                        logger.debug("Formatted table row", row=row_content)

                elif token.type == "fence":
                    code_block = f"```{token.info}\n{token.content}\n```"
                    markdown_lines.append(code_block)
                    logger.debug("Formatted code block", code_block=code_block)

                elif token.type == "inline":
                    inline_text = translated_texts[text_index]
                    markdown_lines.append(inline_text)
                    text_index += 1
                    logger.debug("Formatted inline text", inline_text=inline_text)

                elif token.type == "text":
                    plain_text = translated_texts[text_index]
                    markdown_lines.append(plain_text)
                    text_index += 1
                    logger.debug("Formatted plain text", plain_text=plain_text)

            except Exception as e:
                logger.error("Error formatting token", token_type=token.type, error=str(e))

        rendered_markdown = "\n".join(markdown_lines)
        logger.info("Markdown reassembly complete", markdown_preview=rendered_markdown[:200])
        return rendered_markdown

class FileHandler:
    @staticmethod
    def read_markdown_file(file_path):
        try:
            logger.info("Reading markdown file", file_path=file_path)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                logger.debug("File content read", content_preview=content[:200])
                return content
        except Exception as e:
            logger.error("Error reading file", error=str(e))
            return None

    @staticmethod
    def write_markdown_file(file_path, content):
        try:
            logger.info("Writing to markdown file", file_path=file_path)
            logger.debug("Content being written", content_preview=content[:200])
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content)
        except Exception as e:
            logger.error("Error writing to file", error=str(e))

class TranslationWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int)

    def __init__(self, file_path, translator):
        super().__init__()
        self.file_path = file_path
        self.translator = translator

    def run(self):
        try:
            logger.info("Worker started", file_path=self.file_path)
            content = FileHandler.read_markdown_file(self.file_path)
            if content is None:
                logger.error("No content read from file.")
                self.finished.emit()
                return

            md = MarkdownIt()
            tokens = md.parse(content)
            logger.info("Markdown tokens parsed", token_count=len(tokens))

            text_blocks = MarkdownUtils.extract_markdown_text(tokens)
            total_blocks = len(text_blocks)
            logger.info("Extracted text blocks", total_blocks=total_blocks)

            translated_texts = []
            for idx, block in enumerate(text_blocks):
                try:
                    logger.debug("Translating block", block_index=idx, block_preview=block[:100])
                    translated_text = self.translator.translate(block.strip())
                    translated_texts.append(translated_text)
                    logger.debug("Block translated", translated_text_preview=translated_text[:100])
                    progress = int((idx + 1) / total_blocks * 100)
                    self.progress.emit(progress)
                except Exception as e:
                    logger.error("Error translating block", block_index=idx, error=str(e))
                    translated_texts.append("[Translation Error]")

            translated_markdown = MarkdownUtils.reassemble_markdown(tokens, translated_texts)
            output_path = self.file_path.replace(".md", "-translated.md")
            FileHandler.write_markdown_file(output_path, translated_markdown)

            logger.info("Translation complete", output_path=output_path)
        except Exception as e:
            logger.error("Error during worker execution", error=str(e))
        finally:
            self.finished.emit()

class TranslatorApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Markdown Translator")
        self.setGeometry(100, 100, 500, 300)
        self.setAcceptDrops(True)

        self.label = QtWidgets.QLabel("Drag and drop your Markdown (.md) file here", self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setGeometry(50, 200, 400, 30)
        self.progress_bar.setValue(0)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

        self.translator = TranslateMadlad()
        self.translator.load_model()

        self.worker_thread = QtCore.QThread()
        self.translator.moveToThread(self.worker_thread)
        self.worker_thread.start()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            logger.info("File dropped", file_path=file_path)
            if file_path.endswith(".md"):
                self.label.setText("Processing...")
                self.start_translation(file_path)
            else:
                logger.error("Invalid file type", file_path=file_path)
                self.label.setText("Please drop a valid Markdown (.md) file.")

    def start_translation(self, file_path):
        self.worker = TranslationWorker(file_path, self.translator)
        self.worker.moveToThread(self.worker_thread)
        self.worker.progress.connect(self.update_progress_bar)
        self.worker.finished.connect(self.translation_finished)
        QtCore.QTimer.singleShot(0, self.worker.run)

    def update_progress_bar(self, value):
        logger.debug("Progress bar updated", progress_value=value)
        self.progress_bar.setValue(value)

    def translation_finished(self):
        logger.info("Translation finished")
        self.label.setText("Translation complete! Drag and drop another file.")
        self.progress_bar.setValue(0)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    translator_app = TranslatorApp()
    translator_app.show()
    sys.exit(app.exec_())
