import sys
import torch
from PyQt5 import QtWidgets, QtCore
from transformers import T5Tokenizer, T5ForConditionalGeneration
import mistune
import structlog
import logging
from rich.logging import RichHandler

# Configure rich logging
logging.basicConfig(
    level=logging.DEBUG,
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
        structlog.dev.ConsoleRenderer(),
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
    def parse_markdown(content):
        """Parse Markdown content into tokens."""
        try:
            markdown = mistune.create_markdown(renderer="ast")
            tokens = markdown(content)
            logger.info("Markdown content parsed successfully", token_count=len(tokens))
            logger.debug("Tokens structure: %s", tokens)
            return tokens
        except Exception as e:
            logger.error("Failed to parse Markdown content", error=str(e))
            return None

    @staticmethod
    def extract_text_blocks(tokens):
        """Extract text blocks from tokens."""
        try:
            text_blocks = []

            # Recursive function to handle nested tokens
            def extract_recursive(token_list):
                for token in token_list:
                    # Check for 'text' or 'raw' fields
                    if 'text' in token:
                        text_blocks.append(token['text'])
                    elif 'raw' in token:
                        text_blocks.append(token['raw'])
                    elif 'children' in token:
                        extract_recursive(token['children'])

            extract_recursive(tokens)

            if text_blocks:
                logger.info("Extracted text blocks successfully", block_count=len(text_blocks))
                return text_blocks
            else:
                logger.error("No translatable text blocks found in tokens.")
                return None
        except Exception as e:
            logger.error("Error extracting text blocks", error=str(e))
            return None

    @staticmethod
    def reassemble_markdown(tokens, translated_texts):
        """Reassemble markdown content."""
        text_index = 0
        markdown_lines = []
        ordered_list_index = 1

        logger.info("Starting markdown reassembly.")
        for token in tokens:
            try:
                if token['type'] == 'heading':
                    level = token.get('level', 1)  # Default to level 1
                    markdown_lines.append(f"{'#' * level} {translated_texts[text_index]}")
                    text_index += 1

                elif token['type'] == 'paragraph':
                    markdown_lines.append(translated_texts[text_index])
                    text_index += 1

                elif token['type'] == 'list_item':
                    if token.get('ordered'):
                        markdown_lines.append(f"{ordered_list_index}. {translated_texts[text_index]}")
                        ordered_list_index += 1
                    else:
                        markdown_lines.append(f"- {translated_texts[text_index]}")
                    text_index += 1

                elif token['type'] == 'fence':
                    code_block = f"```{token.get('info', '')}\n{token.get('text', '')}\n```"
                    markdown_lines.append(code_block)

                elif token['type'] == 'text':
                    markdown_lines.append(translated_texts[text_index])
                    text_index += 1

            except IndexError:
                logger.error("Index error during markdown reassembly", token_type=token['type'], token_content=token)
                markdown_lines.append("[Error: Missing Translated Text]")
            except KeyError as e:
                logger.error("KeyError during markdown reassembly", token_type=token['type'], missing_key=str(e))
            except Exception as e:
                logger.error("Unexpected error during markdown reassembly", token_type=token['type'], error=str(e))

        return "\n".join(markdown_lines)


class FileHandler:
    @staticmethod
    def read_markdown_file(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                logger.info("Markdown file read successfully", file_path=file_path)
                return content
        except Exception as e:
            logger.error("Error reading file", file_path=file_path, error=str(e))
            return None

    @staticmethod
    def write_markdown_file(file_path, content):
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content)
                logger.info("Markdown file written successfully", file_path=file_path)
                return True
        except Exception as e:
            logger.error("Error writing file", file_path=file_path, error=str(e))
            return False


class TranslationWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int)

    def __init__(self, file_path, translator):
        super().__init__()
        self.file_path = file_path
        self.translator = translator

    def run(self):
        logger.info("Translation process started", file_path=self.file_path)
        content = FileHandler.read_markdown_file(self.file_path)
        if not content:
            self.finished.emit()
            return

        tokens = MarkdownUtils.parse_markdown(content)
        if not tokens:
            self.finished.emit()
            return

        text_blocks = MarkdownUtils.extract_text_blocks(tokens)
        if not text_blocks:
            self.finished.emit()
            return

        translated_texts = []
        total_blocks = len(text_blocks)
        for idx, block in enumerate(text_blocks):
            translated_text = self.translator.translate(block)
            translated_texts.append(translated_text)
            self.progress.emit(int((idx + 1) / total_blocks * 100))

        output_content = MarkdownUtils.reassemble_markdown(tokens, translated_texts)
        output_path = self.file_path.replace(".md", "-translated.md")
        success = FileHandler.write_markdown_file(output_path, output_content)

        if success:
            logger.info("Translation complete", output_path=output_path)
        else:
            logger.error("Failed to write output file", output_path=output_path)

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
            if file_path.endswith(".md"):
                self.label.setText("Processing...")
                self.start_translation(file_path)
            else:
                self.label.setText("Please drop a valid Markdown (.md) file.")

    def start_translation(self, file_path):
        self.worker = TranslationWorker(file_path, self.translator)
        self.worker.moveToThread(self.worker_thread)
        self.worker.progress.connect(self.update_progress_bar)
        self.worker.finished.connect(self.translation_finished)
        QtCore.QTimer.singleShot(0, self.worker.run)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def translation_finished(self):
        self.label.setText("Translation complete! Drag and drop another file.")
        self.progress_bar.setValue(0)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    translator_app = TranslatorApp()
    translator_app.show()
    sys.exit(app.exec_())
