import os
import sys
import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from PyQt5 import QtWidgets, QtGui, QtCore
from rich.console import Console
from dotenv import load_dotenv
from markdown_it import MarkdownIt
from markdown_it.token import Token
import concurrent.futures

# Load environment variables
load_dotenv()

# Instantiate console for logging with Rich
console = Console()

# Setting up constants for the Model
MODEL_NAME = "jbochi/madlad400-10b-mt"
CHUNK_SIZE = 512
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 128))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 400))  # Increase the max new tokens for longer outputs
NO_REPEAT_NGRAM_SIZE = int(os.getenv("NO_REPEAT_NGRAM_SIZE", 3))  # Prevent repetition of 3-grams
MIN_LENGTH = int(os.getenv("MIN_LENGTH", 50))  # Set a minimum length for translation output

# Load Model and Tokenizer, Setting Device to CUDA if Available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Translator function to translate a given list of sentences using MADLAD model
def translate_madlad(sentences, model, tokenizer):
    translated_sentences = []
    for sentence in sentences:
        try:
            text = "<2en> " + sentence
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(
                input_ids=input_ids,
                max_length=MAX_NEW_TOKENS,
                num_beams=5,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                early_stopping=True,
                temperature=float(os.getenv("TEMPERATURE", 0.7)),  # Control randomness
                top_p=float(os.getenv("TOP_P", 0.9)),  # Nucleus sampling for diversity
                length_penalty=float(os.getenv("LENGTH_PENALTY", 1.2))  # Encourage longer outputs
            )
            translated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated_sentences.append(translated_sentence)
        except Exception as e:
            console.log(f"[bold red]Error during translation:[/bold red] {str(e)}")
            translated_sentences.append("")
    return translated_sentences

# Markdown processing function to extract text content from Markdown tokens
def extract_markdown_text(tokens):
    text_blocks = []
    for token in tokens:
        if token.type == "paragraph_open":
            continue
        if token.type == "paragraph_close":
            continue
        if token.type == "text":
            text_blocks.append(token.content)
    return text_blocks

# Markdown processing function to reassemble translated Markdown
def reassemble_markdown(tokens, translated_texts):
    translated_tokens = []
    text_idx = 0

    for token in tokens:
        if token.type == "text":
            if text_idx < len(translated_texts):
                token.content = translated_texts[text_idx]
                text_idx += 1
        translated_tokens.append(token)

    return translated_tokens

# The GUI Application using PyQt5
class TranslatorApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Window setup
        self.setWindowTitle("Markdown Translator")
        self.setGeometry(100, 100, 500, 300)
        self.setAcceptDrops(True)

        # Label for instructions
        self.label = QtWidgets.QLabel("Drag and drop your Markdown (.md) file here", self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        # Progress Bar Setup
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setGeometry(50, 200, 400, 30)
        self.progress_bar.setValue(0)

        # Set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

        # Thread Pool Executor for background tasks
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    # Override dragEnterEvent to handle drag events
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    # Override dropEvent to handle file drop
    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith(".md"):
                self.label.setText("Processing...")
                self.translate_file_async(file_path)

    # Asynchronous translation to keep GUI responsive
    def translate_file_async(self, file_path):
        self.executor.submit(self.translate_file, file_path)

    # Core function for file translation
    def translate_file(self, file_path):
        try:
            # Read Markdown file
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Parse Markdown content
            md = MarkdownIt()
            tokens = md.parse(content)
            text_blocks = extract_markdown_text(tokens)

            # Log the extracted content
            console.log(f"[bold cyan]Extracted Text Blocks:[/bold cyan] {text_blocks}")

            # Translate sentences using MADLAD model
            translated_texts = translate_madlad(text_blocks, model, tokenizer)

            # Reassemble Markdown with translations
            translated_tokens = reassemble_markdown(tokens, translated_texts)
            translated_markdown = md.renderer.render(translated_tokens)

            # Save to new Markdown file
            output_md_path = file_path.replace('.md', '-translated.md')
            with open(output_md_path, "w", encoding="utf-8") as output_file:
                output_file.write(translated_markdown)

            console.log(f"[bold magenta]Translation complete! Results saved to {output_md_path}[/bold magenta]")

            # Update label upon completion using main thread
            QtCore.QMetaObject.invokeMethod(
                self.label,
                "setText",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, f"Translation complete! Saved as {output_md_path}")
            )
        except Exception as e:
            console.log(f"[bold red]Error:[/bold red] {str(e)}")
            QtCore.QMetaObject.invokeMethod(
                self.label,
                "setText",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, "Error during translation. See logs for details.")
            )

    # Function to update progress bar value
    def update_progress(self, value):
        self.progress_bar.setValue(value)

# Main function to start the GUI Application
def main():
    app = QtWidgets.QApplication(sys.argv)
    translator_app = TranslatorApp()
    translator_app.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
