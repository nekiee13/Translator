import sys
import concurrent.futures
from PyQt5 import QtWidgets, QtCore
from markdown_it import MarkdownIt
from rich.console import Console
from transformers import T5Tokenizer, T5ForConditionalGeneration
from markdown_utils import extract_markdown_text, reassemble_markdown, render_markdown
from file_handling import read_markdown_file, write_markdown_file
import torch

console = Console()

class TranslateMadlad:
    def __init__(self, model_name="jbochi/madlad400-3b-mt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

translate_madlad = TranslateMadlad()

class TranslatorApp(QtWidgets.QWidget):
    progress_signal = QtCore.pyqtSignal(int)  # Signal for updating progress bar

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

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.progress_signal.connect(self.update_progress_bar)

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
                self.translate_file_async(file_path)

    def translate_file_async(self, file_path):
        self.executor.submit(self.translate_file, file_path)

    def translate_file(self, file_path):
        try:
            content = read_markdown_file(file_path)
            md = MarkdownIt()
            tokens = md.parse(content)
            text_blocks = extract_markdown_text(tokens)
            total_blocks = len(text_blocks)

            if total_blocks == 0:
                self.update_label("No translatable content found in the file.")
                return

            console.log(f"Extracted Text Blocks: {text_blocks}")

            translated_texts = []
            for idx, block in enumerate(text_blocks):
                try:
                    if not block.strip():
                        console.log(f"[bold red]Skipping empty text block at index {idx}[/bold red]")
                        translated = "[Empty Block]"
                    else:
                        normalized_block = f"<2en> {block.strip()}"
                        console.log(f"[bold blue]Translating block {idx + 1}/{total_blocks}: {normalized_block}[/bold blue]")
                        input_ids = translate_madlad.tokenizer(normalized_block, return_tensors="pt", max_length=256, truncation=True).input_ids.to(translate_madlad.device)
                        console.log(f"Tokenized Input IDs for block {idx + 1}: {input_ids}")
                        outputs = translate_madlad.model.generate(input_ids, max_length=256)
                        console.log(f"Raw Model Output for block {idx + 1}: {outputs}")
                        translated = translate_madlad.tokenizer.decode(outputs[0], skip_special_tokens=True)
                except Exception as translate_error:
                    console.log(f"[bold red]Error translating block {idx + 1}/{total_blocks}: {translate_error}[/bold red]")
                    translated = "[Translation Error]"
                translated_texts.append(translated)

                # Update progress bar via signal
                progress = int((idx + 1) / total_blocks * 100)
                console.log(f"[bold yellow]Updating progress to {progress}%[/bold yellow]")
                self.progress_signal.emit(progress)

            console.log(f"[bold blue]Translated Texts: {translated_texts}[/bold blue]")

            # Debugging tokens after reassembly
            translated_tokens = reassemble_markdown(tokens, translated_texts)
            console.log(f"[bold blue]Translated Tokens after Reassembly: {translated_tokens}[/bold blue]")

            # Render markdown and debug output
            translated_markdown = render_markdown(translated_tokens)

            output_path = file_path.replace(".md", "-translated.md")
            write_markdown_file(output_path, translated_markdown)

            console.log(f"[bold green]Translation complete! Results saved to {output_path}[/bold green]")
            self.update_label(f"Translation complete! Saved as {output_path}")
        except torch.cuda.CudaError as cuda_error:
            console.log(f"[bold red]CUDA Error: {cuda_error}[/bold red]")
            self.update_label("CUDA error during translation. See logs for details.")
        except AttributeError as attr_error:
            console.log(f"[bold red]AttributeError: Ensure 'translate_madlad' is properly initialized: {attr_error}[/bold red]")
            self.update_label("Model initialization error. See logs for details.")
        except Exception as e:
            console.log(f"[bold red]Error: {e}[/bold red]")
            self.update_label("Error during translation. See logs for details.")
        finally:
            self.log_cuda_debug()

    def update_label(self, text):
        QtCore.QMetaObject.invokeMethod(
            self.label,
            "setText",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, text)
        )

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def check_cuda_availability(self):
        try:
            if not torch.cuda.is_available():
                console.log("[bold yellow]CUDA is not available. Falling back to CPU.[/bold yellow]")
                return False
            return True
        except Exception as cuda_check_error:
            console.log(f"[bold red]Error checking CUDA availability: {cuda_check_error}[/bold red]")
            return False

    def log_cuda_debug(self):
        try:
            console.log("[bold blue]Testing CUDA environment...[/bold blue]")
            console.log(f"CUDA Available: {torch.cuda.is_available()}")
            console.log(f"CUDA Device Count: {torch.cuda.device_count()}")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    console.log(f"Device {i}: {torch.cuda.get_device_name(i)}")
        except Exception as cuda_debug_error:
            console.log(f"[bold red]Error during CUDA debug: {cuda_debug_error}[/bold red]")

def main():
    app = QtWidgets.QApplication(sys.argv)
    translator_app = TranslatorApp()
    translator_app.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
