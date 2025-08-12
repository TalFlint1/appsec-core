# src/ui/cli.py
from dotenv import load_dotenv; load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from ..rag.pipeline import RAGPipeline

def main():
    console = Console()
    console.print(Panel.fit("DocPilot RAG — Ask about your docs (type /exit to quit)"))
    pipe = RAGPipeline()
    while True:
        q = Prompt.ask("[bold cyan]You")
        if not q: continue
        if q.strip().lower() in {"/exit","/quit"}: break
        try:
            ans = pipe.answer(q)
            console.print(Panel(ans, title="Answer", expand=False))
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")

if __name__ == "__main__":
    main()
