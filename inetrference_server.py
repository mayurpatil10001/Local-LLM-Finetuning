#!/usr/bin/env python3
"""
Inference Server for Trained LLM
Simple request-response system for your trained model
"""

import os
import argparse
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

class LLMInference:
    """Simple inference wrapper for trained LLM"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """Initialize the inference engine"""
        
        console.print(f"[yellow]Loading model from {model_path}...[/yellow]")
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )
        
        # Create pipeline for easy inference
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1
        )
        
        console.print(f"[green]✓ Model loaded on {device}[/green]")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> str:
        """Generate response for given prompt"""
        
        # Generate
        outputs = self.generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            truncation=True
        )
        
        # Extract generated text
        generated_text = outputs[0]['generated_text']
        
        # Remove the prompt from output
        response = generated_text[len(prompt):].strip()
        
        return response
    
    def chat(self, question: str) -> str:
        """Simple Q&A interface"""
        
        # Format as Q&A
        prompt = f"Question: {question}\nAnswer:"
        
        response = self.generate(
            prompt,
            max_length=150,
            temperature=0.7,
            top_p=0.9
        )
        
        return response

def interactive_mode(model_path: str):
    """Run interactive chat mode"""
    
    console.print(Panel.fit(
        "[bold cyan]LLM Interactive Mode[/bold cyan]\n"
        "Type your questions and get answers!\n"
        "Commands: 'quit' or 'exit' to stop",
        style="cyan"
    ))
    
    # Initialize model
    llm = LLMInference(model_path)
    
    console.print("\n[green]Ready! Ask me anything:[/green]\n")
    
    while True:
        try:
            # Get user input
            question = console.input("[bold blue]You:[/bold blue] ")
            
            if question.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not question.strip():
                continue
            
            # Generate response
            console.print("[yellow]Thinking...[/yellow]")
            response = llm.chat(question)
            
            # Display response
            console.print(f"[bold green]AI:[/bold green] {response}\n")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

def batch_mode(model_path: str, questions_file: str, output_file: str):
    """Process questions from file"""
    
    console.print(f"[yellow]Processing questions from {questions_file}...[/yellow]")
    
    # Initialize model
    llm = LLMInference(model_path)
    
    # Read questions
    with open(questions_file, 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    # Process each question
    results = []
    for i, question in enumerate(questions, 1):
        console.print(f"[cyan]Processing {i}/{len(questions)}: {question}[/cyan]")
        response = llm.chat(question)
        results.append({
            "question": question,
            "answer": response
        })
    
    # Save results
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"[green]✓ Results saved to {output_file}[/green]")

def single_query_mode(model_path: str, query: str):
    """Process single query"""
    
    llm = LLMInference(model_path)
    
    console.print(f"\n[bold blue]Question:[/bold blue] {query}\n")
    response = llm.chat(query)
    console.print(f"[bold green]Answer:[/bold green] {response}\n")

def main():
    parser = argparse.ArgumentParser(description="Inference server for trained LLM")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--mode", type=str, default="interactive",
                       choices=["interactive", "batch", "single"],
                       help="Inference mode")
    parser.add_argument("--query", type=str, default=None,
                       help="Single query (for single mode)")
    parser.add_argument("--questions-file", type=str, default=None,
                       help="Input file with questions (for batch mode)")
    parser.add_argument("--output-file", type=str, default="answers.json",
                       help="Output file for answers (for batch mode)")
    
    args = parser.parse_args()
    
    # Verify model path exists
    if not os.path.exists(args.model_path):
        console.print(f"[red]Error: Model path not found: {args.model_path}[/red]")
        return
    
    # Run appropriate mode
    if args.mode == "interactive":
        interactive_mode(args.model_path)
    elif args.mode == "batch":
        if not args.questions_file:
            console.print("[red]Error: --questions-file required for batch mode[/red]")
            return
        batch_mode(args.model_path, args.questions_file, args.output_file)
    elif args.mode == "single":
        if not args.query:
            console.print("[red]Error: --query required for single mode[/red]")
            return
        single_query_mode(args.model_path, args.query)

if __name__ == "__main__":
    main()