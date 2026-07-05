import argparse
import asyncio
import os
import sys
from google import genai
from dotenv import load_dotenv

from .client import generate_ensemble
from .reducer import reduce_critic, reduce_voting


async def run_cli(args):
    """
    Main execution logic for CLI. Reads the prompt from file,
    resolves parameters, runs the ensemble, and prints the result.
    """
    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        print(
            "Error: GEMINI_API_KEY is not set in environment or .env file.",
            file=sys.stderr
        )
        sys.exit(1)

    # Read prompt file
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            prompt = f.read()
    except Exception as e:
        print(f"Error reading file '{args.file}': {e}", file=sys.stderr)
        sys.exit(1)

    # Resolve strategy function
    if args.reducer == "critic":
        strategy = reduce_critic
    elif args.reducer == "voting":
        strategy = reduce_voting
    else:
        print(f"Error: Unknown reducer '{args.reducer}'. Use 'critic' or 'voting'.", file=sys.stderr)
        sys.exit(1)

    # Run the ensemble
    try:
        client = genai.Client()
        response = await generate_ensemble(
            client=client,
            prompt=prompt,
            model=args.model,
            n=args.n,
            strategy=strategy,
            reducer_model=args.reducer_model,
            output_language=args.language
        )
        print(response.text)
    except Exception as e:
        print(f"Error executing ensemble: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """
    Entrypoint for the gemini-ensemble command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Run Gemini ensemble queries from the command line using a text file input."
    )
    parser.add_argument(
        "file",
        help="Path to the text file containing the prompt."
    )
    parser.add_argument(
        "-n",
        type=int,
        default=3,
        help="Number of parallel model instances (default: 3)."
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gemini-3.1-flash-lite",
        help="Base model to run in parallel (default: gemini-3.1-flash-lite)."
    )
    parser.add_argument(
        "-r",
        "--reducer",
        choices=["critic", "voting"],
        default="critic",
        help="Reduction strategy to merge responses (default: critic)."
    )
    parser.add_argument(
        "-l",
        "--language",
        help="Target language for the final output (e.g. 'Japanese', 'English')."
    )
    parser.add_argument(
        "--reducer-model",
        help="Specific model to use for reduction (defaults to base model)."
    )

    args = parser.parse_args()
    asyncio.run(run_cli(args))


if __name__ == "__main__":
    main()
