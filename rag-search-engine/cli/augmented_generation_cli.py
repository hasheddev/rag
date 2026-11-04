from utils.rag_parser import init_rag_search_parser, RagCommand
from commands.augmented_generation_commands import (
    rag_search,
    summarize_rag_search,
    rag_search_with_citation,
    answer_question,
)
from utils.types import HybridRFFMovieSearch


def print_results(results: list[HybridRFFMovieSearch]):
    print("Search Results:")
    for movie in results:
        print(f"- {movie["title"]}")
    print("\n\n")


def main() -> None:
    parser = init_rag_search_parser()
    args = parser.parse_args()

    match args.command:
        case RagCommand.RAG:
            query = args.query
            response, movies = rag_search(query)
            print_results(movies)
            print("RAG Response:")
            print(response)
        case RagCommand.SUMMARIZE:
            response, movies = summarize_rag_search(args.query, args.limit)
            print_results(movies)
            print("LLM Summary:")
            print(response)
        case RagCommand.CITATIONS:
            response, movies = rag_search_with_citation(args.query, args.limit)
            print_results(movies)
            print("LLM Answer:")
            print(response)
        case RagCommand.QUESTION:
            response, movies = answer_question(args.question, args.limit)
            print_results(movies)
            print("Answer:")
            print(response)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
