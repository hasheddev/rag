from commands.hybrid_search_commands import normalize, weighted_search, rff_search
from utils.parser import init_hybrid_search_parser, HybridSearchCommand

def main() -> None:
    parser = init_hybrid_search_parser()

    args = parser.parse_args()

    match args.command:
        case HybridSearchCommand.NORMALIZE:
            scores = normalize(args.scores)
            for score in scores:
                print(f"* {score:.4f}")
        case HybridSearchCommand.WEIGHTED_SEARCH:
            results = weighted_search(args.query, args.limit, args.alpha)
            for i, movie in enumerate(results, 1):
                print(f"{i}. {movie['title']}")
                print(f"\tHybrid Score:{movie['hybrid_score']}")
                print(f"\tBM25: {movie['keyword_score']}, Semantic: {movie['semantic_score']}")
                print(f"\t{movie['description']}")
        case HybridSearchCommand.RRF_SEARCH:
            result = rff_search(args.query, args.limit, args.k)
            for i, movie in enumerate(result, 1):
                print(f"{i}. {movie['title']}")
                print(f"\tRFF Score:{movie['rrf_score']}")
                print(f"\tBM25 Rank: {movie['keyword_rank']}, Semantic Rank: {movie['semantic_rank']}")
                print(f"\t{movie['description']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()