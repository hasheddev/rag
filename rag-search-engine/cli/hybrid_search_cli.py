from commands.hybrid_search_commands import normalize, weighted_search, rrf_search
from utils.search_parser import init_hybrid_search_parser, HybridSearchCommand
from utils.query_utils import print_rrf_results


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
                print(
                    f"\tBM25: {movie['keyword_score']}, Semantic: {movie['semantic_score']}"
                )
                print(f"\t{movie['description']}")
        case HybridSearchCommand.RRF_SEARCH:
            result = rrf_search(
                args.query,
                args.limit,
                args.k,
                args.enhance,
                args.rerank_method,
                args.evaluate,
            )
            if args.rerank_method:
                print("\n*************")
                print("Results After Reranking")
                print_rrf_results(result, args.rerank_method)
                print("*************\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
