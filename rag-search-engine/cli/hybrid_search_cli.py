from commands.hybrid_search_commands import normalize, weighted_search, rrf_search
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
            result = rrf_search(args.query, args.limit, args.k, args.enhance, args.rerank_method)
            for i, movie in enumerate(result, 1):
                rerank_str = ""
                if args.rerank_method == "individual":
                    rerank_str = f"Re-rank Score: {movie['llm_score']}/10"
                elif args.rerank_method == "batch":
                    rerank_str = f'Re-rank Rank: {movie["rerank_rank"]}'
                elif args.rerank_method == "cross_encoder":
                    rerank_str = f"Cross Encoder Score: {movie["encoder_rank"]:.3f}"
                print(f"{i}. {movie['title']}")
                if rerank_str != "":
                    print(rerank_str)
                print(f"\tRRF Score:{movie['rrf_score']}")
                print(f"\tBM25 Rank: {movie['keyword_rank']}, Semantic Rank: {movie['semantic_rank']}")
                print(f"\t{movie['description']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()