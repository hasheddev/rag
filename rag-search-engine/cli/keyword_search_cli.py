from pathlib import Path

from utils.init import init_keyword_search_parser, keyword_search_cli_input
from index.inverted_index import InvertedIndex
from commands.keyword_search_commands import (
        build_command, search_command, term_frequency_command, inverse_document_frequency_command,
        tf_idf_command, bm25_idf_command, bm25_tf_command, bm25_search_command
    )
base_dir = Path(__file__).parent.parent.absolute()


movies_path = base_dir / "data/movies.json"
stopword_path = base_dir / "data/stopwords.txt"
cache_path = base_dir / "cache"

index = InvertedIndex(cache_path, movies_path, stopword_path)

def main() -> None:
    parser = init_keyword_search_parser()
    args = parser.parse_args()

    match args.command:
        case keyword_search_cli_input.SEARCH:
            print(f"Searching for: {args.query}")
            result = search_command(args.query, index)
            for movie in result:
                print(f"{movie["title"]}: {movie["id"]}")
        case keyword_search_cli_input.BUILD:
                build_command(index)
        case keyword_search_cli_input.TERM_FREQUENCY:
            result = term_frequency_command(args.term, args.doc_id , index)
            print(result)
        case keyword_search_cli_input.INVERSE_DOCUMENT_FREQUENCY:
            idf  = inverse_document_frequency_command(args.term, index)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case keyword_search_cli_input.TFIDF:
            tf_idf = tf_idf_command(args.term, args.doc_id, index)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case keyword_search_cli_input.BM25_IDF:
            bm25_idf = bm25_idf_command(args.term, index)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case keyword_search_cli_input.BM25_TF:
            bm25_tf = bm25_tf_command(args.term, args.doc_id, args.k1, index, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")
        case keyword_search_cli_input.BM25_SEARCH:
            result = bm25_search_command(args.query, index=index, limit=args.limit)
            for movie_score in result:
                print(f"({movie_score[0]['id']}) {movie_score[0]['title']} - Score: {movie_score[1]:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()