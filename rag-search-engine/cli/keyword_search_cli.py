from pathlib import Path

from utils.parser import init_keyword_search_parser, KeywordSearchCommand
from commands.keyword_search_commands import (
        build_command, search_command, term_frequency_command,
        inverse_document_frequency_command, tf_idf_command,
        bm25_idf_command, bm25_tf_command, bm25_search_command
    )

def main() -> None:
    parser = init_keyword_search_parser()
    args = parser.parse_args()

    match args.command:
        case KeywordSearchCommand.SEARCH:
            print(f"Searching for: {args.query}")
            result = search_command(args.query)
            for movie in result:
                print(f"{movie["title"]}: {movie["id"]}")
        case KeywordSearchCommand.BUILD:
                build_command()
        case KeywordSearchCommand.TERM_FREQUENCY:
            result = term_frequency_command(args.term, args.doc_id)
            print(result)
        case KeywordSearchCommand.INVERSE_DOCUMENT_FREQUENCY:
            idf  = inverse_document_frequency_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case KeywordSearchCommand.TFIDF:
            tf_idf = tf_idf_command(args.term, args.doc_id, )
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case KeywordSearchCommand.BM25_IDF:
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case KeywordSearchCommand.BM25_TF:
            bm25_tf = bm25_tf_command(args.term, args.doc_id, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")
        case KeywordSearchCommand.BM25_SEARCH:
            result = bm25_search_command(args.query, limit=args.limit)
            for movie_score in result:
                print(f"({movie_score['id']}) {movie_score['title']} - Score: {movie_score["score"]:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()