from utils.parser import init_semantic_search_parser, SemanticSearchCommand
from commands.semantic_search_commands import (
        verify_model, embed_text, verify_embeddings, embed_query_text, search,
        chunk, semantic_chunk, embed_chunks, search_chunked
    )

def main() -> None:
    parser = init_semantic_search_parser()
    args = parser.parse_args()

    match args.command:
        case SemanticSearchCommand.VERIFY:
            verify_model()
        case SemanticSearchCommand.EMBED_TEXT:
            embed_text(args.term)
        case SemanticSearchCommand.VERIFY_EMBEDDINGS:
            verify_embeddings()
        case SemanticSearchCommand.EMBED_QUERY:
            embed_query_text(args.term)
        case SemanticSearchCommand.SEARCH:
            search(args.query, limit=args.limit)
        case SemanticSearchCommand.CHUNK:
            chunks = chunk(text=args.text, chunk_size=args.chunk_size, overlap=args.overlap)
            print(f"Chunking {len(args.text)} characters") 
            for index, chunk_text in enumerate(chunks, start=1):
                print(f"{index}. {chunk_text}")
        case SemanticSearchCommand.SEMANTIC_CHUNK:
            chunks = semantic_chunk(text=args.text, max_chunk_size=args.max_chunk_size, overlap=args.overlap)
            print(f"Semantically chunking {len(args.text)} characters") 
            for index, chunk_text in enumerate(chunks, start=1):
                print(f"{index}. {chunk_text}")
        case SemanticSearchCommand.EMBED_CHUNKS:
            embed_chunks()
        case SemanticSearchCommand.SEARCH_CHUNKED:
            search_chunked(args.query, limit=args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()