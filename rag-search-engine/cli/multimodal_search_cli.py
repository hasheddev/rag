from pathlib import Path

from utils.rag_parser import init_multimodal_search_parser, MultiModalCommand
from commands.multimodal_commands import verify_embeddings_command, image_search_command


def main() -> None:
    parser = init_multimodal_search_parser()
    args = parser.parse_args()

    match args.command:
        case MultiModalCommand.VERIFY_EMBEDDING:
            image_path = args.image_path
            verify_embeddings_command(Path(image_path))
        case MultiModalCommand.IMAGE_SEARCH:
            image_path = args.image_path
            result = image_search_command(Path(image_path))
            for i, movie in enumerate(result, 1):
                print(f"{i}. {movie['title']} (similarity: {movie["score"]:.4f})")
                print(f"  {movie['description']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
