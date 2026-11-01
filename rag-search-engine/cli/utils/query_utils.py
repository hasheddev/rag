import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import time
import json
from sentence_transformers import CrossEncoder

from utils.types import HybridRFFMovieSearch

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY")
MODEL = "openrouter/free"

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def enhance_query_spelling(query: str):
    user_message = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": user_message
        }
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    enhanced_query =  response.choices[0].message.content
    print(f"Enhanced query (spell): '{query}' -> '{enhanced_query}'\n")
    return enhanced_query

def rewrite_user_query(query: str):
    user_message = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": user_message
        }
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    enhanced_query =  response.choices[0].message.content
    print(f"Enhanced query (rewrite): '{query}' -> '{enhanced_query}'\n")
    return enhanced_query

def expand_user_query(query: str):
    user_message = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": user_message
        }
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    enhanced_query =  response.choices[0].message.content
    print(f"Enhanced query (expand): '{query}' -> '{enhanced_query}'\n")
    return enhanced_query

def get_individual_message(query: str, movie: HybridRFFMovieSearch):
    message = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {movie.get("title", "")} - {movie.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": message
        }
    ]
    return messages

def rerank_individual(query: str, matches: list[HybridRFFMovieSearch]):
    for movie in matches:
        messages = get_individual_message(query, movie)
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        movie["llm_score"] = int(response.choices[0].message.content or "0")
        time.sleep(2)
    return matches

def construct_movie_details(matches: list[HybridRFFMovieSearch]):
    result = ""
    for movie in matches:
        result += f'Movie ID - ({movie["id"]}): {movie["title"]} - {movie["description"]}\n'
    return result

def rerank_batches(query: str, matches: list[HybridRFFMovieSearch]):
    doc_list_str = construct_movie_details(matches)
    message = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": message
        }
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    ids_by_rank: list[int] = json.loads(response.choices[0].message.content or "")
    movie_by_id = { movie["id"]: movie for movie in matches}
    result: list[HybridRFFMovieSearch] = []
    for i, movie_id in enumerate(ids_by_rank, 1):
        movie = movie_by_id.get(movie_id)
        if movie:
            movie["rerank_rank"] = i
            result.append(movie)
    return result


def rerank_with_cross_encoder(query: str, matches: list[HybridRFFMovieSearch]):
    pairs = [[query, f"{movie["title"]} - {movie["description"]}"] for movie in matches]
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)
    for i, match in enumerate(matches):
        match["encoder_rank"] = float(scores[i])
    return sorted(matches, key=lambda match: match["encoder_rank"], reverse=True)