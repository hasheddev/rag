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
        {"role": "user", "content": user_message}
    ]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    enhanced_query = response.choices[0].message.content
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
        {"role": "user", "content": user_message}
    ]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    enhanced_query = response.choices[0].message.content
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
        {"role": "user", "content": user_message}
    ]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    enhanced_query = response.choices[0].message.content
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
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": message}]
    return messages


def rerank_individual(query: str, matches: list[HybridRFFMovieSearch]):
    for movie in matches:
        messages = get_individual_message(query, movie)
        response = client.chat.completions.create(model=MODEL, messages=messages)
        movie["llm_score"] = int(response.choices[0].message.content or "0")
        time.sleep(2)
    return matches


def construct_movie_details(matches: list[HybridRFFMovieSearch]):
    result = ""
    for movie in matches:
        result += (
            f'Movie ID - ({movie["id"]}): {movie["title"]} - {movie["description"]}\n'
        )
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
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": message}]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    ids_by_rank: list[int] = json.loads(response.choices[0].message.content or "")
    movie_by_id = {movie["id"]: movie for movie in matches}
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


def print_rrf_results(matches: list[HybridRFFMovieSearch], rerank_method: str | None):
    for i, movie in enumerate(matches, 1):
        rerank_str = ""
        if rerank_method == "individual":
            rerank_str = f"\tRe-rank Score: {movie['llm_score']}/10"
        elif rerank_method == "batch":
            rerank_str = f'\tRe-rank Rank: {movie["rerank_rank"]}'
        elif rerank_method == "cross_encoder":
            rerank_str = f"\tCross Encoder Score: {movie["encoder_rank"]:.3f}"
        print(f"{i}. {movie['title']}")
        if rerank_str != "":
            print(rerank_str)
        print(f"\tRRF Score:{movie['rrf_score']}")
        print(
            f"\tBM25 Rank: {movie['keyword_rank']}, Semantic Rank: {movie['semantic_rank']}"
        )
        print(f"\t{movie['description']}")


def evaluate_movie_results(query: str, results: list[HybridRFFMovieSearch]):
    """
    Evaluates search results using an LLM judge on a 0-3 relevance scale
    and prints a structured evaluation report.
    """
    formatted_results = [
        f"- {movie.get('title', 'Unknown Title')}: {movie.get('description', '')}"
        for movie in results
    ]

    user_message = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:
[2, 0, 3, 2, 0, 1]"""

    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": user_message}
    ]

    response = client.chat.completions.create(model=MODEL, messages=messages)
    content = response.choices[0].message.content
    content = content.strip() if content else ""

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:].strip()
        content = content.strip()

    try:
        scores = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {content}")
        raise e

    # Print evaluation report in the requested exact format
    print(f"\nEvaluation Report for Query: '{query}'")
    for i, (movie, score) in enumerate(zip(results, scores), 1):
        title = movie.get("title", "Unknown Title")
        print(f"{i}. {title}: {score}/3")


def generate_response(query: str, results: list[HybridRFFMovieSearch]):
    """
    Evaluates search results using an LLM judge on a 0-3 relevance scale
    and prints a structured evaluation report.
    """
    formatted_results = [
        f"- {movie.get('title', 'Unknown Title')}: {movie.get('description', '')}"
        for movie in results
    ]
    docs = "\n".join(formatted_results)

    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(model=MODEL, messages=messages)
    content = response.choices[0].message.content
    return content


def generate_summary(query: str, results: list[HybridRFFMovieSearch]):
    formatted_results = [
        f"- {movie.get('title', 'Unknown Title')}: {movie.get('description', '')}"
        for movie in results
    ]
    docs = "\n".join(formatted_results)
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{docs}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(model=MODEL, messages=messages)
    content = response.choices[0].message.content
    return content


def generate_response_with_citations(query: str, results: list[HybridRFFMovieSearch]):
    formatted_results = [
        f"- {movie.get('title', 'Unknown Title')}: {movie.get('description', '')}"
        for movie in results
    ]
    docs = "\n".join(formatted_results)
    prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(model=MODEL, messages=messages)
    content = response.choices[0].message.content
    return content


def generate_direct_response(query: str, results: list[HybridRFFMovieSearch]):
    formatted_results = [
        f"- {movie.get('title', 'Unknown Title')}: {movie.get('description', '')}"
        for movie in results
    ]
    context = "\n".join(formatted_results)

    prompt = f"""Answer the following question based on the provided documents.

Question: {query}

Documents:
{context}

General instructions:
- Answer directly and concisely
- Use only information from the documents
- If the answer isn't in the documents, say "I don't have enough information"
- Cite sources when possible

Guidance on types of questions:
- Factual questions: Provide a direct answer
- Analytical questions: Compare and contrast information from the documents
- Opinion-based questions: Acknowledge subjectivity and provide a balanced view

Answer:"""
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(model=MODEL, messages=messages)
    content = response.choices[0].message.content
    return content
