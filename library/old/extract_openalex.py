"""
Old utils to query Open Alex API.
Prefer using OpenAlexConnector.
"""


import requests


# Request OpenAlex API
def search_openalex(
    query: str, cursor="*", per_page: int = 50, from_dois: bool = False, dois: list = None
) -> dict:
    if dois is None:
        dois = []

    if from_dois:
        pipe_separated_dois = "|".join(dois)
        params = {
            "filter": f"open_access.is_oa:true,doi:{pipe_separated_dois}",
            "cursor": cursor,
            "per-page": per_page,
        }
    else:
        params = {
            "filter": "open_access.is_oa:true",
            "search": f"{query}",
            "cursor": cursor,
            "per-page": per_page,
        }

    url = "https://api.openalex.org/works"
    response = requests.get(url, params=params)
    response.raise_for_status()
    query_data = response.json()
    return query_data


# Retrieve PDF urls and OpenAlex IDs
def get_urls_to_fetch(query_data: dict):
    urls_to_fetch = []
    filenames = []
    for i in range(len(query_data["results"])):
        file_title = query_data["results"][i]["id"]
        filenames.append(file_title.split("/")[-1])
        try:
            urls_to_fetch.append(query_data["results"][i]["best_oa_location"]["pdf_url"])
        except TypeError:
            urls_to_fetch.append(query_data["results"][i]["open_access"]["oa_url"])
    return urls_to_fetch, filenames
