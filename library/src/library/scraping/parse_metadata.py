import os
import pickle as pkl
from copy import deepcopy

from library.infra import PDF_STORAGE_FOLDER

output_dir = PDF_STORAGE_FOLDER

def get_all_results() -> list:
    all_results = os.listdir(os.path.join(output_dir, 'pkl_files'))
    return all_results

def remove_unnecessary_fields(result:dict) -> dict:
    keys_to_remove = ["display_name", 
                      "relevance_score", 
                      "publication_year", 
                      "ids", 
                      "primary_location", 
                      "institution_assertions",
                      "countries_distinct_count",
                      "institutions_distinct_count",
                      "corresponding_author_ids",
                      "corresponding_institution_ids",
                      "apc_list",
                      "apc_paid",
                      "fwci",
                      "has_fulltext",
                      "fulltext_origin",
                      "biblio",
                      "primary_topic",
                      "grants",
                      "mesh",
                      "locations_count",
                      "locations",
                      "datasets",
                      "versions",
                      "referenced_works_count",
                      "referenced_works",
                      "related_works",
                      "cited_by_api_url",
                      "cited_by_percentile_year",
                      "counts_by_year",
                      "abstract_inverted_index_v3",
                      ]
    
    clean_result = deepcopy(result)
    for key in keys_to_remove :
        if key in clean_result :
            del(clean_result[key])
    return clean_result

def clean_result_fields(raw_result:dict) -> dict:

    def get_abstract(result:dict) -> str:
        inverted_abstract = result["abstract_inverted_index"]
        if not inverted_abstract :
            return ""
        else :
            keys, values = list(inverted_abstract), list(inverted_abstract.values())
            sequence = ["" for i in range(values[-1][-1] + 1)]
            for i in range(len(keys)) :
                key = keys[i]
                key_values = values[i]
                for j in range(len(key_values)) :
                    value = key_values[j]
                    while value >= len(sequence) :
                        sequence.append("")
                    sequence[value] = key
        return " ".join(sequence)

    def get_topics(result:dict) -> str:
        all_topics = result["topics"]
        parsed_topics = []
        for topic in all_topics :
            display_name = topic["display_name"]
            subfield = topic["subfield"]["display_name"]
            parsed_topics.append(f"{display_name}|{subfield}")
        parsed_topics = '__'.join(parsed_topics)
        return parsed_topics

    def get_oa_url(result:dict) -> str:
        if result["best_oa_location"] :
            if result["best_oa_location"]["pdf_url"] :
                openaccess_url = result["best_oa_location"]["pdf_url"]
                return openaccess_url    
        openaccess_url = result["open_access"]["oa_url"]
        return openaccess_url
    
    trimmed_result = remove_unnecessary_fields(raw_result)

    abstract = get_abstract(trimmed_result)
    topics = get_topics(trimmed_result)
    concepts = "|".join([f"{concept['display_name']}:{concept['score']}" for concept in trimmed_result["concepts"]])
    indexed_in = "|".join(trimmed_result["indexed_in"])
    keywords = "|".join([f"{keyword['display_name']}:{keyword['score']}" for keyword in trimmed_result["keywords"]])
    openaccess_url = get_oa_url(trimmed_result)
    citation_normalized_percentile = trimmed_result["citation_normalized_percentile"]["value"] if trimmed_result["citation_normalized_percentile"] else 0
    sustainable_development_goals = "|".join([f"{sdg['display_name']}:{sdg['score']}" for sdg in trimmed_result["sustainable_development_goals"]])
    author_ids = "|".join([f"{authorship['author']['id'].split('/')[-1]}" for authorship in trimmed_result["authorships"]])
    institution_ids = "|".join([f"{institution['id'].split('/')[-1]}" for authorship in trimmed_result["authorships"] for institution in authorship["institutions"]])
    openalex_id = trimmed_result["id"].split('/')[-1]
    successfully_downloaded = True if os.path.isfile(os.path.join(output_dir, "pdf_files", f"{openalex_id}.pdf")) else False 

    keys_to_remove = ["id", "abstract_inverted_index", "topics", "concepts", "keywords", "authorships",
                      "citation_normalized_percentile", "open_access", "best_oa_location", "sustainable_development_goals"]
    
    for key in keys_to_remove :
        del(trimmed_result[key])

    trimmed_result["openalex_id"] = openalex_id
    trimmed_result["updated_date"] = trimmed_result["updated_date"].split("T")[0]
    trimmed_result["abstract"] = abstract
    trimmed_result["topics"] = topics
    trimmed_result["concepts"] = concepts
    trimmed_result["indexed_in"] = indexed_in
    trimmed_result["keywords"] = keywords
    trimmed_result["citation_normalized_percentile"] = citation_normalized_percentile
    trimmed_result["openaccess_url"] = openaccess_url
    trimmed_result["sustainable_development_goals"] = sustainable_development_goals
    trimmed_result["author_ids"] = author_ids
    trimmed_result["institution_ids"] = institution_ids
    trimmed_result["successfully_downloaded"] = successfully_downloaded
    return trimmed_result

def main() -> list:
    all_results = get_all_results()
    all_clean_results = []
    for result in all_results :
        raw_result = pkl.load(open(os.path.join(output_dir, 'pkl_files', result), "rb"))
        clean_result = clean_result_fields(raw_result)
        all_clean_results.append(clean_result)
    return all_clean_results


if __name__ == "__main__":
    main()
