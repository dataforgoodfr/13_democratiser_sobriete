from taxonomy import paper_taxonomy


# get the installed ollama models
#OLLAMA_MODELS = [m["model"] for m in ollama.list().get("models", [])]

# get the available taxonomies
TAXS = {n: o
    for n, o in {
        name: obj for name, obj in paper_taxonomy.__dict__.items() if callable(obj)
    }.items()
    if o.__module__ == "taxonomy.paper_taxonomy"
}
