# Chat Sufficiency \- Technical Report

# Overview

This technical report describes the three main components of our work: building the sufficiency library, extracting and analyzing policies, and making the results available through a conversational interface called ChatSufficiency. Code can be found on our [Github repository](https://github.com/dataforgoodfr/13_democratiser_sobriete), with three main folders corresponding to these three work packages:

* folder library contains the code for building the sufficiency library;  
* folder policy\_analysis contains the code for policy analysis;  
* folder rag\_system contains the code for Chat Sufficiency.

The following architecture schema maps the data flows between these components.

![][image1]

# Library

## Pre-screening

### Objective

Pre-screening aims to identify in the literature all articles related to the dimensions of sufficiency: the upper limit (planetary boundaries) or the lower limit (well-being for all). Initially, we are using OpenAlex (270 million references) as our sole source. The integration of other sources, in particular gray literature, is planned for the future.

### Methods

Pre-screening is carried out in two stages:

1. Keyword-based filtering by querying the OpenAlex API \-\> 24 million publications.  
2. Semantic filtering using natural language processing (NLP) on abstracts \-\> 2.5 million publications.

#### Keyword-based filtering

[Code](https://github.com/dataforgoodfr/13_democratiser_sobriete/tree/main/library/prescreening/stage1)

2,671 keywords are defined by experts in [this file](https://github.com/dataforgoodfr/13_democratiser_sobriete/blob/main/library/prescreening/stage1/sufficiency_keywords.csv). In order to limit the number of requests to the OpenAlex API, they are grouped into 149 themes with an OR operator. Grouping by themes transfers a large part of the deduplication to OpenAlex, but the themes remain broad and overlap considerably, leaving many duplicates. We must retrieve almost 100M references to obtain 24M unique references after local deduplication using their OpenAlex identifier. Future work could group themes more aggressively to transfer more deduplication to OpenAlex and shorten this process, which takes several days due to the rate limiting of the OpenAlex API.

Since the amount of data retrieved affects query speed, this step has been divided into two sub-steps:

1. Retrieval of IDs only and deduplication.  
2. Retrieval of metadata of interest for the collected unique IDs.

We use the [pyalex](https://github.com/J535D165/pyalex) library via the OpenAlexConnector defined in the local [library](https://github.com/dataforgoodfr/13_democratiser_sobriete/tree/main/library/src/library) package. The main challenge of this step is that it takes several days to run and may fail, so we designed our scripts to be resumable with as little data loss as possible. We do this by saving progress to a local sqlite database.

#### Semantic filtering

[Code](https://github.com/dataforgoodfr/13_democratiser_sobriete/tree/main/library/prescreening/stage2)

Due to the high number of keywords and the systematic use of the OR boolean operator in the queries, keyword-based filtering has high recall but very low precision. We aim to filter out works irrelevant to the topic of sufficiency using a second stage of semantic filtering, using NLP methods on abstracts.

Our approach for semantic filtering is to detect whether the publication is relevant to one of the four pillars on which sufficiency is built: planetary boundaries, natural resources, wellbeing, and justice. We keep publications relevant to at least one of these dimensions.

More specifically, we fine-tuned a [Sentence Transformer](https://huggingface.co/papers/1908.10084) model (BAAI/bge-small-en-v1.5) with [SetFit](https://arxiv.org/pdf/2209.11055) on a dataset of 165 labeled examples, of which 55 were labelled manually and 110 were labelled automatically with Claude Sonnet 3.7. The labels correspond to each of the four dimensions mentioned above, plus a fifth class labeled “other”. The input data is the concatenation of the title and abstract. We framed the problem as multiclass classification, using the setting multi\_target\_strategy="multi-output" of the [setfit library](https://github.com/huggingface/setfit?tab=readme-ov-file). The model outputs five probabilities, one for each dimension of sufficiency plus one for the class “other”.  

We trained on 80% of the dataset and evaluated on the remaining 20%. The fine-tuned model can be found [here](https://huggingface.co/TheoLvs/wsl-prescreening-multi-v0.0). We obtain an accuracy of 85% on individual classes, but relabeling the four dimensions of sufficiency as positive and “other” as negative the accuracy reaches 100%. As an additional evaluation, we labeled a sample of 793 abstracts among the 24M works returned by keyword filtering using Gemini 3 Flash and compared with predictions from our model. This independent evaluation shows an accuracy of only 51%, mainly due to a benign confusion between planetary boundaries and natural resources. Once labels are binarized as “other vs sufficiency dimension”, we obtain an accuracy of 70% with equal precision and recall.

We keep only the publications for which the highest probability is among the 4 dimensions of sufficiency and above 0.8. This results in about 15% of the lines being kept. Since not all lines have an abstract and we limit ourselves to publications with an abstract in English, approximately 10% of the lines are ultimately kept, for a final library of 2.5M works.

### Results

The \~2.5M works obtained after pre-screening and before full-text extraction are stored in [this parquet file](https://huggingface.co/datasets/sufficiencylab/sufficiency-library/blob/main/library_v1_2025-12-08.parquet) in our public HuggingFace dataset.

### Future work

Semantic filtering could be improved by using a larger model, bge-small-en-v1.5 being only 33M parameters. A multilingual model would also allow us to process a larger share of the literature. Finally, we will work in the coming months on setting up regular and automatic updates of the library.

## Conclusions extraction and pre-processing

### Objective

This part of the project aims at transforming the metadata obtained from the prescreening phase into a usable text dataset ready for ingestion into a vector DB and for policy analysis. In particular, we are interested in the conclusion section of each work, where findings are most likely present in a condensed form.

### Methods

[Code](https://github.com/dataforgoodfr/13_democratiser_sobriete/tree/main/library/scraping)

#### Obtaining usable PDF files

The prescreening stage yielded \~2.5M publications from OpenAlex, with \~1.7M of them open access. Among those, \~1.3M have a direct PDF URL, while the rest only have web URLs, which some of the time contain the full text and most of the time links to a landing page with a button to download the PDF. Out of simplicity, we chose to focus on publications with a direct PDF URL. Some of those link to actual PDF files, while others link to web pages that a PDF with javascript or anti-scraping measures. For the V1 of the library, we focus on the most accessible format: the direct PDF URL. Other formats are left to future improvements as working through each of them to find a reliable method to obtain either the original PDF or the full text of the article requires a tremendous amount of effort. Focusing on real PDF URLs, we obtain \~690k PDF files (taking up about 2 TB).

We divide those into 6 batches of 100k files and 1 batch of 90k files. This batch structure is preserved for the following steps.

#### Extracting text from PDF files

The goal of PDF text extraction is to convert each PDF into an equivalent text file, usually in markdown format. Popular libraries like docling handle it very well, but they are too expensive to run for our scale and means : quick tests showed, after extrapolation, that converting the 690k PDF files into markdown would require \~8500 H100-hours \- almost a full year and over 20 k€.

A much faster library with good results on CPU is [pymupdf4llm](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/). However, we encountered major hurdles in the form of memory issues, with out-of-memory errors killing our jobs quickly even on high-memory machines. We could not overcome them despite several days of debugging.

Thus, we switched to a worse-quality but simple and very fast method: extracting raw text from PDFs without layout analysis. We obtain text that corresponds overall to the text of the PDF, but sometimes with noise text in the middle : headers and footers, page numbers, table data and caption, figure caption. Moreover, table structure isn’t preserved, meaning data can’t be reliably extracted. Since this data isn’t necessary for our downstream analysis, we simply filter it out along with the rest. We use regexes and heuristics for denoising raw text extracted with this method, yielding a result of acceptable quality at a very low cost.

#### Extracting sections

Our aim being to extract elements of produced knowledge from the studied articles, it was decided to only keep the results and conclusion/discussion sections of each article. It helped  reduce the scale of our work and decrease processing costs and duration. It also increased the signal-to-noise ratio for policy analysis.

The markdown format makes it easy using hashtags (although PDF-to-markdown conversion tools don't preserve title hierarchy well), but not the raw text format we eventually worked with, which doesn’t preserve that information well. We used a heuristic based on the fact that section titles in scientific articles usually follow a standardized pattern: Introduction \- Methods \- Results \- Conclusion \- Discussion \- References, with slight variations in the choice of words and structure, and that those words usually appear alone on their line \- an information that is preserved by raw text extraction.

### Results

The above processing stages produces the following results :

* OpenAlex contains 240M works  
* Keyword-based filtering yields 25M works  
* Semantic filtering yields 2.5M works  
* Text extraction yields 690k full texts  
* We successfully extracted conclusions for 557k of them

The main results files can be found in our [public dataset on HuggingFace](https://huggingface.co/datasets/sufficiencylab/sufficiency-library/tree/main).

### Future work

This step can be improved in two main ways: getting access to more texts or PDFs through more advanced web scraping or crawling, and converting PDFs to clean markdown instead of noisy raw text. Potentially, these efforts could be made much easier and cheaper by leveraging [FinePDF](https://huggingface.co/datasets/HuggingFaceFW/finepdfs), a dataset containing 475M PDF documents whose text has already been extracted. We can assume that a large part, if not all, of the 1.3M \- 1.7M open-access publications in the sufficiency library are part of FinePDF.

Once texts are extracted as markdown, since text structure and title hierarchy are preserved, conclusion extraction becomes easier and more reliable, and finer chunking techniques become available.

## Ingestion

### Objective

Ingestion is the act of inserting data into a database to make it exploitable by downstream systems. Vector databases have emerged as the standard for semantic information retrieval. Texts of variable lengths are first split into chunks of more predictable length, and those chunks are then mapped to a vector space by an embedding model. The dimension of the vector space depends on the embedding model. Vector databases enable semantic retrieval based on the cosine similarity between the user query (potentially rewritten by a LLM) and the stored chunks. Efficient indexing methods like [Hierarchical Navigable Small World](https://arxiv.org/abs/1603.09320) (HNSW) allow for fast querying across millions of vectors.

### Methods

[Code](https://github.com/dataforgoodfr/13_democratiser_sobriete/tree/main/library/ingestion)

#### Chunking

Chunking is the process of cutting texts of variable length (extracted conclusions span two orders of magnitude in length) into so-called chunks of more predictable and manageable length, more amenable to AI processing.

Ideally, we would like semantically meaningful chunks like sentences or paragraphs. Since our text extraction process is noisy, as described above, this structure is not cleanly preserved in the extracted text, so we opt for a cruder but battle-tested method of chunking by length.

We use chunks of 1024 tokens with an overlap of 100 tokens. We use langchain’s [SentenceTransformersTokenTextSplitter](https://docs.langchain.com/oss/python/integrations/splitters/split_by_token#sentencetransformers) with the tokenizer of our embedding model : [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

Chunk size choice is a balance between cost and accuracy. In the context of [retrieval augmented generation](https://arxiv.org/abs/2005.11401), shorter chunks allow more precise retrieval and fitting more chunks into the final model’s context for answer generation. However, each chunk corresponds to a vector (or N vectors in the multi-vector setting, which we didn’t have resources to explore) to store in the vector database, so chunks have a fixed storage cost no matter their length. Few large chunks are thus cheaper to store and index in a vector database than many small chunks. Chunks of 1024 tokens are long enough that over 80% of extracted conclusions are long of 3 chunks or less, and short enough to be accurately represented by embedding models.

#### Vector embeddings

We used [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) as our embedding model. It has the best size-to-performance ratio according to the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) as of the time of the work. It’s also small enough to run on a CPU. It outputs vectors of dimension 1024\. To save on cost, we further reduce this dimension to 128 before insertion into the vector database, leveraging the fact that the model is trained with [Matrioshka Representation Learning](https://arxiv.org/abs/2205.13147) (MRL), a method that makes the model output vectors whose N first dimensions are still valid embedding for any N, with larger N yielding richer representations and smaller N rougher ones.

Using a small model and truncating its output affects the performance of retrieval. These choices were mostly motivated by cost, and the evaluation of their impacts on performances and subsequent optimization are left to future work.

## Taxonomy extraction

### Objective

Following a workshop, we developed a taxonomy to classify the papers. The taxonomy has two levels: some categories are extracted at the article level, while others are extracted at the chunk level. Further details on the extracted taxons are provided in this document: [https://docs.google.com/document/d/1AAXEJSQS5e4WJp5i8Sd1X56WPhRrhcoz/edit](https://docs.google.com/document/d/1AAXEJSQS5e4WJp5i8Sd1X56WPhRrhcoz/edit).

The taxonomy extraction phase serves three main objectives. First, it supports the policy analysis phase by enabling the labeling of text chunks according to different types of impacts. Second, article-level information will be incorporated into the chat system to help narrow the scope of generated answers. Third, the satisfiers identified in the literature will be used for the preparation of scientific publications.

### Methods

Two types of taxons are used in the policy analysis phase. The first concerns the geographical scope of the chunk, which allows the analysis to be differentiated according to the location under study. The second relates to the impacts and satisfiers identified within the text chunks. The specific challenges of the geographical taxonomy led us to treat it separately from the other two.

#### Geography

Detecting the relevant geographical elements from a text chunk is harder than merely extracting country names. Indeed, many chunks leave little context to infer the relevant country or geographical area, often only mentioning very specific names of regions or natural elements like rivers. We need a model with broad geographical knowledge to map those elements to their respective country. This is why we opted for a method based on a Large Language Model (LLM).

We used a dataset of 82 labeled examples to achieve two goals : tune the prompt using [DSPy](https://dspy.ai/), a python framework for structured prompt optimization, and select the best LLM model out of those made available by our inference provider (Scaleway). Model evaluation was based on their F1-score on a held-out test set representing 50% of the labeled dataset, or 40 samples. The best model scored a macro-average F1 score of 0.81. The selected model is [Mistral-Small-3.2-24B-Instruct](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506). Manual inspection across more examples confirmed that the results were satisfactory.

We ran the full extraction on the Jean Zay supercomputer. We used the [VLLM](https://github.com/vllm-project/vllm) library for fast inference with structured outputs for valid json formatting. Since VLLM is not compatible with DSPy in offline mode, we extracted the prompt from the tuned DSPy model beforehand. Processing the full dataset took about 70 GPU-hours using A100 GPUs.

#### Impacts and satisfiers

For impacts and satisfiers, we initially tried the same approach described above for geography, but we couldn’t get small open-source LLMs to generate satisfactory results. They would tend to generate a similar number of taxons for all chunks, which is far from correct and usually amounts to too many or too few. Even larger models, although better, didn’t produce satisfactory results.

Since extracting impacts and satisfiers requires semantic understanding but not the specific knowledge that large models excel at compared to smaller models, we switched to a classification approach based on the already-computed vector embeddings of chunks. This allowed us to use a very efficient classical machine learning approach from [scikit-learn](https://arxiv.org/abs/1201.0490). We framed the problem as multiclass classification using sklearn’s OneVsRestClassifier along with a logistic regression classifier.

The dataset of 82 labelled samples mentioned above wasn’t sufficient for the number of taxons in the taxonomy, leaving many taxons with very little or no training example. We thus used Gemini 3 Flash to label 2,000 chunks and then trained on this synthetic data. We used 5-fold cross-validation (CV) across the 82 hand-labeled chunks plus the 2,000 AI-labeled chunks, giving more weight both in training and evaluation to hand-labeled data. We used a weight ratio of 5-to-1 between ground truth labels and AI labels. CV was set to optimize the F2-score, a variant of the Fβ-score that gives twice more importance to recall over precision. The average evaluation result over the 5 folds is a F1 score of 0.65 with a precision of 0.54 and a recall of 0.80. There is large room for future improvements by hand-labelling more data, using a larger embedding model, or using a different approach like SetFit.

Using the already-computed embeddings, prediction for all chunks takes only a few minutes on a consumer laptop.

## Policy extraction

### Objective

### Methods

Extracting policies is a semantically complex generative task that only a LLM could perform without specific fine-tuning. For this task, hand-labelling a training dataset for DSPy required too much effort. We thus did without DSPy and carefully tuned the prompt, structured output format, and model selection before judging the response satisfactory. Again, the best cost-to-quality ratio was provided by [Mistral-Small-3.2-24B-Instruct](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506).

Additionally, we ask the LLM to tag each policy with a sector in the following list:

* BUILDING: Built Environment & Housing  
* URBAN: Urban Planning, Land Use & Spatial Development  
* MOBILITY: Mobility, Transport & Accessibility  
* ENERGY: Energy Systems  
* MATERIALS: Material Cycles, Waste & Water Systems  
* FOOD: Food, Agriculture & Fisheries Systems  
* INDUSTRY: Industrial Production & Manufacturing  
* LOGISTICS: Logistics & Supply Chains  
* NATURE: Natural Systems & Biodiversity  
* SOCIAL: Social Systems & Human Development (including wellbeing, health, education)  
* MACROECONOMIC: Macroeconomic & Cross-Sectoral Governance  
* LEISURE: Leisure, Tourism & Cultural Systems

Tagging means that we ask the model to extract each policy in the format: \[SECTOR\] policy.

### Results

We evaluated our best setup on a set of 1000 chunks, manually marking extracted policies as true positives, false positives, true negatives or false negatives. Results were excellent, with about 1% false positive rate and 2% false negative rate.

We then proceeded for the full extraction on Jean Zay with VLLM, as described above for the geographical taxonomy. Again, this took 70 GPU-hours on A100 GPUs. We obtained 1.46M policies.

## Policy clustering

### Objective

Policies can be analyzed across the literature only if a given policy can reliably be identified across publications despite different wordings and contexts. Policy extraction independently extracts policies from each chunk. From this first list of 1.46M policies, we then want to identify unique policies and build a bidirectional mapping between those and extracted policies. In machine learning terms, this corresponds to clustering.

### Methods

The policy clustering stage combined a challenging task \- determining the similarity of policies, which unfortunately doesn’t map neatly to the cosine similarity of their embeddings \- and a rather large scale of \~1.5M extracted policies.

High cosine similarity is sometimes due to semantic similarity and sometimes to morphological similarity, meaning it’s not an ideal distance metric for this task. For the same reason, we found the silhouette score, commonly used for evaluating the quality of clustering results, rather unreliable when compared to manual inspection. Manual inspection was therefore our main judge in evaluating clustering methods.

We first tried K-means and HDBSCAN, but results weren’t satisfactory: clusters tended to group policies that were too dissimilar. HDBSCAN also produced a high number of orphan policies. We also tried using an LLM to summarize all policies into a smaller set of unique policies. This worked well with a large-enough model ([Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)), but scaling this method to all policies was too expensive. We therefore stuck to clustering.

A challenge in clustering is the so-called chaining problem: if cos(A, B) \> 0.99 and cos(B, C) \> 0.99, cos(A, C) can still be below 0.99. A dense line of vectors, each very close to its neighbors, can end up as a single cluster even if its extremities are very far from each other. HDBSCAN, as a density-based method, is particularly prone to this. K-means solves it at the cost of often arbitrary cluster boundaries.

An elegant method to tackle this challenge is graph-based community detection, which identifies groups of densely connected nodes within a larger graph, accounting for both cluster density and cluster radius. We thus opted for this method, using the Leiden algorithm, which creates cohesive clusters whose granularity is easy to tune with a single parameter.

We used a machine with 128 GB of RAM for the following steps. To build the graph, we first index our policy embeddings with a HNSW index using [FAISS](https://arxiv.org/abs/2401.08281), setting the number of neighbours to 128 (HNSW being an approximate index, more neighbours is more accurate but more computationally demanding). We then build the graph by iterating on our policy embeddings using the range-search function of FAISS. This function returns points within a radius of epsilon to the reference point, measured in cosine similarity (in practice the HNSW index works with L2-distances, so we adapt the threshold using the formula L2 \= 2\*(1-cosine) for normalized vectors). We obtain a weighted graph whose number of edges depend on epsilon (an edge exists only between nodes whose cosine similarity is more than epsilon) and with edge weight equal to the cosine similarity, rescaled to a range of 0-1.

Once the graph is built, we use the NetworKit library to run the Leiden algorithm, as its parallel implementation is much faster than the reference leidenalg library. The parameter gamma can be used to control the granularity of the clusters. Since the edges are weighted, we set a rather low epsilon of 0.5 in the graph building phase so that we obtain a graph with few connected components (as it sets the minimum number of clusters), and use gamma as our main hyperparameter for controlling cluster cohesiveness.

In practice, given the large number of health-related works and policies that had a large influence on the other sectors, we perform the clustering separately by sector, as extracted in a tag in the policy extraction phase.

For naming clusters, we had two options: asking an LLM to summarize a manageable sample of policies (e.g. 50-100) within each cluster, or finding the medoid of each cluster. The medoid is the member of the cluster closest to its center, i.e. in our case the policy whose embedding is closest to the normalized mean of all embeddings within the cluster. We tried the second approach first as the cheapest and least energy-intensive of the two, and we judged the results good enough to keep them and not try the LLM-based option.

### Results

When working sector by sector, we found that a gamma of 7 best corresponds to the granularity we were looking for. After eliminating clusters of size 1 and 2, we obtain 2625 clusters.

## Clustered policy sufficiency classification (to complete-

### Objective

From this initial clustering, an evaluation performed by domain experts both on policy consistency & the clusters formed using the Leiden algorithm and its configuration. This evaluation was performed on a dataset presenting 90 clusters, with for each : 

- The bucket size (medium or large policies cluster)  
- A sector sub-taxonomy encoding 10 larger categories on each sector, selected by the domain experts  
- The cluster medoid  
- 5 policies randomly selected from the cluster

The experts concluded on a ratio of 60% perfectly coherent clusters, 28% partly and 

## 

## Policy impacts extraction

### Objectives

Once unique policies are identified, i.e. once clusters are named, our goal is to extract and compile their reported effects across the literature. We aim, for each chunk, each policy present in that chunk, and each impact detected in that chunk during the taxonomy extraction phase, to determine the direction of impact of the policy on the impact dimension as reported in the chunk. Here, the direction of impact can take four values: positive, negative, neutral, and unknown.

### Methods

Since this task requires fine contextual understanding, we again resorted to using a LLM with structured outputs. For each chunk containing at least one policy, we list the policies it contains and map each of them to its cluster. Then for each cluster name and each impact taxon previously identified within the chunk, we ask the LLM for the direction of impact of the policy as stated in the chunk. As in previous steps, we kept Mistral-Small-3.2-24B-Instruct for its cost-to-performance ratio.

The above method requires a combinatorial number of predictions : each chunk that contains at least one policy corresponds to several dozen unique (policy cluster, impact dimension) tuples. To make this computationally manageable, we leverage the fact that a LLM reads input tokens much faster than it generates output tokens, and that LLVM provides Automatic Prefix Caching (APC), meaning that if two or more prompts in each batch share the same prefix, the computational cost of reading it is paid only once. This means that it’s much faster to ask a single output token for each of:

- (system prompt, chunk text, policy cluster, impact1)  
- (system prompt, chunk text, policy cluster, impact2)  
- (system prompt, chunk text, policy cluster, impact3)  
- …

than asking a model to classify all impacts at once for (system prompt, chunk text, policy cluster). We use [choice-based structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/#offline-inference) (as opposed to json-based) to force the model to output a valid class. This method sets all logits to \-∞ except for valid tokens, meaning the probability of invalid tokens being generated is 0\.

With these optimizations, we obtain a throughput of \~300,000 tok/s in input and \~200 tok/s in output on a single A100 GPU, confirming that input caching is heavily used.

The results of these predictions allows us to form a table that, for each policy cluster and each impact dimension, gives the number of positive/negative/neutral pieces of evidence (i.e. chunks) and lists the corresponding chunk identifiers. This table gives us, for each cluster, a quantitative summary of its impacts across all dimensions that are cited in the library.

### Results

Takes \~28 GPU hours total on A100s.

### Future work

To further decrease the cost of this step, we could frame it as a classification problem. We would first build a dataset where each row corresponds to a (chunk, policy, impact) triplet and then ask the classifier to choose a direction of impact. We could then leverage a small hand-labeled dataset to train a SetFit model, enabling cheap and fast classification.

## Classification as Sufficiency / Possible Sufficiency / Not Sufficiency

### Objective

As a last step of the pipeline, we aim to identify policies that meet all criteria of sufficiency.

### Method

We obtained from the impacts extraction phase an assessment of the impacts of each cluster along each dimension of the taxonomy, with for each direction of impact (positive, negative, neutral), a list of evidence chunks for a given (cluster, impact dimension, direction) tuple.

We leverage this to build contextual information to be fed into an LLM of the final classification phase. For each cluster, we create the context by concatenating:

* the cluster name (representative policy) ;  
* a quantitative summary of impacts along each dimension ;  
* at most two example chunks for each (impact dimension, direction) tuple, for additional contextual and qualitative information.

With this impact, we ask the model to first classify the policy as: 

* DECARBONATION: supply-side energy decarbonation (renewables, nuclear...)  
* EFFICIENCY: efficiency-based policy (optimization, energy intensity reduction...)  
* SUFFICIENCY-COMPATIBLE: avoiding upfront the demand for resources  
* NOT-COMPATIBLE: no or negative impact on resource use

Then, if the class is SUFFICIENCY-COMPATIBLE, to classify further as:

* Sufficiency (S): Primarily and directly contributes to human well-being while significantly reducing resource demand and staying within planetary boundaries.  
* Potential Sufficiency (PS): Has the potential to achieve sufficiency, but requires explicit, specific, and significant transformation (e.g., equity corrections, policy integration) to overcome limitations and align with all boundaries.  
* Not Sufficiency (NS): The policy's effect on human needs is indirect, often resulting in breaches of planetary limits or social foundations. The policy might turn out to be a violator of basic needs.  Or intrinsically undermines one or both boundaries (social foundations or planetary limits).

We do so using structured outputs with additional reasoning fields before each classification stage, to leave the model more “room for thinking”. Qwen3-235B-A22B-Instruct-2507 was used for this stage.

### Results

# Integration into Chat Sufficiency

## Objectives

ChatSufficiency is a community-facing conversational interface making the results of our work broadly accessible.

## Methods

[Code](https://github.com/dataforgoodfr/13_democratiser_sobriete/tree/main/rag_system)

Chat Sufficiency is an AI chatbot based on [Retrieval Augmented Generation](https://arxiv.org/abs/2005.11401) (RAG). RAG tackles hallucinations by directly providing query-relevant knowledge to the LLM before it answers. A RAG system is built around a retriever and a generator. The retriever analyses the user’s query and fetches relevant knowledge from a knowledge base or document base. This context is then given along with the user query to the generator, usually a LLM. The LLM can thus answer the user query with precise and up-to-date information. A simple but faithful mental model to understand RAG is to imagine a LLM summarizing the first N results returned by a search engine, extracting only information relevant to the user query.

Most of the complexity of a RAG system sits in the retriever and the upstream effort to build the knowledge base. Our work is a good illustration of this, as the knowledge base we used is the result of policy analysis. We convert each cluster name into a vector embedding using Qwen3-Embedding-8B (affordable due to the much lower number of clusters compared to chunks), that we insert into a Qdrant vector database along with a payload containing the quantitative summary of impacts obtained from policy analysis. We also embed all chunks in the library using Qwen3-Embedding-0.6B (much cheaper) and insert them into a separate collection on Qdrant.

When the user asks a question, the following steps happen in this order:

1. The query is rewritten by a LLM. This step is needed in particular in multi-turn conversations where the user might reference previous messages. If the LLM judges at this step that the query is off-topic or can be answered without specific knowledge, it answers immediately without triggering the following steps.  
2. The rewritten query is embedded using the same embedding model, and the vector database is queried for the closest N results in cosine similarity. We obtain the policies most relevant to the user query as well as their impacts, as obtained from policy analysis.  
3. A LLM rates the relevance of each policy against the user query on a scale of 0-9 and we keep those that score above 5\.  
4. The retrieved information contains the identifiers of chunks that were used in evaluating the policy impact. We want to retrieve some of those chunks to provide additional context to the next step. However, many of those chunks are irrelevant to the user query : they simply correspond to a (policy cluster, impact dimension, impact direction) tuple that is relevant to the user query, but that criterion is too coarse to yield relevant results. Thus, we run a second semantic retrieval against chunks, filtering by the list of chunk identifiers obtained from the policy impact assessment.  
5. The policy and chunk information is passed in the context of the final generation model, along with the user query and previous messages, and the model answers.

This system is split between a TypesScript frontend built with SvelteKit and a Python backend built with FastAPI that handles the above logic. We don’t use a specific orchestration library like LangChain but standard python with FastAPI and Pydantic. As previously stated, our vector database is Qdrant. The LLM used for reranking is Mistral-Small-3.2-24B-Instruct, and that used for query rewriting and answer generation is [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507), both provided by Scaleway Generative API.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAloAAAEnCAYAAABxD+1oAABPFklEQVR4Xu29eZQcxYHu6z/emznjeXfe+M59nhmfN3P87pt78Xi8XKzxJiFgbDEYrGYzW9ssFpjGiIHBNLINGEl40AK0aYTwIIEQ+GE1yGAJ1AYtLMKSDZYREotYGmQsGSRkFgkLsRvi9RelKGV+WV1VkRWZVZX1/c75ncyMzMrKisyO+joyqup9RgghhBBCZML7uEAIIYQQQoRBQUsIIYQQIiMUtIQQQgghMkJBSwghhBAiIxS0hBBCCCEyonbQemuXMU8OSillPr66jVuhuln/ojG3PSOllPn46tvcCiWpHbReeFxKKfM1JU+/IqWU+Xn/89wKJVHQklK2ninhRlBKKbM0eNB65cL/Zsz8L7SEf5w72uy68rPJBlpK2f6mxDV+T+4w5n9e+IAZ+1NTtw++kGxEpZSymsGDFoedlpAbaCll+5sS1/j9zam3JIJULT90+q2JRlRKKauZWdB63/veZ6eP/sdnzQOTP212XbVfMgBV8aCP/1WibN3UT5snp3+uvHxB14fN7/pGmz/O+5fEtjG5gZZStr8pcY1fNGiBl94ozR94mzFfHizN377JmJd3lytoSSnTmHnQeuT7nzFrp3zaXHjY/2M+8rfvNwOn/pM58p8/aC7v/h/mA3/+v5mh4eD05U/8lfnGvh+y2885YS87/ce//XPz13/xJ2bb5fuYb/3r35kNw4ENQevpmaWg9fUxf2sO3fu/mc2XjrbPMWGfvzXLzv5f5vn+fcyoD/8XM+Z//J8KWlIW2ZS4xi8atA69fU+Yuvc5Y+ZuMOYLi4054o5S2Sn3KGhJKdOZW9B67gdjzCs/3Nf8uOejtvz//sCfmhW9/2s4QH3GLDnzE7bn6/jRf1N+/LiP/lc7veNbnzRnHfB35uELP1MOWtjnJUf/gzn7wL8vBy2sQy8YerlevWo/BS0pi25KXOMXDVoIUrve3tO7Bbp+ZsyRS0tlfesUtKSU6cw0aEEXtKYd8d/NXn/z/nLQQm/Uvv/zL8tBC2Vf+McPlB//0Q/9ufm//sv/bueP/vQHzYEf+6+xHi30YEV7tFzQ2nLZGPPJv/s/FLSkLLopcY0f92jtt8iYax4r3TbE7cPz7jdm6SZjdu4OYApaUso0Bg9ab/b/456A0wLumPpXyQZaStn+psQ1fof1/6IcoOr1oJn3JhpRKaWsZvCgZbY9al75/gfNq9M/1HR3XvbRZOMspSyGKYk2gAhb6Nmqx7866spEAyqllLUMH7SklDIPU8KNoJRSZqmClpSyPU0JN4JSSpmlClpSyvY0JdwISilllipoSSnb05RwIyillFmqoCWlbE9Two2glFJmqYKWlLI9TQk3glJKmaUtEbR2/f635rVXXpQF8Y8vPJk4x2l845UXEvuWnWFd11BKuBGUUsosbYmgJYrFu+++mzzHnu7a/gLvVnQaFa6LmCnhRlBKKbNUQUtkA59jT3e+8jLvUXQaFa6LmCnhRlBKKbO05YLW1q1bm+oTTzwROx6REj7HnkaD1rp16xLnqRX97W9/u+f1i8apcF3ETAk3gml88803zTvvvCN3++z21xN1FFJ+vk6X6yekf3j11cTzdbIv7fhDoo58bamg9fDDDyfevJrh+vXrIy9OpILPsafRoMXnp5X93e9+F6kE0RAVrouYKeFGMI0izh//+MdEHYXy2e2v8dN1PFxHIRVJuI58bcmgtWbNGvO+973PfPvb3068kYXwK1/5SqIsqoJWAPgce1opaOGagNFzddttt5n/+I//KC//5Cc/SZxP+NxzzyXKoi5fvtw89NBDifKo2Pfxxx9vnn322cQ6p4JWQCpcFzFTwo1gGkUcBa184ToKqUjCdeRrywYtvJm9//3vN3/yJ39iNm/ebPbdd1/T399v7rzzTrvNb37zG/P444+bs88+23zyk580K1assG+6V111lfn0pz9tbwGibP/99zcnnXSSfdzTTz9tDj74YHPQQQeZww47zNx3333m+uuvN3/6p39qxo8fb2/9YN8KWgHgc+zpSEHrAx/4gPn1r39tw9VXv/rVRND6t3/7N/P1r3/dbnPttdfa8wyj4egv//IvzaZNm+z81772NdPd3W0uvfRSc9ddd5l99tnHrvvVr35l5z/0oQ+ZJ5980nrmmWeaI4880l6fWMb6D3/4w+b+++9X0MqCCtdFzJRwI5hGR60A34gI/u1CXkEry/puVSvBdRRSR5513epDdriOfG3ZoOVOAN4UMT3kkEMSJwdvjh/5yEfsGyuW8aaLKYIUpghSZ511lp13AQ1vjlh/9NFH2+XZs2ebUaNG2VCmoBUQPseejhS0MH3ggQdsuDrmmGMqBi0EMLd877332u2j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>