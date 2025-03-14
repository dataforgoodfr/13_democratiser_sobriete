PROMPT_TEMPLATE_BEGINNING = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and no prior knowledge, \
generate only questions based on the below query.
"""

# OK
POLITICIAN_QA_GENERATE_PROMPT_TMPL = (
    PROMPT_TEMPLATE_BEGINNING
    + """\
You are a policymaker responsible for shaping regulations \
and incentives. Your task is to generate {num_questions_per_chunk} \
questions that policymakers might ask when considering \
sufficiency-oriented policies in areas such as energy consumption, \
urban planning, or economic systems. Ensure the questions cover \
different aspects of policy implementation and feasibility.
"""
)

CITIZEN_QA_GENERATE_PROMPT_TMPL = (
    PROMPT_TEMPLATE_BEGINNING
    + """\
You are an informed citizen seeking to understand sufficiency. \
Your task is to generate {num_questions_per_chunk} questions that \
an everyday person might ask to explore how sufficiency affects \
their daily life, financial savings, or overall well-being. The \
questions should be diverse, covering personal, social, and \
economic perspectives.
"""
)

SCIENTIST_QA_GENERATE_PROMPT_TMPL = (
    PROMPT_TEMPLATE_BEGINNING
    + """\
You are a researcher exploring sufficiency studies. Your task \
is to generate {num_questions_per_chunk} questions that scientists \
might pose to investigate theoretical gaps, methodological \
challenges, or future research directions. Ensure the questions \
address different aspects of sufficiency research, such as \
empirical validation, interdisciplinary links, or policy implications.
"""
)


DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and no prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."
"""

###### Evaluation Prompt ######
# These prompts are part of automated LLM-as-a-Judge on 3 different criteria to assess the quality of the generated question given a context.
# Groundedness , Relevance , Stand-alone question

# Assessing whether the question can be answered using the provided context
GROUNDNESS_CRITIC_PROMPT = """
You will receive a research-based context and a question.  

Your task is to evaluate how clearly and unambiguously the given question can be answered using the provided context. Consider the following factors:  
- Does the context contain sufficient information to answer the question?  
- Can the answer be derived without speculation or external knowledge?  

Provide your rating on a scale from 1 to 5:  
- **1**: The question cannot be answered at all given the context.  
- **5**: The question can be answered clearly and unambiguously with the given context.  

Provide your answer as follows:  

Evaluation: (your justification for the score, in text form)  
Total score: (your score, a number between 1 and 5)  

You MUST provide values for 'Evaluation:' and 'Total score:' in your answer and nothing else.  

Now, here is the question and the context.  

**Question:** {question}  
**Context:** {context}  
"""

# Assessing how useful the question is for different user groups
RELEVANCE_CRITIC_PROMPT = """
You will receive a question.  

Your task is to evaluate how relevant and useful this question is for users of the chatbot, including:  
- **Academics** (e.g., researchers analyzing sufficiency in various fields)  
- **Policymakers** (e.g., government officials making decisions based on sufficiency research)  
- **Non-technical citizens** (e.g., individuals seeking to understand sufficiency in daily life)  

Rate the question’s usefulness on a scale from 1 to 5:  
- **1**: The question is not useful at all for any of these groups.  
- **5**: The question is highly relevant and insightful for at least one of these groups.  

Provide your answer as follows:  

Evaluation: (your justification for the score, in text form)  
Total score: (your score, a number between 1 and 5)  

You MUST provide values for 'Evaluation:' and 'Total score:' in your answer.  

Now, here is the question.  

**Question:** {question}
"""

# Assessing whether the question makes sense on its own without additional context
STANDALONE_CRITIC_PROMPT = """
You will receive a question.  

Your task is to evaluate how self-contained this question is, meaning whether it makes sense **on its own** without requiring additional context.  

Consider the following:  
- If the question refers to an unspecified “context” or “document,” it likely depends on external information and should receive a lower score.  
- If the question contains technical terms (e.g., sufficiency-related metrics, policy frameworks, or statistical concepts), that is acceptable as long as an informed user (such as an academic, policymaker, or citizen with access to relevant resources) can understand it.  

Rate the question’s independence on a scale from 1 to 5:  
- **1**: The question is unclear or depends on missing context to be understood.  
- **5**: The question makes full sense on its own, even if technical.  

Provide your answer as follows:  

Evaluation: (your justification for the score, in text form)  
Total score: (your score, a number between 1 and 5)  

You MUST provide values for 'Evaluation:' and 'Total score:' in your answer.  

Now, here is the question.  

**Question:** {question}  

"""
