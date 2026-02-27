import asyncio
import os
import json
import copy
import math
import random
import re
import logging
from .utils import *

logger = logging.getLogger(__name__)


################### check title in page #########################################################
async def check_title_appearance(item, page_list, start_index=1, model=None):
    title=item['title']
    if 'physical_index' not in item or item['physical_index'] is None:
        return {'list_index': item.get('list_index'), 'answer': 'no', 'title':title, 'page_number': None}


    page_number = item['physical_index']
    page_text = page_list[page_number-start_index][0]


    prompt = f"""
    Your job is to check if the given section appears or starts in the given page_text.

    Note: do fuzzy matching, ignore any space inconsistency in the page_text.

    The given section title is {title}.
    The given page_text is {page_text}.

    Reply format:
    {{

        "thinking": <why do you think the section appears or starts in the page_text>
        "answer": "yes or no" (yes if the section appears or starts in the page_text, no otherwise)
    }}
    Directly return the final JSON structure. Do not output anything else."""

    response = await _llm_json(model=model, prompt=prompt)
    if 'answer' in response:
        answer = response['answer']
    else:
        answer = 'no'
    return {'list_index': item['list_index'], 'answer': answer, 'title': title, 'page_number': page_number}


async def check_title_appearance_in_start(title, page_text, model=None, _logger=None):
    prompt = f"""
    You will be given the current section title and the current page_text.
    Your job is to check if the current section starts in the beginning of the given page_text.
    If there are other contents before the current section title, then the current section does not start in the beginning of the given page_text.
    If the current section title is the first content in the given page_text, then the current section starts in the beginning of the given page_text.

    Note: do fuzzy matching, ignore any space inconsistency in the page_text.

    The given section title is {title}.
    The given page_text is {page_text}.

    reply format:
    {{
        "thinking": <why do you think the section appears or starts in the page_text>
        "start_begin": "yes or no" (yes if the section starts in the beginning of the page_text, no otherwise)
    }}
    Directly return the final JSON structure. Do not output anything else."""

    response = await _llm_json(model=model, prompt=prompt)
    log = _logger or logger
    log.debug(f"check_title_appearance_in_start response: {response}")
    return response.get("start_begin", "no")


async def check_title_appearance_in_start_concurrent(structure, page_list, model=None, _logger=None):
    log = _logger or logger
    log.info("Checking title appearance in start concurrently")

    # skip items without physical_index
    for item in structure:
        if item.get('physical_index') is None:
            item['appear_start'] = 'no'

    # only for items with valid physical_index
    tasks = []
    valid_items = []
    for item in structure:
        if item.get('physical_index') is not None:
            page_text = page_list[item['physical_index'] - 1][0]
            tasks.append(check_title_appearance_in_start(item['title'], page_text, model=model, _logger=_logger))
            valid_items.append(item)

    log.info(f"[appear_start] Checking {len(valid_items)} items concurrently")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(valid_items, results):
        if isinstance(result, Exception):
            log.error(f"Error checking start for {item['title']}: {result}")
            item['appear_start'] = 'no'
        else:
            item['appear_start'] = result

    yes_count = sum(1 for item in valid_items if item.get('appear_start') == 'yes')
    log.info(f"[appear_start] Results: {yes_count} yes, {len(valid_items) - yes_count} no")
    return structure


async def toc_detector_single_page(content, model=None):
    prompt = f"""
    Your job is to detect if there is a table of content provided in the given text.

    Given text: {content}

    return the following JSON format:
    {{
        "thinking": <why do you think there is a table of content in the given text>
        "toc_detected": "<yes or no>",
    }}

    Directly return the final JSON structure. Do not output anything else.
    Please note: abstract,summary, notation list, figure list, table list, etc. are not table of contents."""

    json_content = await _llm_json(model=model, prompt=prompt)
    logger.info(f"[toc_detector] Page content_len={len(content)} => {json_content.get('toc_detected', 'no')}")
    return json_content.get('toc_detected', 'no')


async def check_if_toc_extraction_is_complete(content, toc, model=None):
    prompt = f"""
    You are given a partial document  and a  table of contents.
    Your job is to check if the  table of contents is complete, which it contains all the main sections in the partial document.

    Reply format:
    {{
        "thinking": <why do you think the table of contents is complete or not>
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    prompt = prompt + '\n Document:\n' + content + '\n Table of contents:\n' + toc
    json_content = await _llm_json(model=model, prompt=prompt)
    return json_content.get('completed', 'no')


async def check_if_toc_transformation_is_complete(content, toc, model=None):
    prompt = f"""
    You are given a raw table of contents and a  table of contents.
    Your job is to check if the  table of contents is complete.

    Reply format:
    {{
        "thinking": <why do you think the cleaned table of contents is complete or not>
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    prompt = prompt + '\n Raw Table of contents:\n' + content + '\n Cleaned Table of contents:\n' + toc
    json_content = await _llm_json(model=model, prompt=prompt)
    return json_content.get('completed', 'no')

async def extract_toc_content(content, model=None):
    prompt = f"""
    Your job is to extract the full table of contents from the given text, replace ... with :

    Given text: {content}

    Directly return the full table of contents content. Do not output anything else."""

    response, finish_reason = await _llm_completion(model=model, prompt=prompt)

    if_complete = await check_if_toc_transformation_is_complete(content, response, model)
    logger.info(f"[extract_toc_content] Initial extraction: content_len={len(content)}, finish_reason={finish_reason}, complete={if_complete}")
    if if_complete == "yes" and finish_reason == "finished":
        return response

    chat_history = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    prompt = f"""please continue the generation of table of contents , directly output the remaining part of the structure"""
    new_response, finish_reason = await _llm_completion(model=model, prompt=prompt, chat_history=chat_history)
    response = response + new_response
    if_complete = await check_if_toc_transformation_is_complete(content, response, model)

    iteration_count = 0
    while not (if_complete == "yes" and finish_reason == "finished"):
        iteration_count += 1
        chat_history.append({"role": "user", "content": prompt})
        new_response, finish_reason = await _llm_completion(model=model, prompt=prompt, chat_history=chat_history)
        chat_history.append({"role": "assistant", "content": new_response})
        response = response + new_response
        if_complete = await check_if_toc_transformation_is_complete(content, response, model)
        logger.info(f"[extract_toc_content] Continuation round {iteration_count}: complete={if_complete}, finish_reason={finish_reason}")

        if iteration_count >= 5:
            raise Exception('Failed to complete table of contents after maximum retries')

    return response

async def detect_page_index(toc_content, model=None):
    logger.info('Detecting page index in ToC')
    prompt = f"""
    You will be given a table of contents.

    Your job is to detect if there are page numbers/indices given within the table of contents.

    Given text: {toc_content}

    Reply format:
    {{
        "thinking": <why do you think there are page numbers/indices given within the table of contents>
        "page_index_given_in_toc": "<yes or no>"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    json_content = await _llm_json(model=model, prompt=prompt)
    logger.info(f"[detect_page_index] Result: {json_content.get('page_index_given_in_toc', 'no')}")
    return json_content.get('page_index_given_in_toc', 'no')

async def toc_extractor(page_list, toc_page_list, model):
    def transform_dots_to_colon(text):
        text = re.sub(r'\.{5,}', ': ', text)
        # Handle dots separated by spaces
        text = re.sub(r'(?:\. ){5,}\.?', ': ', text)
        return text

    toc_content = ""
    for page_index in toc_page_list:
        toc_content += page_list[page_index][0]
    toc_content = transform_dots_to_colon(toc_content)
    logger.info(f"[toc_extractor] Extracting from ToC pages {toc_page_list}, combined_len={len(toc_content)} chars")
    has_page_index = await detect_page_index(toc_content, model=model)
    logger.info(f"[toc_extractor] Result: page_index_given={has_page_index}")

    return {
        "toc_content": toc_content,
        "page_index_given_in_toc": has_page_index
    }




async def toc_index_extractor(toc, content, model=None):
    logger.info(f"[toc_index_extractor] Mapping {len(toc) if isinstance(toc, list) else 'N/A'} ToC items to pages, content_len={len(content)} chars")
    tob_extractor_prompt = """
    You are given a table of contents in a json format and several pages of a document, your job is to add the physical_index to the table of contents in the json format.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format:
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "physical_index": "<physical_index_X>" (keep the format)
        },
        ...
    ]

    Only add the physical_index to the sections that are in the provided pages.
    If the section is not in the provided pages, do not add the physical_index to it.
    Directly return the final JSON structure. Do not output anything else."""

    prompt = tob_extractor_prompt + '\nTable of contents:\n' + str(toc) + '\nDocument pages:\n' + content
    json_content = await _llm_json(model=model, prompt=prompt)
    logger.info(f"[toc_index_extractor] Result: {len(json_content) if isinstance(json_content, list) else 'N/A'} items returned")
    return json_content



async def toc_transformer(toc_content, model=None):
    logger.info(f"[toc_transformer] Input ToC length: {len(toc_content)} chars")
    init_prompt = """
    You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

    structure is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    IMPORTANT: If a section title includes a leading number or numbering scheme (e.g. "1.2 Background", "Chapter 3: Methods", "A.1 Appendix"), you MUST include that number as part of the title.

    The response should be in the following JSON format:
    {
    table_of_contents: [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section, including any leading section number>,
            "page": <page number or None>,
        },
        ...
        ],
    }
    You should transform the full table of contents in one go.
    Directly return the final JSON structure, do not output anything else. """

    prompt = init_prompt + '\n Given table of contents\n:' + toc_content
    last_complete, finish_reason = await _llm_completion(model=model, prompt=prompt)
    if_complete = await check_if_toc_transformation_is_complete(toc_content, last_complete, model)
    logger.info(f"[toc_transformer] First attempt: complete={if_complete}, finish_reason={finish_reason}")
    if if_complete == "yes" and finish_reason == "finished":
        last_complete = extract_json(last_complete)
        if isinstance(last_complete, list):
            toc_list = last_complete
        elif isinstance(last_complete, dict):
            toc_list = last_complete.get('table_of_contents', [])
        else:
            toc_list = []
        cleaned_response=convert_page_to_int(toc_list)
        logger.info(f"[toc_transformer] Output: {len(cleaned_response)} items")
        return cleaned_response

    last_complete = get_json_content(last_complete)
    iteration_count = 0
    while not (if_complete == "yes" and finish_reason == "finished"):
        iteration_count += 1
        if iteration_count > 10:
            logger.warning("[toc_transformer] Max iterations reached, returning best result so far")
            break
        position = last_complete.rfind('}')
        if position != -1:
            last_complete = last_complete[:position+1]
        prompt = f"""
        Your task is to continue the table of contents json structure, directly output the remaining part of the json structure.
        The response should be in the following JSON format:

        The raw table of contents json structure is:
        {toc_content}

        The incomplete transformed table of contents json structure is:
        {last_complete}

        Please continue the json structure, directly output the remaining part of the json structure."""

        new_complete, finish_reason = await _llm_completion(model=model, prompt=prompt)

        if new_complete.startswith('```json'):
            new_complete = get_json_content(new_complete)
        last_complete = last_complete + new_complete

        if_complete = await check_if_toc_transformation_is_complete(toc_content, last_complete, model)
        logger.info(f"[toc_transformer] Continuation round {iteration_count}: complete={if_complete}, finish_reason={finish_reason}")


    last_complete = extract_json(last_complete)

    if isinstance(last_complete, list):
        toc_list = last_complete
    elif isinstance(last_complete, dict):
        toc_list = last_complete.get('table_of_contents', [])
    else:
        toc_list = []
    cleaned_response=convert_page_to_int(toc_list)
    logger.info(f"[toc_transformer] Output: {len(cleaned_response)} items")
    return cleaned_response




async def find_toc_pages(start_page_index, page_list, opt, _logger=None):
    log = _logger or logger
    log.info(f"[find_toc_pages] Scanning from page {start_page_index}, max_check={opt.toc_check_page_num}")
    last_page_is_yes = False
    toc_page_list = []
    i = start_page_index

    while i < len(page_list):
        # Only check beyond max_pages if we're still finding TOC pages
        if i >= opt.toc_check_page_num and not last_page_is_yes:
            break
        detected_result = await toc_detector_single_page(page_list[i][0],model=opt.model)
        if detected_result == 'yes':
            log.info(f'Page {i} has toc')
            toc_page_list.append(i)
            last_page_is_yes = True
        elif detected_result == 'no' and last_page_is_yes:
            log.info(f'Found the last page with toc: {i-1}')
            break
        i += 1

    if not toc_page_list:
        log.info('No toc found')

    log.info(f"[find_toc_pages] Result: {len(toc_page_list)} ToC pages found: {toc_page_list}")
    return toc_page_list

def remove_page_number(data):
    if isinstance(data, dict):
        data.pop('page_number', None)
        for key in list(data.keys()):
            if 'nodes' in key:
                remove_page_number(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_page_number(item)
    return data

def extract_matching_page_pairs(toc_page, toc_physical_index, start_page_index):
    pairs = []
    for phy_item in toc_physical_index:
        for page_item in toc_page:
            if phy_item.get('title') == page_item.get('title'):
                physical_index = phy_item.get('physical_index')
                if physical_index is not None and int(physical_index) >= start_page_index:
                    pairs.append({
                        'title': phy_item.get('title'),
                        'page': page_item.get('page'),
                        'physical_index': physical_index
                    })
    return pairs


def calculate_page_offset(pairs):
    differences = []
    for pair in pairs:
        try:
            physical_index = pair['physical_index']
            page_number = pair['page']
            difference = physical_index - page_number
            differences.append(difference)
        except (KeyError, TypeError):
            continue

    if not differences:
        return None

    difference_counts = {}
    for diff in differences:
        difference_counts[diff] = difference_counts.get(diff, 0) + 1

    most_common = max(difference_counts.items(), key=lambda x: x[1])[0]

    return most_common

def add_page_offset_to_toc_json(data, offset):
    for i in range(len(data)):
        if data[i].get('page') is not None and isinstance(data[i]['page'], int):
            data[i]['physical_index'] = data[i]['page'] + offset
            del data[i]['page']

    return data



def page_list_to_group_text(page_contents, token_lengths, max_tokens=PAGEINDEX_TOC_MAX_TOKENS_PER_CHUNK, overlap_page=1):
    num_tokens = sum(token_lengths)

    if num_tokens <= max_tokens:
        # merge all pages into one text
        page_text = "".join(page_contents)
        return [page_text]

    subsets = []
    current_subset = []
    current_token_count = 0

    expected_parts_num = math.ceil(num_tokens / max_tokens)
    average_tokens_per_part = math.ceil(((num_tokens / expected_parts_num) + max_tokens) / 2)

    for i, (page_content, page_tokens) in enumerate(zip(page_contents, token_lengths)):
        if current_token_count + page_tokens > average_tokens_per_part:

            subsets.append(''.join(current_subset))
            # Start new subset from overlap if specified
            overlap_start = max(i - overlap_page, 0)
            current_subset = page_contents[overlap_start:i]
            current_token_count = sum(token_lengths[overlap_start:i])

        # Add current page to the subset
        current_subset.append(page_content)
        current_token_count += page_tokens

    # Add the last subset if it contains any pages
    if current_subset:
        subsets.append(''.join(current_subset))

    logger.info(f'Divided page_list into {len(subsets)} groups')
    return subsets

async def add_page_number_to_toc(part, structure, model=None):
    fill_prompt_seq = """
    You are given an JSON structure of a document and a partial part of the document. Your task is to check if the title that is described in the structure is started in the partial given document.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    If the full target section starts in the partial given document, insert the given JSON structure with the "start": "yes", and "start_index": "<physical_index_X>".

    If the full target section does not start in the partial given document, insert "start": "no",  "start_index": None.

    The response should be in the following format.
        [
            {
                "structure": <structure index, "x.x.x" or None> (string),
                "title": <title of the section>,
                "start": "<yes or no>",
                "physical_index": "<physical_index_X> (keep the format)" or None
            },
            ...
        ]
    The given structure contains the result of the previous part, you need to fill the result of the current part, do not change the previous result.
    Directly return the final JSON structure. Do not output anything else."""

    logger.info(f"[add_page_number_to_toc] Matching {len(structure)} items against text ({len(part) if isinstance(part, str) else sum(len(p) for p in part)} chars)")
    prompt = fill_prompt_seq + f"\n\nCurrent Partial Document:\n{part}\n\nGiven Structure\n{json.dumps(structure, indent=2)}\n"
    json_result = await _llm_json(model=model, prompt=prompt)

    for item in json_result:
        if 'start' in item:
            del item['start']
    return json_result


def remove_first_physical_index_section(text):
    """
    Removes the first section between <physical_index_X> and <physical_index_X> tags,
    and returns the remaining text.
    """
    pattern = r'<physical_index_\d+>.*?<physical_index_\d+>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Remove the first matched section
        return text.replace(match.group(0), '', 1)
    return text

### add verify completeness
async def generate_toc_continue(toc_content, part, model=None):
    model = model or DEFAULT_MODEL
    logger.info(f"[generate_toc_continue] Previous items: {len(toc_content)}, new text: {len(part)} chars")
    prompt = """
    You are an expert in extracting hierarchical tree structure.
    You are given a tree structure of the previous part and the text of the current part.
    Your task is to continue the tree structure from the previous part to include the current part.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency. IMPORTANT: If a section title includes a leading number or numbering scheme (e.g. "1.2 Background", "Chapter 3: Methods", "A.1 Appendix"), you MUST include that number as part of the title.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X. \

    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format.
        [
            {
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title including any leading section number>,
                "physical_index": "<physical_index_X> (keep the format)"
            },
            ...
        ]

    Directly return the additional part of the final JSON structure. Do not output anything else."""

    prompt = prompt + '\nGiven text\n:' + part + '\nPrevious tree structure\n:' + json.dumps(toc_content, indent=2)
    response, finish_reason = await _llm_completion(model=model, prompt=prompt)
    if finish_reason == 'finished':
        result = extract_json(response)
        logger.info(f"[generate_toc_continue] finish_reason={finish_reason}, extracted {len(result) if isinstance(result, list) else 'N/A'} new items")
        return result
    else:
        raise Exception(f'finish reason: {finish_reason}')

### add verify completeness
async def generate_toc_init(part, model=None):
    logger.info(f"[generate_toc_init] Input text: {len(part)} chars")
    prompt = """
    You are an expert in extracting hierarchical tree structure, your task is to generate the tree structure of the document.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency. IMPORTANT: If a section title includes a leading number or numbering scheme (e.g. "1.2 Background", "Chapter 3: Methods", "A.1 Appendix"), you MUST include that number as part of the title.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X.

    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format.
        [
            {{
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title including any leading section number>,
                "physical_index": "<physical_index_X> (keep the format)"
            }},

        ],


    Directly return the final JSON structure. Do not output anything else."""

    prompt = prompt + '\nGiven text\n:' + part
    response, finish_reason = await _llm_completion(model=model, prompt=prompt)

    if finish_reason == 'finished':
        result = extract_json(response)
        logger.info(f"[generate_toc_init] finish_reason={finish_reason}, extracted {len(result) if isinstance(result, list) else 'N/A'} items")
        return result
    else:
        raise Exception(f'finish reason: {finish_reason}')

async def process_no_toc(page_list, start_index=1, model=None, _logger=None):
    log = _logger or logger
    log.info(f"[process_no_toc] Processing {len(page_list)} pages from start_index={start_index}")
    page_contents=[]
    token_lengths=[]
    for page_index in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))
    group_texts = page_list_to_group_text(page_contents, token_lengths)
    log.info(f"[process_no_toc] Split into {len(group_texts)} text groups")

    toc_with_page_number= await generate_toc_init(group_texts[0], model)
    log.info(f"[process_no_toc] Initial extraction: {len(toc_with_page_number) if isinstance(toc_with_page_number, list) else 'N/A'} items")
    for idx, group_text in enumerate(group_texts[1:]):
        toc_with_page_number_additional = await generate_toc_continue(toc_with_page_number, group_text, model)
        toc_with_page_number.extend(toc_with_page_number_additional)
        log.info(f"[process_no_toc] After group {idx+2}/{len(group_texts)}: +{len(toc_with_page_number_additional)} items, total={len(toc_with_page_number)}")
    log.info(f'generate_toc: {toc_with_page_number}')

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    none_count = sum(1 for item in toc_with_page_number if item.get('physical_index') is None)
    log.info(f"[process_no_toc] After int conversion: {len(toc_with_page_number)} items, {none_count} with None physical_index")

    return toc_with_page_number

async def process_toc_no_page_numbers(toc_content, toc_page_list, page_list,  start_index=1, model=None, _logger=None):
    log = _logger or logger
    log.info(f"[process_toc_no_page_numbers] {len(page_list)} pages, start_index={start_index}")
    page_contents=[]
    token_lengths=[]
    toc_content = await toc_transformer(toc_content, model)
    log.info(f"[process_toc_no_page_numbers] Transformed ToC: {len(toc_content)} items")
    for page_index in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))

    group_texts = page_list_to_group_text(page_contents, token_lengths)
    log.info(f"[process_toc_no_page_numbers] Split into {len(group_texts)} text groups for page-number assignment")

    toc_with_page_number=copy.deepcopy(toc_content)
    for group_text in group_texts:
        toc_with_page_number = await add_page_number_to_toc(group_text, toc_with_page_number, model)
    log.info(f'add_page_number_to_toc: {toc_with_page_number}')

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    log.info(f'convert_physical_index_to_int: {toc_with_page_number}')

    return toc_with_page_number



async def process_toc_with_page_numbers(toc_content, toc_page_list, page_list, toc_check_page_num=None, model=None, _logger=None):
    log = _logger or logger
    log.info(f"[process_toc_with_page_numbers] {len(page_list)} pages, {len(toc_page_list)} ToC pages")
    toc_with_page_number = await toc_transformer(toc_content, model)
    log.info(f'toc_with_page_number: {toc_with_page_number}')

    toc_no_page_number = remove_page_number(copy.deepcopy(toc_with_page_number))

    start_page_index = toc_page_list[-1] + 1
    main_content = ""
    for page_index in range(start_page_index, min(start_page_index + toc_check_page_num, len(page_list))):
        main_content += f"<physical_index_{page_index+1}>\n{page_list[page_index][0]}\n<physical_index_{page_index+1}>\n\n"

    toc_with_physical_index = await toc_index_extractor(toc_no_page_number, main_content, model)
    log.info(f'toc_with_physical_index: {toc_with_physical_index}')

    toc_with_physical_index = convert_physical_index_to_int(toc_with_physical_index)
    log.info(f'toc_with_physical_index: {toc_with_physical_index}')

    matching_pairs = extract_matching_page_pairs(toc_with_page_number, toc_with_physical_index, start_page_index)
    log.info(f'matching_pairs: {matching_pairs}')

    offset = calculate_page_offset(matching_pairs)
    log.info(f'offset: {offset}')

    toc_with_page_number = add_page_offset_to_toc_json(toc_with_page_number, offset)
    log.info(f'toc_with_page_number: {toc_with_page_number}')

    toc_with_page_number = await process_none_page_numbers(toc_with_page_number, page_list, model=model)
    log.info(f'toc_with_page_number: {toc_with_page_number}')

    return toc_with_page_number



##check if needed to process none page numbers
async def process_none_page_numbers(toc_items, page_list, start_index=1, model=None):
    logger.info(f"[process_none_page_numbers] Filling missing physical_index for {sum(1 for item in toc_items if 'physical_index' not in item)}/{len(toc_items)} items")
    for i, item in enumerate(toc_items):
        if "physical_index" not in item:
            # Find previous physical_index
            prev_physical_index = 0  # Default if no previous item exists
            for j in range(i - 1, -1, -1):
                if toc_items[j].get('physical_index') is not None:
                    prev_physical_index = toc_items[j]['physical_index']
                    break

            # Find next physical_index
            next_physical_index = -1  # Default if no next item exists
            for j in range(i + 1, len(toc_items)):
                if toc_items[j].get('physical_index') is not None:
                    next_physical_index = toc_items[j]['physical_index']
                    break

            page_contents = []
            for page_index in range(prev_physical_index, next_physical_index+1):
                # Add bounds checking to prevent IndexError
                list_index = page_index - start_index
                if list_index >= 0 and list_index < len(page_list):
                    page_text = f"<physical_index_{page_index}>\n{page_list[list_index][0]}\n<physical_index_{page_index}>\n\n"
                    page_contents.append(page_text)
                else:
                    continue

            item_copy = copy.deepcopy(item)
            item_copy.pop('page', None)
            result = await add_page_number_to_toc(page_contents, item_copy, model)
            if result and isinstance(result, list) and len(result) > 0:
                phys_idx = result[0].get('physical_index')
                if isinstance(phys_idx, str) and phys_idx.startswith('<physical_index'):
                    item['physical_index'] = int(phys_idx.split('_')[-1].rstrip('>').strip())
                    item.pop('page', None)

    still_missing = sum(1 for item in toc_items if item.get('physical_index') is None)
    logger.info(f"[process_none_page_numbers] Done. Still missing: {still_missing}/{len(toc_items)}")
    return toc_items




async def check_toc(page_list, opt=None):
    toc_page_list = await find_toc_pages(start_page_index=0, page_list=page_list, opt=opt)
    if len(toc_page_list) == 0:
        logger.info('No ToC found')
        return {'toc_content': None, 'toc_page_list': [], 'page_index_given_in_toc': 'no'}
    else:
        logger.info('ToC found')
        toc_json = await toc_extractor(page_list, toc_page_list, opt.model)

        if toc_json['page_index_given_in_toc'] == 'yes':
            logger.info('Page index found in ToC')
            return {'toc_content': toc_json['toc_content'], 'toc_page_list': toc_page_list, 'page_index_given_in_toc': 'yes'}
        else:
            current_start_index = toc_page_list[-1] + 1

            while (toc_json['page_index_given_in_toc'] == 'no' and
                   current_start_index < len(page_list) and
                   current_start_index < opt.toc_check_page_num):

                additional_toc_pages = await find_toc_pages(
                    start_page_index=current_start_index,
                    page_list=page_list,
                    opt=opt
                )

                if len(additional_toc_pages) == 0:
                    break

                additional_toc_json = await toc_extractor(page_list, additional_toc_pages, opt.model)
                logger.info(f"[check_toc] Retry scan from page {current_start_index}: "
                            f"found {len(additional_toc_pages)} additional ToC pages, "
                            f"page_index_given={additional_toc_json.get('page_index_given_in_toc', 'no')}")
                if additional_toc_json['page_index_given_in_toc'] == 'yes':
                    logger.info('Page index found in ToC')
                    return {'toc_content': additional_toc_json['toc_content'], 'toc_page_list': additional_toc_pages, 'page_index_given_in_toc': 'yes'}

                else:
                    current_start_index = additional_toc_pages[-1] + 1
            logger.info('Page index not found in ToC')
            return {'toc_content': toc_json['toc_content'], 'toc_page_list': toc_page_list, 'page_index_given_in_toc': 'no'}






################### fix incorrect toc #########################################################
async def single_toc_item_index_fixer(section_title, content, model=None):
    model = model or DEFAULT_MODEL
    tob_extractor_prompt = """
    You are given a section title and several pages of a document, your job is to find the physical index of the start page of the section in the partial document.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    Reply in a JSON format:
    {
        "thinking": <explain which page, started and closed by <physical_index_X>, contains the start of this section>,
        "physical_index": "<physical_index_X>" (keep the format)
    }
    Directly return the final JSON structure. Do not output anything else."""

    prompt = tob_extractor_prompt + '\nSection Title:\n' + str(section_title) + '\nDocument pages:\n' + content
    json_content = await _llm_json(model=model, prompt=prompt)
    physical_index = json_content.get('physical_index')
    if physical_index is None:
        return None
    return convert_physical_index_to_int(physical_index)



async def fix_incorrect_toc(toc_with_page_number, page_list, incorrect_results, start_index=1, model=None, _logger=None):
    log = _logger or logger
    log.info(f'Fixing incorrect ToC with {len(incorrect_results)} incorrect results')
    incorrect_indices = {result['list_index'] for result in incorrect_results}

    end_index = len(page_list) + start_index - 1

    incorrect_results_and_range_logs = []
    # Helper function to process and check a single incorrect item
    async def process_and_check_item(incorrect_item):
        toc_idx = incorrect_item['list_index']

        # Check if toc_idx is valid
        if toc_idx < 0 or toc_idx >= len(toc_with_page_number):
            # Return an invalid result for out-of-bounds indices
            return {
                'list_index': toc_idx,
                'title': incorrect_item['title'],
                'physical_index': incorrect_item.get('physical_index'),
                'is_valid': False
            }

        # Find the previous correct item
        prev_correct = None
        for i in range(toc_idx-1, -1, -1):
            if i not in incorrect_indices and i >= 0 and i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get('physical_index')
                if physical_index is not None:
                    prev_correct = physical_index
                    break
        # If no previous correct item found, use start_index
        if prev_correct is None:
            prev_correct = start_index - 1

        # Find the next correct item
        next_correct = None
        for i in range(toc_idx+1, len(toc_with_page_number)):
            if i not in incorrect_indices and i >= 0 and i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get('physical_index')
                if physical_index is not None:
                    next_correct = physical_index
                    break
        # If no next correct item found, use end_index
        if next_correct is None:
            next_correct = end_index

        incorrect_results_and_range_logs.append({
            'list_index': toc_idx,
            'title': incorrect_item['title'],
            'prev_correct': prev_correct,
            'next_correct': next_correct
        })
        log.info(f"[fix_incorrect_toc] Fixing '{incorrect_item['title']}' (idx={toc_idx}): "
                 f"search range pages [{prev_correct}..{next_correct}]")

        page_contents = []
        for page_index in range(prev_correct, next_correct+1):
            page_list_idx = page_index - start_index
            if 0 <= page_list_idx < len(page_list):
                page_text = f"<physical_index_{page_index}>\n{page_list[page_list_idx][0]}\n<physical_index_{page_index}>\n\n"
                page_contents.append(page_text)
        content_range = ''.join(page_contents)

        physical_index_int = await single_toc_item_index_fixer(incorrect_item['title'], content_range, model)
        log.info(f"[fix_incorrect_toc] LLM suggested page {physical_index_int}, verifying...")

        # Check if the result is correct
        check_item = incorrect_item.copy()
        check_item['physical_index'] = physical_index_int
        check_result = await check_title_appearance(check_item, page_list, start_index, model)

        return {
            'list_index': toc_idx,
            'title': incorrect_item['title'],
            'physical_index': physical_index_int,
            'is_valid': check_result['answer'] == 'yes'
        }

    # Process incorrect items concurrently
    tasks = [
        process_and_check_item(item)
        for item in incorrect_results
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(incorrect_results, results):
        if isinstance(result, Exception):
            log.error(f"Processing item {item} generated an exception: {result}")
            continue
    results = [result for result in results if not isinstance(result, Exception)]

    # Update the toc_with_page_number with the fixed indices and check for any invalid results
    invalid_results = []
    for result in results:
        if result['is_valid']:
            # Add bounds checking to prevent IndexError
            list_idx = result['list_index']
            if 0 <= list_idx < len(toc_with_page_number):
                toc_with_page_number[list_idx]['physical_index'] = result['physical_index']
            else:
                # Index is out of bounds, treat as invalid
                invalid_results.append({
                    'list_index': result['list_index'],
                    'title': result['title'],
                    'physical_index': result['physical_index'],
                })
        else:
            invalid_results.append({
                'list_index': result['list_index'],
                'title': result['title'],
                'physical_index': result['physical_index'],
            })

    log.info(f'incorrect_results_and_range_logs: {incorrect_results_and_range_logs}')
    log.info(f'invalid_results: {invalid_results}')

    return toc_with_page_number, invalid_results



async def fix_incorrect_toc_with_retries(toc_with_page_number, page_list, incorrect_results, start_index=1, max_attempts=3, model=None, _logger=None):
    log = _logger or logger
    log.info(f"[fix_retries] Starting with {len(incorrect_results)} incorrect items, max_attempts={max_attempts}")
    fix_attempt = 0
    current_toc = toc_with_page_number
    current_incorrect = incorrect_results

    while current_incorrect:
        log.info(f"Fixing {len(current_incorrect)} incorrect results")

        current_toc, current_incorrect = await fix_incorrect_toc(current_toc, page_list, current_incorrect, start_index, model, _logger=_logger)

        fix_attempt += 1
        log.info(f"[fix_retries] Attempt {fix_attempt}: {len(current_incorrect)} still incorrect")
        if fix_attempt >= max_attempts:
            log.info("Maximum fix attempts reached")
            break

    log.info(f"[fix_retries] Finished after {fix_attempt} attempts, {len(current_incorrect)} items remaining incorrect")
    return current_toc, current_incorrect


async def _fix_tree_ranges(
    tree_nodes, range_end, page_list,
    page_list_base_index=1, model=None, _logger=None
):
    """Fix text ranges using hierarchical parent-child bounds.

    Invariants enforced:
    - siblings sorted by start_index
    - start_index monotonic non-decreasing (allow equal pages)
    - each child.start_index within [parent_start .. parent_end]
    - recompute end_index from next sibling
    """
    log = _logger or logger
    if not tree_nodes:
        return

    # 0) Sort siblings by start_index (stable)
    tree_nodes.sort(key=lambda n: (n.get("start_index", 10**12), n.get("title", "")))

    # 1) Compute end_index for siblings at this level
    for i, node in enumerate(tree_nodes):
        if i < len(tree_nodes) - 1:
            node["end_index"] = tree_nodes[i + 1]["start_index"]
        else:
            node["end_index"] = range_end

    # 2) Validate/fix children under each node
    for node in tree_nodes:
        children = node.get("nodes") or []
        if not children:
            continue

        parent_start = node.get("start_index", page_list_base_index)
        parent_end = node["end_index"]

        # Sort children (LLM can output unsorted)
        children.sort(key=lambda n: (n.get("start_index", 10**12), n.get("title", "")))

        needs_recompute = False

        for i, child in enumerate(children):
            # Normalize missing start_index
            if child.get("start_index") is None:
                child["start_index"] = parent_start
                needs_recompute = True

            # Ensure monotonic sibling order (allow equal page)
            if i > 0 and child["start_index"] < children[i - 1]["start_index"]:
                log.info(
                    f"[_fix_tree_ranges] Non-monotonic child '{child.get('title','')}' "
                    f"{child['start_index']} < prev {children[i-1]['start_index']} -> clamp"
                )
                child["start_index"] = children[i - 1]["start_index"]
                needs_recompute = True

            # Enforce child within parent upper bound
            if child["start_index"] > parent_end:
                prev_start = children[i - 1]["start_index"] if i > 0 else parent_start
                fixed = await _try_constrained_llm_fix(
                    title=child.get("title", ""),
                    search_start=prev_start,
                    search_end=parent_end,
                    page_list=page_list,
                    page_list_base_index=page_list_base_index,
                    model=model,
                    _logger=log
                )
                if fixed is not None:
                    log.info(
                        f"[_fix_tree_ranges] Out-of-range '{child.get('title','')}' "
                        f"{child['start_index']} > {parent_end} -> LLM {fixed}"
                    )
                    child["start_index"] = fixed
                else:
                    log.info(
                        f"[_fix_tree_ranges] Out-of-range '{child.get('title','')}' "
                        f"{child['start_index']} > {parent_end} -> clamp {prev_start}"
                    )
                    child["start_index"] = prev_start
                needs_recompute = True

            # Enforce child not before parent start
            if child["start_index"] < parent_start:
                log.info(
                    f"[_fix_tree_ranges] Child '{child.get('title','')}' "
                    f"start {child['start_index']} < parent_start {parent_start} -> clamp"
                )
                child["start_index"] = parent_start
                needs_recompute = True

        # Recompute children's end_index
        if needs_recompute:
            children.sort(key=lambda n: (n.get("start_index", 10**12), n.get("title", "")))
            for i, child in enumerate(children):
                child["end_index"] = (
                    children[i + 1]["start_index"] if i < len(children) - 1 else parent_end
                )

        # Recurse
        await _fix_tree_ranges(
            children, parent_end, page_list,
            page_list_base_index=page_list_base_index,
            model=model, _logger=log
        )


async def _try_constrained_llm_fix(
    title, search_start, search_end, page_list,
    page_list_base_index=1, model=None, _logger=None
):
    log = _logger or logger
    if search_start > search_end:
        return None
    page_contents = []
    for page_idx in range(search_start, search_end + 1):
        list_idx = page_idx - page_list_base_index
        if 0 <= list_idx < len(page_list):
            page_contents.append(
                f"<physical_index_{page_idx}>\n{page_list[list_idx][0]}\n<physical_index_{page_idx}>\n\n"
            )
    if not page_contents:
        return None
    content = "".join(page_contents)
    result = await single_toc_item_index_fixer(title, content, model)
    if result is not None and search_start <= result <= search_end:
        log.info(f"[_try_constrained_llm_fix] '{title}' -> {result} in [{search_start},{search_end}]")
        return result
    log.info(f"[_try_constrained_llm_fix] '{title}' not found in [{search_start},{search_end}]")
    return None


################### verify toc #########################################################
async def verify_toc(page_list, list_result, start_index=1, N=None, model=None):
    logger.info(f"[verify_toc] {len(list_result)} items, last_physical_index=pending, page_count={len(page_list)}")
    # Find the last non-None physical_index
    last_physical_index = None
    for item in reversed(list_result):
        if item.get('physical_index') is not None:
            last_physical_index = item['physical_index']
            break

    # Early return if we don't have valid physical indices
    if last_physical_index is None or last_physical_index < len(page_list)/2:
        logger.info(f"[verify_toc] Early return: last_physical_index={last_physical_index} < {len(page_list)}/2")
        return 0, []

    # Determine which items to check
    if N is None:
        logger.info('Checking all items')
        sample_indices = range(0, len(list_result))
    else:
        N = min(N, len(list_result))
        logger.info(f'Checking {N} items')
        sample_indices = random.sample(range(0, len(list_result)), N)

    # Prepare items with their list indices
    indexed_sample_list = []
    for idx in sample_indices:
        item = list_result[idx]
        # Skip items with None physical_index (these were invalidated by validate_and_truncate_physical_indices)
        if item.get('physical_index') is not None:
            item_with_index = item.copy()
            item_with_index['list_index'] = idx  # Add the original index in list_result
            indexed_sample_list.append(item_with_index)

    # Run checks concurrently
    tasks = [
        check_title_appearance(item, page_list, start_index, model)
        for item in indexed_sample_list
    ]
    results = await asyncio.gather(*tasks)

    # Process results
    correct_count = 0
    incorrect_results = []
    for result in results:
        if result['answer'] == 'yes':
            correct_count += 1
        else:
            incorrect_results.append(result)

    # Calculate accuracy
    checked_count = len(results)
    accuracy = correct_count / checked_count if checked_count > 0 else 0
    logger.info(f"[verify_toc] Checked {checked_count} items: {correct_count} correct, {len(incorrect_results)} incorrect ({accuracy*100:.2f}%)")
    return accuracy, incorrect_results





################### main process #########################################################
async def meta_processor(page_list, mode=None, toc_content=None, toc_page_list=None, start_index=1, opt=None, _logger=None):
    log = _logger or logger
    log.info(f"[meta_processor] mode={mode}, pages={len(page_list)}, start_index={start_index}")

    if mode == 'process_toc_with_page_numbers':
        toc_with_page_number = await process_toc_with_page_numbers(toc_content, toc_page_list, page_list, toc_check_page_num=opt.toc_check_page_num, model=opt.model, _logger=_logger)
    elif mode == 'process_toc_no_page_numbers':
        toc_with_page_number = await process_toc_no_page_numbers(toc_content, toc_page_list, page_list, model=opt.model, _logger=_logger)
    else:
        toc_with_page_number = await process_no_toc(page_list, start_index=start_index, model=opt.model, _logger=_logger)

    toc_with_page_number = [item for item in toc_with_page_number if item.get('physical_index') is not None]
    log.info(f"[meta_processor] Extracted {len(toc_with_page_number)} items (after None filter)")

    toc_with_page_number = validate_and_truncate_physical_indices(
        toc_with_page_number,
        len(page_list),
        start_index=start_index,
        _logger=_logger
    )
    log.info(f"[meta_processor] After validation: {len(toc_with_page_number)} items remain")

    accuracy, incorrect_results = await verify_toc(page_list, toc_with_page_number, start_index=start_index, model=opt.model)

    log.info(f"[meta_processor] Verification: accuracy={accuracy:.1%}, incorrect={len(incorrect_results)}, "
             f"decision={'accept' if accuracy == 1.0 else 'fix' if accuracy > 0.6 else 'fallback'}")
    log.info({
        'mode': mode,
        'accuracy': accuracy,
        'incorrect_results': incorrect_results
    })
    if accuracy == 1.0 and len(incorrect_results) == 0:
        return toc_with_page_number
    if accuracy > 0.6 and len(incorrect_results) > 0:
        toc_with_page_number, incorrect_results = await fix_incorrect_toc_with_retries(toc_with_page_number, page_list, incorrect_results,start_index=start_index, max_attempts=3, model=opt.model, _logger=_logger)
        return toc_with_page_number
    else:
        if mode == 'process_toc_with_page_numbers':
            return await meta_processor(page_list, mode='process_toc_no_page_numbers', toc_content=toc_content, toc_page_list=toc_page_list, start_index=start_index, opt=opt, _logger=_logger)
        elif mode == 'process_toc_no_page_numbers':
            return await meta_processor(page_list, mode='process_no_toc', start_index=start_index, opt=opt, _logger=_logger)
        else:
            raise Exception('Processing failed')


async def process_large_node_recursively(node, page_list, opt=None, _logger=None):
    log = _logger or logger
    node_page_list = page_list[node['start_index']-1:node['end_index']]
    token_num = sum([page[1] for page in node_page_list])

    if node['end_index'] - node['start_index'] > opt.max_page_num_each_node and token_num >= opt.max_token_num_each_node:
        log.info(f'Large node: {node["title"]}, start_index: {node["start_index"]}, end_index: {node["end_index"]}, token_num: {token_num}')

        node_toc_tree = await meta_processor(node_page_list, mode='process_no_toc', start_index=node['start_index'], opt=opt, _logger=_logger)
        node_toc_tree = await check_title_appearance_in_start_concurrent(
            node_toc_tree, page_list, model=opt.model, _logger=_logger)

        # Filter out items with None physical_index before post_processing
        valid_node_toc_items = [item for item in node_toc_tree if item.get('physical_index') is not None]

        if valid_node_toc_items and node['title'].strip() == valid_node_toc_items[0]['title'].strip():
            node['nodes'] = post_processing(valid_node_toc_items[1:], node['end_index'])
            await _fix_tree_ranges(
                node['nodes'], node['end_index'], node_page_list,
                page_list_base_index=node['start_index'],
                model=opt.model, _logger=_logger
            )
            node['end_index'] = (
                max(node['start_index'], node['nodes'][0]['start_index'] - 1)
                if node['nodes'] else node['end_index']
            )
        else:
            node['nodes'] = post_processing(valid_node_toc_items, node['end_index'])
            await _fix_tree_ranges(
                node['nodes'], node['end_index'], node_page_list,
                page_list_base_index=node['start_index'],
                model=opt.model, _logger=_logger
            )
            node['end_index'] = (
                max(node['start_index'], node['nodes'][0]['start_index'] - 1)
                if node['nodes'] else node['end_index']
            )

        log.info(f"[large_node] Split '{node['title']}' into {len(node.get('nodes', []))} children")
    else:
        log.info(f"[large_node] Skipping '{node['title']}': {node['end_index'] - node['start_index']} pages, {token_num} tokens (below thresholds)")

    if 'nodes' in node and node['nodes']:
        tasks = [
            process_large_node_recursively(child_node, page_list, opt, _logger=_logger)
            for child_node in node['nodes']
        ]
        await asyncio.gather(*tasks)

    return node

async def tree_parser(page_list, opt, doc=None, _logger=None):
    log = _logger or logger
    log.info(f"[tree_parser] Starting: {len(page_list)} pages")
    check_toc_result = await check_toc(page_list, opt)
    log.info(f"[tree_parser] ToC detection result: has_toc={bool(check_toc_result.get('toc_content'))}, "
             f"page_index_in_toc={check_toc_result.get('page_index_given_in_toc', 'n/a')}, "
             f"toc_pages={check_toc_result.get('toc_page_list', [])}")

    if check_toc_result.get("toc_content") and check_toc_result["toc_content"].strip() and check_toc_result["page_index_given_in_toc"] == "yes":
        toc_with_page_number = await meta_processor(
            page_list,
            mode='process_toc_with_page_numbers',
            start_index=1,
            toc_content=check_toc_result['toc_content'],
            toc_page_list=check_toc_result['toc_page_list'],
            opt=opt,
            _logger=_logger)
    elif check_toc_result.get("toc_content") and check_toc_result["toc_content"].strip():
        toc_with_page_number = await meta_processor(
            page_list,
            mode='process_toc_no_page_numbers',
            start_index=1,
            toc_content=check_toc_result['toc_content'],
            toc_page_list=check_toc_result['toc_page_list'],
            opt=opt,
            _logger=_logger)
    else:
        toc_with_page_number = await meta_processor(
            page_list,
            mode='process_no_toc',
            start_index=1,
            opt=opt,
            _logger=_logger)

    log.info(f"[tree_parser] meta_processor returned {len(toc_with_page_number)} ToC items")
    toc_with_page_number = add_preface_if_needed(toc_with_page_number)
    toc_with_page_number = await check_title_appearance_in_start_concurrent(
        toc_with_page_number, page_list, model=opt.model, _logger=_logger)

    appear_yes = sum(1 for item in toc_with_page_number if item.get('appear_start') == 'yes')
    log.info(f"[tree_parser] appear_start check: {appear_yes}/{len(toc_with_page_number)} items start at page beginning")

    # Filter out items with None physical_index before post_processing
    valid_toc_items = [item for item in toc_with_page_number if item.get('physical_index') is not None]

    toc_tree = post_processing(valid_toc_items, len(page_list))

    def _count_nodes(nodes):
        c = len(nodes)
        for n in nodes:
            c += _count_nodes(n.get('nodes', []))
        return c

    log.info(f"[tree_parser] post_processing built tree with {_count_nodes(toc_tree)} total nodes, {len(toc_tree)} root nodes")

    # Fix out-of-range nodes using parent-range validation
    await _fix_tree_ranges(
        toc_tree, len(page_list), page_list,
        page_list_base_index=1, model=opt.model, _logger=_logger
    )
    log.info(f"[tree_parser] _fix_tree_ranges complete")

    tasks = [
        process_large_node_recursively(node, page_list, opt, _logger=_logger)
        for node in toc_tree
    ]
    await asyncio.gather(*tasks)

    log.info(f"[tree_parser] Large-node recursive splitting complete. Final tree: {_count_nodes(toc_tree)} nodes")
    return toc_tree


def page_index_main(doc, opt=None):
    _logger = JsonLogger(doc)

    is_valid_pdf = (
        (isinstance(doc, str) and os.path.isfile(doc) and doc.lower().endswith(".pdf")) or
        isinstance(doc, BytesIO)
    )
    if not is_valid_pdf:
        raise ValueError("Unsupported input type. Expected a PDF file path or BytesIO object.")

    logger.info('Parsing PDF...')
    page_list = get_page_tokens(doc)

    _logger.info({'total_page_number': len(page_list)})
    _logger.info({'total_token': sum([page[1] for page in page_list])})

    async def page_index_builder():
        structure = await tree_parser(page_list, opt, doc=doc, _logger=_logger)
        if opt.if_add_node_id == 'yes':
            write_node_id(structure)
        if opt.if_add_node_text == 'yes':
            add_node_text(structure, page_list)
        if opt.if_add_node_summary == 'yes':
            if opt.if_add_node_text == 'no':
                add_node_text(structure, page_list)
            await generate_summaries_for_structure(structure, model=opt.model)
            if opt.if_add_node_text == 'no':
                remove_structure_text(structure)
            if opt.if_add_doc_description == 'yes':
                # Create a clean structure without unnecessary fields for description generation
                clean_structure = create_clean_structure_for_description(structure)
                doc_description = await generate_doc_description(clean_structure, model=opt.model)
                return {
                    'doc_name': get_pdf_name(doc),
                    'doc_description': doc_description,
                    'structure': structure,
                }
        return {
            'doc_name': get_pdf_name(doc),
            'structure': structure,
        }

    return asyncio.run(page_index_builder())


def page_index(doc, model=None, toc_check_page_num=None, max_page_num_each_node=None, max_token_num_each_node=None,
               if_add_node_id=None, if_add_node_summary=None, if_add_doc_description=None, if_add_node_text=None):

    user_opt = {
        arg: value for arg, value in locals().items()
        if arg != "doc" and value is not None
    }
    opt = ConfigLoader().load(user_opt)
    return page_index_main(doc, opt)


def validate_and_truncate_physical_indices(toc_with_page_number, page_list_length, start_index=1, _logger=None):
    """
    Validates and truncates physical indices that exceed the actual document length.
    This prevents errors when TOC references pages that don't exist in the document (e.g. the file is broken or incomplete).
    """
    log = _logger or logger
    if not toc_with_page_number:
        return toc_with_page_number

    max_allowed_page = page_list_length + start_index - 1
    truncated_items = []

    for i, item in enumerate(toc_with_page_number):
        if item.get('physical_index') is not None:
            original_index = item['physical_index']
            if original_index > max_allowed_page:
                item['physical_index'] = None
                truncated_items.append({
                    'title': item.get('title', 'Unknown'),
                    'original_index': original_index
                })
                log.info(f"Removed physical_index for '{item.get('title', 'Unknown')}' (was {original_index}, too far beyond document)")

    if truncated_items:
        log.info(f"Total removed items: {len(truncated_items)}")

    log.info(f"Document validation: {page_list_length} pages, max allowed index: {max_allowed_page}")
    if truncated_items:
        log.info(f"Truncated {len(truncated_items)} TOC items that exceeded document length")

    return toc_with_page_number
