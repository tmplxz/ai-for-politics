import os
import re
import json
import tqdm

from PyPDF2 import PdfReader
import transformers
import torch
transformers.utils.logging.set_verbosity_error()


def load_translation_model(model_name="Unbabel/TowerInstruct-13B-v0.1"):
    pipe = transformers.pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")
    return pipe


def translate(model, input, sentence_batch_size=-1, source_lan='German', target_lan='English'):

    if isinstance(input, str):
        paragraphs = [input]
    else:
        if not isinstance(input, list) or not isinstance(input[0], str):
            raise TypeError('Input must be a string or a list of strings.')
        if sentence_batch_size < 0:
            paragraphs = input
        else: # create paragraphs of individual sentences
            paragraphs = []
            for i, s in enumerate(range(0, len(input), sentence_batch_size)):
                paragraphs.append(' '.join(input[s:(s+sentence_batch_size)]))
    
    translated = []
    for paragraph in tqdm.tqdm(paragraphs, desc=f'Translating {len(paragraphs)} paragraphs from {source_lan} to {target_lan}'):
        if len(paragraph.strip()) > 0:
            messages = [{"role": "user", "content": f"Translate the following sentences from {source_lan} into {target_lan}.\n{source_lan}: {paragraph}\n{target_lan}:"},]
            prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = model(prompt, max_new_tokens=256, do_sample=False)
            translated.append(outputs[0]["generated_text"].split('<|im_start|>assistant')[1].strip())
    return translated if len(translated) > 1 else translated[0]


def translate_pdf(model, fname, sentence_batch_size=10, break_after=None):
    reader = PdfReader(fname)
    full_text = []
    for p_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        lines = text.split('\n')
        for line in lines:
            if len(line) > 10 and not line.upper() == line:
                full_text.append(line.strip())
        if break_after and len(full_text) > break_after:
            break
    full_text = ' '.join(full_text)
    # Split the text into sentences using ., ?, or ! as delimiters
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    print(f'Number of sentences in {fname}: {len(sentences)}, to be processed in {len(sentences) // sentence_batch_size} batches')
    # translate single paragraphs
    translated = translate(model, sentences, sentence_batch_size=sentence_batch_size)

    # concat paragraphs, split sentences across lines, write into file
    translated = ' '.join(translated).replace('e.g.', 'for example')
    sentences = re.split(r'(?<=[.!?])\s+', translated)
    return '\n'.join(sentences)


def summarize_content(content, fname, sentence_batch_size=20, summarization_rate=0.25, model_id="facebook/bart-large-cnn"):
    sentences = content.split('\n')
    summarizer = transformers.pipeline("summarization", model=model_id)
    summary_sentences = []
    for i, s in enumerate(range(0, len(sentences), sentence_batch_size)):
        paragraph = ' '.join(sentences[s:(s+sentence_batch_size)])
        maxl, minl = int(len(paragraph) * summarization_rate / 4), int(len(paragraph) * summarization_rate / 8)
        summary = summarizer(paragraph, max_length=maxl, min_length=minl, do_sample=False)
        summary = summary[0]['summary_text']
        summary_splitted = re.split(r'(?<=[.!?])\s+', summary)
        for sentence in summary_splitted:
            summary_sentences.append(sentence.strip())
        print(f'Summarized sentence batch {i+1} / {len(sentences)//sentence_batch_size+1}, summary has {len(summary)} characters instead of {len(paragraph)} ({len(summary)/len(paragraph)*100:4.2f}% of original size)')
    return '\n'.join(summary_sentences)


def reason(model, tokenizer, prompt, max_new_tokens=5000, end_of_reasoning_token=151668):
    # construct query
    messages = [ {"role": "user", "content": prompt} ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # run and parse output
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(end_of_reasoning_token) # rindex finding 151668 (</think>)
    except ValueError:
        index = 0
    reasoning = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    answer = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return reasoning, answer


def load_llm(model_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    return (model, tokenizer)


def reason_about_visual_points(llm, input, type_of_input='election program summary', n_points=5,
                               prompt='Identify {p} important visual aspects of a city appearance that would be affected or impacted by this political program. Describe each aspect in an informative and concise way, with 3 to 6 words. Return these {p} visual descriptions as a comma-separated list.'):
    model, tokenizer = llm
    full_prompt = f'Analyze the following {type_of_input}. {prompt.format(p=n_points)}\n\n{input}'
    reasoning, impact_points = reason(model, tokenizer, full_prompt)
    return impact_points, reasoning


def reason_about_impact_points(summary, fname, model_name="Qwen/Qwen3-30B-A3B"):
    point_fname, reasoning_fname = fname.replace('.pdf', '_impact_points.json'), fname.replace('.pdf', '_impact_point_reasoning.txt'), 
    if os.path.isfile(point_fname):
        with open(point_fname, 'r') as f:
            impact_points = json.load(f)
        with open(reasoning_fname, 'r') as f:
            reasoning = f.read()
        print_str = f'Loading pre-compiled impact points from {point_fname}'
    else:
        # init model
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        # run model
        baseprompt = 'From the following election program summary, identify ten central "impact points" relating to specific local aspects that would be affected or changed if the program is coming into effect. Find meaningful identifier strings that summarize each point and also formulate a short description for each of them. Assign an importance weight betweeon 0 and 1 to each impact point, based on how pronounced and rich the point is discussed in the program, and formulate a short respective explanation. Return the results as a JSON-formatted string (list of impact points with "identifier", "description", "importance" and "importance_reasoning"). Make sure to not use quotation marks or other JSON-syntax symbols for the descriptions and reasoning.'
        reasoning, impact_points = reason(model, tokenizer, baseprompt + '\n\n' + summary)
        # write results
        with open(point_fname, 'w') as f:
            f.write(impact_points)
        with open(reasoning_fname, 'w') as f:
            f.write(reasoning)

        # check for json compatibility
        file_ok, iter = False, 5
        while not file_ok and iter > 0:
            try: # check if json is readble
                with open(point_fname, 'r') as f:
                    fstring = f.read()
                impact_points = json.loads(fstring)
                if ('impact_points' not in impact_points) or \
                        (not isinstance(impact_points['impact_points'], list)) or \
                        ('identifier' not in impact_points['impact_points'][0]) or \
                        ('description' not in impact_points['impact_points'][0]) or \
                        ('importance' not in impact_points['impact_points'][0]) or \
                        ('importance_reasoning' not in impact_points['impact_points'][0]):
                    raise RuntimeError
                file_ok = True
            except json.JSONDecodeError:
                prompt = 'The following JSON content is malformed and raises a JSONDecodeError, please fix it and only provide the correctly formatted json string:\n\n' + fstring
            except RuntimeError:
                prompt = 'The JSON content does not have the desired structure, on root level, it should contain the "impact_points" as a list, with each element having an "identifier" (str), "description" (str), "importance" (float between 0.0 - 1.0) and "importance_reasoning" (str). Please fix it and only provide the correctly formatted json string:\n\n' + fstring
            if not file_ok:
                # try to fix JSON with the reasoning model
                print(f'\n\nJSON ERROR! ITERATION {iter}\n\n')
                json_reasoning, fixed_fstring = reason(model, tokenizer, prompt)
                print(json_reasoning)
                with open(point_fname, 'w') as f:
                    f.write(fixed_fstring)
                iter -= 1
                
        # finalize
        print_str = f'Compiled impact points into {point_fname}'
        if not file_ok:
            print_str = print_str + ', but with remaining json errors'
            impact_points = fstring

    print('\n', print_str, '\n', "reasoning content:", reasoning, '\n', str(impact_points))
    return impact_points, reasoning

def impact_point_comparison_analysis(dirname, model_name="Qwen/Qwen3-30B-A3B"):
    fname_analysis, fname_analysis_reasons = os.path.join(dirname, 'impact_analysis.txt'), os.path.join(dirname, 'impact_analysis_reasoning.txt')
    if os.path.isfile(fname_analysis):
        with open(fname_analysis, 'r') as f:
            analysis = f.read()
        with open(fname_analysis_reasons, 'r') as f:
            reasoning = f.read()
        print_str = f'Loading pre-compiled impact point analysis from {fname_analysis}'
    else:
        # load impact points from all parties
        impact_points = {}
        for root, _, files in os.walk(dirname):
            for f in files:
                if '_impact_points.json' in f:
                    if root in impact_points:
                        raise RuntimeError('Found two impact point files in ' + root)
                    try:
                        with open(os.path.join(root, f), 'r') as content:
                            impact_points[root] = json.load(content)['impact_points']
                    except:
                        raise RuntimeError('Could not load impact points from ' + os.path.join(root, f))
        points_txt = '\n'.join([f'Program {program_idx} Point {p_idx}: {p["identifier"]} - {p["description"]}' for program_idx, plist in enumerate(impact_points.values()) for p_idx, p in enumerate(plist)])
        print('Analyzing Impact Points:\n' + points_txt)
        # init model
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        # analyze the impact points
        prompt = 'Analyze the following impact points obtained from different political programs. Investigate common points and highlight notable differences in the programs. Think about how the different political programs and impact points would affect the visual appearance of the region. For each program, identify five visual changes that stand out strongest in comparison to the other programs, summarized as short keypoints. The final answer should consist of a summary of the analysis in the first textline, followed by individual textlines for each program, with comma-seperated keypoints summarizing the most crucial visual changes. Accordingly, the response format should be "Summary: [summary of your toughts] \nProgram 1: [keypoints for visual changes] \nProgram 2: [keypoints for visual changes] \nProgram 3: [keypoints for visual changes]")'
        reasoning, analysis = reason(model, tokenizer, prompt + '\n' + points_txt)
        with open(fname_analysis, 'w') as f:
            f.write(analysis)
        with open(fname_analysis_reasons, 'w') as f:
            f.write(reasoning)
        # check for output compatibility
        file_ok, iter = False, 5
        while not file_ok and iter > 0:
            try: # check if json is readble
                per_program = analysis.split('\n')[1:]
                assert len(per_program) == len(impact_points)
                for ana, root in zip(per_program, impact_points.keys()):
                    with open(os.path.join(root, 'prompts', 'visual_impact_points.txt'), 'w') as f:
                        f.write(re.sub(r'Program \d: ', '', ana))
                file_ok = True
            except:
                prompt = 'The formatting is not correct, the response format should be "Summary: ... \nProgram 1: ... \nProgram 2: ... \nProgram 3: ...") Please fix it:\n\n' + analysis
            if not file_ok:
                # try to fix JSON with the reasoning model
                print(f'\n\nJSON ERROR! ITERATION {iter}\n\n')
                fix_reasoning, fixed_analysis = reason(model, tokenizer, prompt)
                print(fix_reasoning)
                with open(fname_analysis, 'w') as f:
                    f.write(fixed_analysis)
                iter -= 1
        print_str = f'Compiled impact point analysis into {fname_analysis}'
        if not file_ok:
            print_str.append(', but could not extract the individual program prompts!')

    print(print_str)
    return analysis

def load_vlm(model_name: str = "Qwen/Qwen2-VL-7B-Instruct") -> tuple[transformers.Qwen2VLForConditionalGeneration, transformers.AutoProcessor]:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return (model, processor)

# Global cache for the embedding model to avoid reloading it for every evaluation
_embedding_model_cache = None

def get_embedding_model():
    global _embedding_model_cache
    from sentence_transformers import SentenceTransformer
    if _embedding_model_cache is None:
        _embedding_model_cache = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model_cache

def describe_image_vlm(model: transformers.Qwen2VLForConditionalGeneration, processor: transformers.AutoProcessor, image_paths: list[str], n_points: int = 5) -> list[str]:
    from PIL import Image
    
    descriptions = []
    
    # Process each image individually
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        content = [{"type": "image", "image": image}]
        content.append({"type": "text", "text": f"Analyze this image of a city. Identify {n_points} key urban planning or policy-related visual aspects visible in this image. Focus on infrastructure, public spaces, transportation systems, environmental features, and architectural elements. Describe each aspect using a descriptive phrase of at least 2 words (e.g., adjective + noun). Do not use single words. Return ONLY a comma-separated list ending with a period, nothing else."})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        descriptions.append(output_text[0])
        
    return descriptions

def evaluate_similarity(reference: str, candidate: str) -> dict:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Tokenize for BLEU (split by comma and clean up)
    ref_tokens = [token.strip().lower() for token in reference.replace('.', '').split(',')]
    cand_tokens = [token.strip().lower() for token in candidate.replace('.', '').split(',')]
    
    # BLEU score with smoothing (handles short sequences)
    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
    
    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference.lower(), candidate.lower())
    
    # Cosine similarity using sentence embeddings
    embedding_model = get_embedding_model()
    ref_embedding = embedding_model.encode([reference])
    cand_embedding = embedding_model.encode([candidate])
    cos_sim = cosine_similarity(ref_embedding, cand_embedding)[0][0]
    
    return {
        'bleu': round(bleu_score, 4),
        'rouge1_f': round(rouge_scores['rouge1'].fmeasure, 4),
        'rouge2_f': round(rouge_scores['rouge2'].fmeasure, 4),
        'rougeL_f': round(rouge_scores['rougeL'].fmeasure, 4),
        'cosine_similarity': round(float(cos_sim), 4)
    }


def evaluate_alignment_with_llm(llm, reference: str, candidate: str) -> dict:
    model, tokenizer = llm
    
    judge_prompt = f"""You are an impartial evaluator assessing how well an AI-generated image represents intended policy impacts.

    **Intended Policy Impacts (what the image should show):**
    {reference}

    **Observed Image Description (what the AI actually generated):**
    {candidate}

    **Your Task:**
    1. Analyze how well the observed image description captures the key elements from the intended policy impacts.
    2. Consider semantic similarity, not just exact word matches. For example, "bike lanes" and "cycling infrastructure" should be considered equivalent.
    3. Rate the alignment on a scale of 1-10:
    - 1-3: Poor alignment - Most key elements are missing or misrepresented
    - 4-6: Moderate alignment - Some key elements are captured, but significant aspects are missing
    - 7-9: Good alignment - Most key elements are well represented with minor omissions
    - 10: Excellent alignment - All key elements are accurately represented

    **Respond with ONLY a JSON object in this exact format (no other text):**
    {{"score": <integer 1-10>, "reasoning": "<brief explanation in 1-2 sentences>"}}
    
    """

    _, answer = reason(model, tokenizer, judge_prompt, max_new_tokens=2048)
    
    # Parse the JSON response
    try:
        # Try to extract JSON from the answer
        import re
        json_match = re.search(r'\{[^}]+\}', answer)
        if json_match:
            result = json.loads(json_match.group())
            score = int(result.get('score', 5))
            reasoning = result.get('reasoning', 'No reasoning provided')
        else:
            # Fallback: try to parse the whole answer
            result = json.loads(answer)
            score = int(result.get('score', 5))
            reasoning = result.get('reasoning', 'No reasoning provided')
    except (json.JSONDecodeError, ValueError, AttributeError):
        # If parsing fails, try to extract score from text
        score = 5  # fallback score
        reasoning = f"Failed to parse LLM response: {answer[:200]}"
        # Try to find a number in the response
        numbers = re.findall(r'\b([1-9]|10)\b', answer)
        if numbers:
            score = int(numbers[0])
    
    return {
        'llm_score': score,
        'llm_reasoning': reasoning
    }
    