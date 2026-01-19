import os
import argparse
import shutil
import glob
from datetime import datetime

import pandas as pd
import numpy as np
from codecarbon import OfflineEmissionsTracker

from llm import load_llm, load_translation_model, translate, translate_pdf, reason_about_visual_points, summarize_content, load_vlm, describe_image_vlm, evaluate_similarity, evaluate_alignment_with_llm
from images import load_model_diffusers, generate_images_diffusers


def translate_kommunalomat(model, fname, output_dir, only_approved=True):
    translation_fname = os.path.join(output_dir, os.path.basename(fname.replace('.xlsx', '_translated.txt')))

    if os.path.isfile(translation_fname):
        print(f"Translation file {translation_fname} already exists")
        with open(translation_fname, 'r') as f:
            results = f.read()
    else:
        # Load the Excel file into a pandas DataFrame
        df = pd.read_excel(fname)
        question_cols = list(range(8, 122, 2))
        response_cols = list(np.array(question_cols) + 1)

        # Translate questions
        questions = df.columns[question_cols].map(lambda q: q.replace('.', '').strip()).to_list()
        translated = translate(model, questions)
        df = df.rename({q: qf for q, qf in zip(df.columns[question_cols], translated)}, axis=1)

        # Translate responses
        all_responses = []
        for col in response_cols:
            all_responses.extend(df.iloc[:, col].dropna().tolist())
        translated = translate(model, all_responses, source_lan='German', target_lan='English')
        idx = 0
        for col in response_cols:
            col_responses = df.iloc[:, col].dropna().index
            for row in col_responses:
                df.iat[row, col] = translated[idx]
                idx += 1

        # translate scores
        df = df.fillna('')
        df = df.replace('Starke Zustimmung', 'Strong Approval')
        df = df.replace('Zustimmung', 'Approval')
        df = df.replace('Ablehnung', 'Rejection')
        df = df.replace('Starke Ablehnung', 'Strong Rejection')

        # Extract and format the results for each party
        results = []
        for party_idx in range(df.shape[0]):
            party_results = []
            for question_idx in question_cols:
                question = df.columns[question_idx]
                score = df.iloc[party_idx, question_idx]
                reasoning = df.iloc[party_idx, question_idx+1].strip()
                if not only_approved or 'Approval' in score:
                    party_results.append(f'{question}:\n{score} - {reasoning}\n')
            results.append(f'Party {party_idx}\n\n' + '\n'.join(party_results))

        results = '\n\n\n'.join(results)
        with open(translation_fname, 'w') as f:
            f.write(results)

        print(f"Created translation file {translation_fname}")

    per_party = [party_response.split('\n\n')[1:] for party_response in results.split('\n\n\n')]
    # synch with party identifiers, manually read from excel
    per_party = {p: r for p, r in zip(['tierschutz', 'diepartei', 'spd', 'linke', 'cdu', 'gruene', 'bvt', 'fdp', 'bli', 'volt'], per_party)}
    for p, r in per_party.items():
        out_dir = os.path.join(output_dir, p)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "kommunalomat_responses_en.txt")
        with open(out_file, "w") as f:
            f.write('\n\n'.join(r))
        print(f"Wrote kommunalomat results for {p} to {out_file}")
    return per_party


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Iterative processing of party programs.")
    parser.add_argument("--mode", type=str, default='translate', choices=["translate", "summarize", "reason", "translate_results", "generate_images", "describe_image", "evaluate"], help="Processing mode")
    parser.add_argument("--input_dir", type=str, default="programs", help="Directory containing party program pdfs")
    parser.add_argument("--output_dir", type=str, default="src/frontend/public/political_content_2025", help="Output directory for results")
    parser.add_argument("--override", action='store_true', help="Override existing files")
    # reasoning
    parser.add_argument("--llm", type=str, default="Qwen/Qwen3-30B-A3B", help="Name of the LLM to use for reasoning")
    parser.add_argument("--n_points", type=int, default=5, help="Number of visual points to generate")
    # image generation
    parser.add_argument("--image_generator", type=str, default="black-forest-labs/FLUX.1-schnell", help="Model name or path for image generation")
    parser.add_argument("--guidance", type=float, default=0., help="Prompt guidance strength")
    parser.add_argument("--num_steps", type=int, default=5, help="Diffusion steps")
    parser.add_argument("--n_images", type=int, default=5, help="Number of images to create")
    # VLM describing
    parser.add_argument("--vlm", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Name of the vision language model for descriping images.")
    # evaluation
    parser.add_argument("--eval_method", type=str, default="embedding", choices=["embedding", "llm", "both"], help="Evaluation method: 'embedding' (BLEU/ROUGE/Cosine), 'llm' (LLM-as-Judge), or 'both'")
    parser.add_argument("--parties", type=str, nargs='+', default=None, help="Specific parties to evaluate (e.g., --parties spd cdu). If not specified, all parties are evaluated.")
    parser.add_argument("--sources", type=str, nargs='+', default=None, choices=["program", "kommunalomat"], help="Specific sources to evaluate (e.g., --sources program). If not specified, both are evaluated.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    party_dirs = [d for d in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, d))]

    # Use a timestamp to ensure emissions are saved in separate files and not overriding
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    emissions_dir = os.path.join(args.output_dir, "emissions")
    os.makedirs(emissions_dir, exist_ok=True)
    emissions_filename = f"emissions_{args.mode}_{timestamp}.csv"
    
    tracker = OfflineEmissionsTracker(
        log_level='error', 
        country_iso_code="DEU", 
        output_dir=emissions_dir, 
        output_file=emissions_filename,
        experiment_id=f"{args.mode}_{timestamp}"
    )
    tracker.start()

    if args.mode == "translate":
        print('TRANSLATING KOMMUNALOMAT')
        model = load_translation_model()

        # translate kommunalomat data
        komm_data = os.path.join(args.input_dir, "kommunalomat_data.xlsx")
        komm_translations = translate_kommunalomat(model, komm_data, args.output_dir)

        # translate pdfs
        print('TRANSLATING PROGRAMS')
        pdfs = glob.glob(os.path.join(args.input_dir, "*.pdf"))
        for pdf_path in pdfs:
            party = os.path.splitext(os.path.basename(pdf_path))[0]
            out_dir = os.path.join(args.output_dir, party)
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, "program_en.txt")
            if os.path.isfile(out_file):
                if args.override:
                    os.remove(out_file)
                else:
                    print(f"{out_file} already exists, skipping.")
                    continue
            print(f"Translating {pdf_path} -> {out_file}")
            translated = translate_pdf(model, pdf_path)
            with open(out_file, "w") as f:
                f.write(translated)

    elif args.mode == "summarize":
        print('SUMMARIZING PROGRAMS')
        for party in party_dirs:
            in_file = os.path.join(args.output_dir, party, "program_en.txt")
            out_file = os.path.join(args.output_dir, party, "program_sum.txt")
            if not os.path.isfile(in_file):
                print(f"{in_file} not found, skipping.")
                continue
            if os.path.isfile(out_file):
                if args.override:
                    os.remove(out_file)
                else:
                    print(f"{out_file} already exists, skipping.")
                    continue
            print(f"Summarizing {in_file} -> {out_file}")
            with open(in_file, "r") as f:
                content = f.read()
            summary = summarize_content(content, in_file)
            with open(out_file, "w") as f:
                f.write(summary)

    elif args.mode == "reason":
        print('REASONING ABOUT VISUAL POINTS')
        dirname = f'p{args.n_points}_{args.llm.replace("/", "_")}'
        llm = load_llm(args.llm)
        for party in party_dirs:
            for in_file, out_points, input_type in [("program_sum.txt", "program", "election program summary"), ("kommunalomat_responses_en.txt", "kommunalomat", "open election compass answers")]:
                in_file_full = os.path.join(args.output_dir, party, in_file)
                out_points_full = os.path.join(args.output_dir, party, f'results_{dirname}_{out_points}',  "prompt.txt")
                out_reasoning = out_points_full.replace('.txt', '_reasoning.txt')
                if not os.path.isfile(in_file_full):
                    print(f"{in_file_full} not found, skipping.")
                    continue
                if os.path.isfile(out_points_full) and os.path.isfile(out_reasoning):
                    if args.override:
                        os.remove(out_points_full)
                        os.remove(out_reasoning)
                    else:
                        print(f"{out_points_full} and {out_reasoning} already exist, skipping.")
                print(f"Reasoning for {party} {input_type}")
                with open(in_file_full, "r") as f:
                    input = f.read()
                visual_points, reasoning = reason_about_visual_points(llm, input, type_of_input=input_type, n_points=args.n_points)
                os.makedirs(os.path.dirname(out_points_full), exist_ok=True)
                with open(out_points_full, "w") as f:
                    f.write(visual_points)
                with open(out_reasoning, "w") as f:
                    f.write(reasoning)

    elif args.mode == "translate_results":
        print('TRANSLATING REASONING RESULTS TO GERMAN')
        model = load_translation_model()

        for root, dirs, files in os.walk(args.output_dir):
            for input_fname in files:
                if os.path.basename(input_fname) in ["prompt.txt", "prompt_reasoning.txt"]:
                    out_name = os.path.join(root, input_fname.replace('.txt', '_de.txt'))
                    if os.path.isfile(out_name):
                        if args.override:
                            os.remove(out_name)
                        else:
                            print(f"{out_name} already exists, skipping.")
                            continue        
                    print(f"Translating results for {os.path.join(root, input_fname)}")
                    with open(os.path.join(root, input_fname), "r") as f:
                        input = f.read().split('\n')
                    translated = translate(model, input, source_lan='English', target_lan='German')
                    if isinstance(translated, list):
                        translated = '\n'.join(translated)
                    with open(out_name, "w") as f:
                        f.write(translated)

    elif args.mode == "generate_images":
        print('GENERATING IMAGES FOR VISUAL POINTS')
        model = load_model_diffusers(args.image_generator)
    
        for root, dirs, files in os.walk(args.output_dir):
            for input_fname in files:
                if os.path.basename(input_fname) == "prompt.txt":
                    with open(os.path.join(root, input_fname), "r") as f:
                        input = f.read()
                    save_path = os.path.join(root, f'img_{args.image_generator.split("/")[-1]}_guid{args.guidance}_nsteps{args.num_steps}')
                    if os.path.exists(save_path):
                        if args.override:
                            shutil.rmtree(save_path)
                        else:
                            print(f"{save_path} already exists, skipping.")
                            continue
                    print(f"Generating images for {save_path}")
                    generate_images_diffusers(model, input, save_path, args.guidance, args.num_steps, args.n_images)
                    
                                     
    elif args.mode == "describe_image":
        print('DESCRIBING IMAGES WITH VLM')
        
        parties_to_process = args.parties if args.parties else party_dirs
        if args.parties:
             parties_to_process = [p for p in args.parties if p in party_dirs]
             print(f"Processing specific parties: {parties_to_process}")
        
        model, processor = load_vlm(args.vlm)
        
        for root, dirs, files in os.walk(args.output_dir):
            rel_path = os.path.relpath(root, args.output_dir)
            if rel_path == '.':
                continue
            
            current_party = rel_path.split(os.sep)[0]
            if current_party not in parties_to_process:
                continue

            for input_fname in files:
                if os.path.basename(input_fname) == "prompt.txt":
                    # Find corresponding image directory
                    img_dir_name = f'img_{args.image_generator.split("/")[-1]}_guid{args.guidance}_nsteps{args.num_steps}'
                    img_dir = os.path.join(root, img_dir_name)
                    
                    if not os.path.isdir(img_dir):
                        continue
                        
                    # Output file for descriptions
                    party_name = os.path.basename(os.path.dirname(root))
                    result_folder_name = os.path.basename(root)
                    source_type = "program" if "program" in result_folder_name else "kommunalomat"
                    
                    # Save in party directory (parent of root)
                    party_dir = os.path.dirname(root)
                    desc_dir = os.path.join(party_dir, "descriptions")
                    os.makedirs(desc_dir, exist_ok=True)
                    
                    # Check if ANY description files already exist for this batch
                    desc_pattern = os.path.join(desc_dir, f'{party_name}_{source_type}_images_description_{args.vlm.split("/")[-1]}_*.txt')
                    existing_files = glob.glob(desc_pattern)
                    
                    if existing_files and not args.override:
                         print(f"Description files for {party_name} already exist ({len(existing_files)} found) in {desc_dir}, skipping.")
                         continue

                    print(f"Describing images in {img_dir}")
                    
                    # Collect all image paths
                    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
                    image_paths = [os.path.join(img_dir, f) for f in image_files]
                    
                    if not image_paths:
                        print(f"No images found in {img_dir}, skipping.")
                        continue
                    
                    print(f"  - Processing {len(image_paths)} images individually...")
                    descriptions = describe_image_vlm(model, processor, image_paths, n_points=args.n_points)
                    
                    for i, (img_path, desc) in enumerate(zip(image_paths, descriptions)):

                        img_filename = os.path.basename(img_path) 
                        img_id = os.path.splitext(img_filename)[0]
                        # If img filename is just a number, use it.
                        
                        desc_file = os.path.join(desc_dir, f'{party_name}_{source_type}_images_description_{args.vlm.split("/")[-1]}_{img_id}.txt')
                        
                        with open(desc_file, "w") as f:
                            f.write(desc)
                        print(f"Saved description to {desc_file}")

    elif args.mode == "evaluate":
        print('EVALUATING SIMILARITY BETWEEN PROMPTS AND VLM DESCRIPTIONS')
        print(f'Evaluation method: {args.eval_method}')
        results = []
        
        # Filter parties if specified
        parties_to_evaluate = args.parties if args.parties else party_dirs
        if args.parties:
            # Validate that specified parties exist
            invalid_parties = [p for p in args.parties if p not in party_dirs]
            if invalid_parties:
                print(f"Warning: The following parties were not found: {invalid_parties}")
            parties_to_evaluate = [p for p in args.parties if p in party_dirs]
            print(f"Evaluating specific parties: {parties_to_evaluate}")
        
        llm = None
        if args.eval_method in ["llm", "both"]:
            print("Loading LLM for evaluation...")
            llm = load_llm(args.llm)
        
        sources_to_evaluate = args.sources if args.sources else ["program", "kommunalomat"]
        
        total_evaluations = len(parties_to_evaluate) * len(sources_to_evaluate)
        current_eval = 0
        
        for party in parties_to_evaluate:
            party_path = os.path.join(args.output_dir, party)
            
            for source_type in sources_to_evaluate:
                results_dir = os.path.join(party_path, f'results_p{args.n_points}_{args.llm.replace("/", "_")}_{source_type}')
                prompt_file = os.path.join(results_dir, "prompt.txt")
                
                if not os.path.isfile(prompt_file):
                    print(f"Prompt file not found: {prompt_file}, skipping.")
                    continue
                
                # Find all VLM description files for this party/source
                vlm_model_name = args.vlm.split("/")[-1]
                desc_prefix = f'{party}_{source_type}_images_description_{vlm_model_name}_'
                
                desc_files = []
                desc_dir = os.path.join(party_path, "descriptions")
                
                # Check descriptions folder first
                if os.path.isdir(desc_dir):
                    for f in os.listdir(desc_dir):
                        if f.startswith(desc_prefix) and f.endswith(".txt"):
                             desc_files.append(os.path.join(desc_dir, f))
                
                # Fallback: check party dir (migration support)
                if not desc_files and os.path.isdir(party_path):
                     for f in os.listdir(party_path):
                        if f.startswith(desc_prefix) and f.endswith(".txt"):
                             desc_files.append(os.path.join(party_path, f))

                # Fallback: check for the old aggregated file if no individual files found
                if not desc_files:
                    old_desc_file = os.path.join(party_path, f'{party}_{source_type}_images_description_{vlm_model_name}.txt')
                    if os.path.isfile(old_desc_file):
                         desc_files.append(old_desc_file)
                
                if not desc_files:
                    print(f"No description files found for {party} - {source_type}, skipping.")
                    continue
                
                desc_files.sort()

                for desc_file in desc_files:
                    basename = os.path.basename(desc_file)
                    if basename.startswith(desc_prefix):
                         image_id = basename[len(desc_prefix):-4]
                    else:
                         image_id = "aggregated"

                    with open(prompt_file, "r") as f:
                        reference = f.read().strip()
                    with open(desc_file, "r") as f:
                        candidate = f.read().strip()
                    
                    current_eval += 1
                    print(f"\n[{current_eval}/{total_evaluations}] Evaluating {party} - {source_type} - Image {image_id}")
                    
                    # Initialize scores dict
                    scores = {'party': party, 'source': source_type, 'image_id': image_id, 'reference': reference, 'candidate': candidate}
                    
                    try:
                        # Compute embedding-based metrics (BLEU, ROUGE, Cosine)
                        if args.eval_method in ["embedding", "both"]:
                            embedding_scores = evaluate_similarity(reference, candidate)
                            scores.update(embedding_scores)
                            print(f"  Embedding Scores: BLEU={embedding_scores['bleu']}, Cosine={embedding_scores['cosine_similarity']}")
                        
                        # Compute LLM-based evaluation
                        if args.eval_method in ["llm", "both"]:
                            llm_scores = evaluate_alignment_with_llm(llm, reference, candidate)
                            scores.update(llm_scores)
                            print(f"  LLM Score: {llm_scores['llm_score']}/10")
                        
                        results.append(scores)
                    except Exception as e:
                        print(f"  ERROR: Failed to evaluate {party} - {source_type} - {image_id}: {e}")
                        print(f"  Continuing with next evaluation...")
        
        if results:
            results_df = pd.DataFrame(results)
            
            # Determine column order based on eval method
            base_cols = ['party', 'source', 'image_id']
            metric_cols = []
            if args.eval_method in ["embedding", "both"]:
                metric_cols.extend(['bleu', 'rouge1_f', 'rouge2_f', 'rougeL_f', 'cosine_similarity'])
            if args.eval_method in ["llm", "both"]:
                metric_cols.extend(['llm_score', 'llm_reasoning'])
            end_cols = ['reference', 'candidate']
            
            for col in base_cols + metric_cols + end_cols:
                if col not in results_df.columns:
                    results_df[col] = None

            column_order = base_cols + metric_cols + end_cols
            results_df = results_df[column_order]
            
            csv_filename = f'evaluation_results_{args.eval_method}.csv'
            eval_dir = os.path.join(args.output_dir, "evaluations")
            os.makedirs(eval_dir, exist_ok=True)
            csv_path = os.path.join(eval_dir, csv_filename)

            if args.parties and os.path.isfile(csv_path):
                existing_df = pd.read_csv(csv_path)
                
                if 'image_id' not in existing_df.columns:
                    existing_df['image_id'] = 'aggregated'

                # Remove rows for the parties we just evaluated to avoid duplicates
                # Processed all sources for the selected parties.
                existing_df = existing_df[~existing_df['party'].isin(parties_to_evaluate)]
                
                # Combine existing data with new results
                results_df = pd.concat([existing_df, results_df], ignore_index=True)
                results_df = results_df.sort_values(['party', 'source', 'image_id']).reset_index(drop=True)
                print(f"\nUpdated existing CSV with results for: {parties_to_evaluate}")
            
            results_df.to_csv(csv_path, index=False)
            print(f"Saved evaluation results to {csv_path}")
            
            # Print summary statistics
            print("\n=== Summary Statistics ===")
            if args.eval_method in ["embedding", "both"]:
                print(results_df[['bleu', 'rouge1_f', 'rouge2_f', 'rougeL_f', 'cosine_similarity']].describe())
            if args.eval_method in ["llm", "both"]:
                print(f"\nLLM Score Statistics:")
                print(results_df['llm_score'].describe())
                    
    tracker.stop()