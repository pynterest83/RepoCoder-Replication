# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from make_window import MakeWindowWrapper
from build_vector import BuildVectorWrapper, BagOfWords
from search_code import CodeSearchWrapper
from build_prompt import BuildPromptWrapper
from codegen import CodeGenModel
from compute_score import compute_score_by_repo_with_metadata

from utils import CONSTANTS, CodeGenTokenizer, FilePathBuilder, Tools

def make_repo_window(repos, window_sizes, slice_sizes):
    worker = MakeWindowWrapper(None, repos, window_sizes, slice_sizes)
    worker.window_for_repo_files()

def make_repo_vector(repos, window_sizes, slice_sizes, vectorizer):
    worker = BuildVectorWrapper(None, vectorizer, repos, window_sizes, slice_sizes)
    worker.vectorize_repo_windows()


def run_RG1_and_oracle_method(benchmark, repos, window_sizes, slice_sizes, vectorizer_type):
    # build code snippets for all the repositories
    make_repo_window(repos, window_sizes, slice_sizes)
    # build code snippets for vanilla retrieval-augmented approach and ground truth
    MakeWindowWrapper(benchmark, repos, window_sizes, slice_sizes).window_for_baseline_and_ground()
    # build vector for vanilla retrieval-augmented approach and ground truth
    vectorizer = BagOfWords
    make_repo_vector(repos, window_sizes, slice_sizes, vectorizer)
    BuildVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_baseline_and_ground_windows()
    # search code for vanilla retrieval-augmented approach and ground truth
    CodeSearchWrapper(vectorizer_type, benchmark, repos, window_sizes, slice_sizes).search_baseline_and_ground()
    # build prompt for vanilla retrieval-augmented approach and ground truth
    tokenizer = CodeGenTokenizer
    mode = CONSTANTS.rg
    output_file_path = 'prompts/rg-' + vectorizer_type + '-ws-20-ss-2.jsonl'
    BuildPromptWrapper(vectorizer_type, benchmark, repos, window_sizes, slice_sizes, tokenizer).build_first_search_prompt(mode, output_file_path)

    mode = CONSTANTS.gt
    output_file_path = 'prompts/gt-' + vectorizer_type + '-ws-20-ss-2.jsonl'
    BuildPromptWrapper(vectorizer_type, benchmark, repos, window_sizes, slice_sizes, tokenizer).build_first_search_prompt(mode, output_file_path)


def run_RepoCoder_method(benchmark, repos, window_sizes, slice_sizes, prediction_path, mode, vectorizer_type):
    MakeWindowWrapper(benchmark, repos, window_sizes, slice_sizes).window_for_prediction(mode, prediction_path)
    vectorizer = BagOfWords
    BuildVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_prediction_windows(mode, prediction_path)
    CodeSearchWrapper(vectorizer_type, benchmark, repos, window_sizes, slice_sizes).search_prediction(mode, prediction_path)
    tokenizer = CodeGenTokenizer
    output_file_path = 'prompts/' + mode.replace("-", "") + '-' + vectorizer_type + '-ws-20-ss-2.jsonl'
    BuildPromptWrapper(vectorizer_type, benchmark, repos, window_sizes, slice_sizes, tokenizer).build_prediction_prompt(mode, prediction_path, output_file_path)

def get_repos(benchmark):
    if benchmark == 'short_api_benchmark':
        mode = CONSTANTS.short_api_benchmark
    elif benchmark == 'short_line_benchmark':
        mode = CONSTANTS.short_line_benchmark
    elif benchmark == 'api_benchmark':
        mode = CONSTANTS.api_benchmark
    else:
        mode = CONSTANTS.line_benchmark

    repos = []
    repo_base_dir = ''
    if mode == 'short_api':
        repo_base_dir = 'repositories/line_and_api_level'
    else:
        repo_base_dir = 'repositories/function_level'
    
    # base dir contains folders for each repo with the repo name
    for repo in os.listdir(repo_base_dir):
        repos.append(repo)
    return mode, repos, repo_base_dir

def check_boundaries(oracle_method_score, cur_score):
    for repo in oracle_method_score:
        if cur_score[repo] >= oracle_method_score[repo] - 0.01:
            return True
    return False

def run_iteration(cur_mode, cg, benchmark_mode, repos, window_sizes, slice_sizes, vectorizer_type, model_name):
    input_path, output_path = FilePathBuilder.build_input_output_path(cur_mode, vectorizer_type, model_name)
    cg.generate_by_batch_size(input_path, output_path, max_new_tokens=100)
    cur_score = compute_score_by_repo_with_metadata(repos, Tools.load_jsonl(output_path), args.score_type, passk=1)
    prediction_path = output_path
    cur_mode = cur_mode + '-' + cur_mode
    run_RepoCoder_method(benchmark_mode, repos, window_sizes, slice_sizes, prediction_path, cur_mode, vectorizer_type)

    return cur_score, cur_mode

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run the pipeline for RepoCoder')
    parser.add_argument('--bench_mark', type=str, default='short_api_benchmark')
    parser.add_argument('--window_sizes', nargs='+', type=int, default=[20])
    parser.add_argument('--slice_sizes', nargs='+', type=int, default=[2])
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--vectorizer_type', type=str, default='one-gram')
    parser.add_argument('--model_name', type=str, default='Salesforce/codegen-350M-mono')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--score_type', type=str, default='EM')

    args = parser.parse_args()

    benchmark = args.bench_mark
    benchmark_mode, repos, repo_base_dir = get_repos(benchmark)
    FilePathBuilder.set_repo_base_dir(repo_base_dir)

    window_sizes = args.window_sizes  # 20
    slice_sizes = args.slice_sizes  # 2
    vectorizer_type = args.vectorizer_type
    model_name = args.model_name
    batch_size = args.batch_size
    iterations = args.iterations
    cg = CodeGenModel(model_name, batch_size)

    if args.iterations == 0:
        # iter until the score converges for the ground truth method
        run_RG1_and_oracle_method(benchmark_mode, repos, window_sizes, slice_sizes, vectorizer_type)
        # compute the score for the ground truth method
        cg.generate_by_batch_size('prompts/gt-' + vectorizer_type + '-ws-20-ss-2.jsonl', 'predictions/gt-' + vectorizer_type + '-ws-20-ss-2.jsonl', max_new_tokens=100)
        oracle_method_score = compute_score_by_repo_with_metadata(repos, Tools.load_jsonl('predictions/gt-' + vectorizer_type + '-ws-20-ss-2.jsonl'), args.score_type, passk=1)
        cur_mode = 'r-g'
        cur_score = {repo: 0 for repo in repos}
        while not check_boundaries(oracle_method_score, cur_score):
            cur_score, cur_mode = run_iteration(cur_mode, cg, benchmark_mode, repos, window_sizes, slice_sizes, vectorizer_type, model_name)

    else:
        run_RG1_and_oracle_method(benchmark_mode, repos, window_sizes, slice_sizes, vectorizer_type)
        cur_mode = 'r-g'
        # iter for iterations times.
        for i in range (iterations):
            _, cur_mode = run_iteration(cur_mode, cg, benchmark_mode, repos, window_sizes, slice_sizes, vectorizer_type, model_name)