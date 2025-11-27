"""
LLM-based Assembly to Source Code Matching
判断两个函数是否来自同一源码
"""
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import pandas as pd
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import argparse
import os

from data_play import FuncDataset


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("LLM_Matching_Logger")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def matching_res_parser(response: str) -> dict:
    """
    解析LLM的响应, 提取分析结果和置信度。
    
    Args:
        response: LLM的完整响应文本
        
    Returns:
        dict: 包含'analysis'和'if_same_source'的字典
    """
    if_same_source = response.splitlines()[0].strip()
    
    return {
        "if_same_source": if_same_source
    }

def load_prompt_from_file(file_path: str) -> PromptTemplate:
    with open(file_path, 'r', encoding='utf-8') as file:
        template = file.read()
    return PromptTemplate(input_variables=["asm_func", "src_func"], template=template)


def match_assembly_to_source(model_name, asm_func: str, src_func: str, prompt_path: str) -> str:
    prompt = load_prompt_from_file(prompt_path)
    model = OllamaLLM(model=model_name, base_url="http://localhost:11434")

    chain = prompt | model
    result = chain.invoke({
        "asm_func": asm_func,
        "src_func": src_func
    })

    return result

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="LLM-based Assembly to Source Code Matching")
    argparser.add_argument('--prompt_path', type=str, default="tmp/use_llm_matching_BS_no_analysis.md", help='Path to the prompt template file')
    argparser.add_argument('--data_path', type=str, default="resources/datasets/dataset.csv", help='Path to the dataset CSV file')
    argparser.add_argument('--log_path', type=str, default=f"resources/logs/llm_matching_BS_{int(time.time())}.log", help='Path to the log file')
    argparser.add_argument('--items_num', type=int, default=1500, help='Number of items to load from the dataset')
    argparser.add_argument('--ratio_pos_neg', type=float, default=0.5, help='Ratio of positive to negative samples [0.5, 1.0]')
    args = argparser.parse_args()

    dataset = FuncDataset(args.data_path, items_num=args.items_num, ratio_pos_neg=args.ratio_pos_neg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    
    logger = setup_logger(args.log_path)
    logger.info(f"Log path: {args.log_path}")
    model_list = ["qwen2.5-coder:7b", "qwen3-coder:30b"]
    for model_name in model_list:
        logger.info("==================================================================")
        logger.info(f"Evaluating model: {model_name}")
        TP, TN, FP, FN = 0, 0, 0, 0
        logger.info(f"Starting")
        logger.info(f"The ratio of positive pairs to negative pairs: pos / neg = {args.ratio_pos_neg}")
        start_time = time.time()
        progress_bar = tqdm(enumerate(dataloader), 
                        total=len(dataloader),
                        desc=f"Evaluating {model_name}",
                        ncols=150)
        for idx, batch in progress_bar:
            for pair in batch:
                asm_func, src_func, label = pair
                try:
                    result = match_assembly_to_source(model_name, asm_func, src_func, args.prompt_path)
                    result = matching_res_parser(response=result)
                    if_same_source = int(result['if_same_source'])
                    if label == 1 and if_same_source==1:
                        TP += 1
                    elif label == 0 and if_same_source==0:
                        TN += 1
                    elif label == 0 and if_same_source==1:
                        FP += 1
                    elif label == 1 and if_same_source==0:
                        FN += 1
                except Exception as e:
                    logger.error(f"Error parsing response: {result}, error: {e}")
                    continue
        end_time = time.time()
        epoch_duration = end_time - start_time
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        precision = TP / (TP + FP + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        hours = int(epoch_duration / 3600)
        minutes = int((epoch_duration % 3600) / 60)
        seconds = epoch_duration % 60
        logger.info(f"completed in {hours}h {minutes}m {seconds:.2f}s.")
        logger.info("total pairs processed: {}".format(TP + TN + FP + FN))
        logger.info(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        logger.info(f"Accuracy:{accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")