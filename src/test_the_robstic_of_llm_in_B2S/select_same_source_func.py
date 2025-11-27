"""
以asm_func为anchor, 分别与src_func和不同函数的N个src_func进行匹配, 计算mrr
"""
from use_llm_matching_BS import setup_logger, load_prompt_from_file
from langchain_ollama import OllamaLLM
from data_play import FuncDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import time

def selection_res_parser(response: str) -> dict:
    """
    解析LLM的响应, 提取分析结果和置信度。
    
    Args:
        response: LLM的完整响应文本
        
    Returns:
        possible_ranking: 索引列表
    """
    possible_ranking = response.splitlines()[0].strip()
    
    return possible_ranking.split('$')

def match_assembly_to_source(asm_func: str, src_func_list: str, prompt_path: str) -> str:
    prompt = load_prompt_from_file(prompt_path)
    model = OllamaLLM(model="qwen3-coder:30b", base_url="http://localhost:11434")

    chain = prompt | model
    result = chain.invoke({
        "asm_func": asm_func,
        "src_func_list": src_func_list,
        "batch_size": len(src_func_list)
    })

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-based Assembly to Source Code Matching")
    parser.add_argument('--data_csv', type=str, default="resources/datasets/dataset.csv", help='Path to the CSV file containing function pairs')
    parser.add_argument('--items_num', type=int, default=1024, help='Number of items to process from the dataset')
    parser.add_argument('--ratio_pos_neg', type=float, default=1.0, help='Ratio of positive to negative samples')
    parser.add_argument('--log_path', type=str, default=f'resources/logs/llm_selection_{time.time()}.log', help='Path to the log file')
    parser.add_argument('--prompt_path', type=str, default="tmp/use_llm_selecting_BS.md", help='Path to the prompt template file')
    
    args = parser.parse_args()
    
    logger = setup_logger(args.log_path)
    dataset = FuncDataset(csv_file=args.data_csv, items_num=args.items_num, ratio_pos_neg=args.ratio_pos_neg, task_type="selection")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    process_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing function pairs", ncols=150)
    avg = []
    all_query_num = 0
    for idx, (asm_func_list, src_func_list) in process_bar:
        ans = 0
        batch_query = 0
        for idx, asm_func in enumerate(asm_func_list):
            try:
                response = match_assembly_to_source(asm_func, src_func_list, prompt_path=args.prompt_path)
                possible_ranking = selection_res_parser(response)
                positon = possible_ranking.index(str(idx))
                ans += 1.0 / (positon + 1)
                all_query_num += 1
                batch_query += 1
            except Exception as e:
                logger.error(f"Error processing function pair index {idx}: {e}")
        batch_avg_mrr = ans / batch_query if batch_query > 0 else 0
        logger.info(f"Batch {idx+1}: MRR = {batch_avg_mrr}")
        avg.append(batch_avg_mrr)
    mrr = sum(avg) / len(avg)
    logger.info(f"Total queries: {all_query_num}")
    logger.info(f"Total MRR: {mrr}")
