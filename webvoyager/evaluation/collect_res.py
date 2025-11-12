import argparse
import json
import os
import re
from typing import Dict, List, Set
from openai import OpenAI

# Default configuration for LLM service
DEFAULT_API_URL = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1")
DEFAULT_API_TOKEN = os.getenv("LLM_API_TOKEN", "")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-r1:free")

# Default checkpoint configuration
DEFAULT_SAVE_INTERVAL = 10  # Save every 10 tasks

def _load_existing_results(output_path: str) -> tuple[List[Dict], Set[str]]:
    """Load existing results from the output file.
    
    Returns:
        tuple: (List of existing results, Set of processed task IDs)
    """
    results = []
    processed_ids = set()
    
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    result = json.loads(line)
                    results.append(result)
                    processed_ids.add(result["id"])
        print(f"[INFO] Loaded {len(results)} existing results")
    
    return results, processed_ids

def _save_results(results: List[Dict], output_path: str) -> None:
    """Save current results to the output file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for item in results:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved {len(results)} results to {output_path}")


def _load_interact_messages(path: str, use_all_messages: bool = False) -> str:
    """Return assistant message content from the interact_messages.json file."""
    with open(path, "r", encoding="utf-8") as file:
        messages: List[Dict[str, str]] = json.load(file)

    if use_all_messages:
        # Collect all assistant messages
        assistant_contents = []
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content:
                    assistant_contents.append(content)
        return "\n\n---\n\n".join(assistant_contents)
    else:
        # Return only the last assistant message
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
    return ""


def _judge_result_with_llm(
    instruction: str,
    feedback: str,
    model: str,
    api_url: str,
    token: str
) -> str:
    """Determine final result using the LLM service."""
    system_prompt = (
        "You are an evaluation assistant. Given a test instruction and its feedback, "
        "determine if the specified functionality has been successfully implemented. "
        "Consider the following:\n"
        "1. Does the feedback directly address the required functionality?\n"
        "2. Does it indicate successful completion or implementation?\n"
        "3. Are there any reported errors or failures?\n"
        "4. Is there insufficient information to make a clear determination?\n\n"
        "Respond with exactly:\n"
        "'YES' if the functionality is implemented successfully\n"
        "'NO' if it failed or is incomplete\n"
        "'UNCERTAIN' if there is insufficient information to make a determination\n\n"
        "Instruction: {instruction}\n"
        "Test Feedback: {feedback}"
    )

    # Print LLM input
    # print("\n[LLM Input]")
    # print(system_prompt.format(instruction=instruction, feedback=feedback))

    try:
        client = OpenAI(base_url=api_url, api_key=token)
        completion = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "system",
                "content": system_prompt.format(instruction=instruction, feedback=feedback)
            }],
            temperature=0.1
        )
        response = completion.choices[0].message.content.strip().upper()
        
        # Print LLM output
        # print("\n[LLM Output]")
        # print(response)
        
        if "YES" in response:
            return "YES"
        elif "NO" in response:
            return "NO"
        return "UNCERTAIN"
            
    except Exception as error:
        print(f"[WARN] LLM call failed: {error}. Defaulting to 'NO'.")
        return "NO"


def split_ques_to_records(task: Dict) -> List[Dict]:
    """Split the 'ques' field into multiple records based on numbered items."""
    matches = re.findall(r'(\d+)\.([^\n]+)', task.get('ques', ''))
    return [{
        **task.copy(),
        'ques': content.strip(),
        'id': f"{task['web_name']}-{idx}"
    } for idx, content in matches]


def _process_tasks(
    tasks_file: str,
    root_dir: str,
    output_path: str,
    model: str,
    api_url: str,
    token: str,
    use_all_messages: bool = False,
    save_interval: int = DEFAULT_SAVE_INTERVAL,
) -> None:
    """Process tasks and write results to output file."""
    # Load existing results if any
    results, processed_ids = _load_existing_results(output_path)
    tasks_since_save = 0

    with open(tasks_file, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
                
            task = json.loads(line)
            task_id = task.get("id", "")
            im_path = os.path.join(root_dir, f"task{task_id}", "interact_messages.json")

            # Handle missing or empty interaction files
            if not os.path.isfile(im_path):
                print(f"[WARN] Missing interact_messages.json for {task_id}")
                sub_tasks = split_ques_to_records(task)
                for sub_task in sub_tasks:
                    if sub_task["id"] not in processed_ids:
                        sub_task["res"] = "NO"
                        results.append(sub_task)
                        processed_ids.add(sub_task["id"])
                continue

            messages = _load_interact_messages(im_path, use_all_messages)
            if not messages:
                print(f"[WARN] No assistant message found for {task_id}")
                sub_tasks = split_ques_to_records(task)
                for sub_task in sub_tasks:
                    if sub_task["id"] not in processed_ids:
                        sub_task["res"] = "NO"
                        results.append(sub_task)
                        processed_ids.add(sub_task["id"])
                continue

            # Process each sub-task
            for sub_task in split_ques_to_records(task):
                if sub_task["id"] in processed_ids:
                    print(f"[INFO] Skipping already processed task: {sub_task['id']}")
                    continue
                    
                res = _judge_result_with_llm(
                    instruction=sub_task['ques'],
                    feedback=messages,
                    model=model,
                    api_url=api_url,
                    token=token
                )
                sub_task["res"] = res
                results.append(sub_task)
                processed_ids.add(sub_task["id"])
                print(f"[INFO] {task_id}-{sub_task['id']}: {res}")
                
                # Check if we should save based on count
                tasks_since_save += 1
                if tasks_since_save >= save_interval:
                    _save_results(results, output_path)
                    tasks_since_save = 0

    # Final save and summary
    _save_results(results, output_path)

    yes_count = sum(1 for r in results if r["res"] == "YES")
    no_count = sum(1 for r in results if r["res"] == "NO")
    uncertain_count = sum(1 for r in results if r["res"] == "UNCERTAIN")
    
    print(f"\n[SUMMARY] Test Results:")
    print(f"  YES: {yes_count}")
    print(f"  NO: {no_count}")
    print(f"  UNCERTAIN: {uncertain_count}")
    print(f"  Total: {len(results)}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Collect results from web agent runs.")
    
    parser.add_argument(
        "--tasks-file",
        default="webvoyager/data/tasks_test_realdev.jsonl",
        help="Path to the tasks JSONL file.",
    )
    parser.add_argument(
        "--root-dir",
        default="webvoyager/realdev_test",
        help="Root directory containing task folders.",
    )
    parser.add_argument(
        "--output",
        default="webvoyager/realdev_test/results.jsonl",
        help="Output path for the augmented JSONL with results.",
    )
    parser.add_argument(
        "--use-all-messages",
        action="store_true",
        help="If set, use all assistant messages for judgment instead of just the last one.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="LLM API URL.")
    parser.add_argument("--token", default=DEFAULT_API_TOKEN, help="Bearer token for the LLM API.")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=DEFAULT_SAVE_INTERVAL,
        help="Number of tasks to process before saving results.",
    )
    
    return parser.parse_args()


def main() -> None:
    """Script entry-point."""
    args = parse_args()
    _process_tasks(
        tasks_file=args.tasks_file,
        root_dir=args.root_dir,
        output_path=args.output,
        model=args.model,
        api_url=args.api_url,
        token=args.token,
        use_all_messages=args.use_all_messages,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main() 