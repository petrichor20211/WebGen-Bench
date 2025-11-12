@echo off
cd /d G:\personal\WebGen-Bench

python webvoyager/evaluation/collect_res.py ^
--model deepseek/deepseek-r1-0528 ^
--token xxx ^
--api-url https://openrouter.ai/api/v1 ^
--tasks-file webvoyager/data/tasks_test_realdev_feature.jsonl ^
--root-dir G:\personal\cua\claude_feature ^
--output G:\personal\cua\claude_feature\result.jsonl ^
--use-all-messages ^
--save-interval 10