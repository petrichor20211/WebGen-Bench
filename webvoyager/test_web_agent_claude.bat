@echo off
python -u run.py ^
    --test_file ./data/tasks_test_realdev_feature.jsonl ^
    --api_model claude-3-5-sonnet-v2 ^
    --api_key xxx ^
    --max_iter 15 ^
    --max_attached_imgs 3 ^
    --temperature 0 ^
    --fix_box_color ^
    --window_height 1080 ^
    --window_width 1920 ^
    --output_dir realdev_res_feature_claude ^
    --seed 42 > tasks_realdev_1.log 