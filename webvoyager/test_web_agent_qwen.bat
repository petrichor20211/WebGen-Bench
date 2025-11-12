@echo off
python -u run.py ^
    --test_file ./data/tasks_test_realdev_feature.jsonl ^
    --api_model qwen/qwen2.5-vl-32b-instruct ^
    --api_key xxx ^
    --max_iter 15 ^
    --max_attached_imgs 3 ^
    --temperature 0 ^
    --fix_box_color ^
    --window_height 1080 ^
    --window_width 1920 ^
    --output_dir realdev_res_feature_qwen ^
    --seed 42 > tasks_realdev.log 