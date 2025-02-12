# from /root/transformers/examples/pytorch/language-modeling/run_clm.py
python3 run_clm.py --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --do_train --do_eval --output_dir /tmp/test-clm --overwrite_output_dir --max_steps 10
