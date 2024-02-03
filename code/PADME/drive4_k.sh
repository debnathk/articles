CUDA_VISIBLE_DEVICES=1
spec='python driver.py --dataset kiba --cross_validation 
--model tf_regression --prot_desc_path KIBA_data/prot_desc.csv 
--model_dir ./model_dir/model_dir4_kiba_ct --cold_target 
--arithmetic_mean --aggregate toxcast --filter_threshold 6 
--intermediate_file ./interm_files/intermediate_cv_ctarget_2.csv '
eval $spec
