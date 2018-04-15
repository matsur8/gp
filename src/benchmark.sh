python src/benchmark.py sklgp_exact --limit_n_train 6400
python src/benchmark.py sklgp_exact --limit_n_train 6400 --optimize 
python src/benchmark.py gpy_exact --limit_n_train 6400
python src/benchmark.py gpy_exact --limit_n_train 6400 --optimize 
python src/benchmark.py gpflow_exact --limit_n_train 6400
python src/benchmark.py gpflow_exact --limit_n_train 6400 --optimize 
python src/benchmark.py gpml_exact --limit_n_train 6400
python src/benchmark.py gpml_exact --limit_n_train 6400  --optimize 

python src/benchmark.py gpy_sgpr --make_model_option 10 --limit_n_train 12800
python src/benchmark.py gpy_sgpr --make_model_option 100 --limit_n_train 12800 --optimize 
python src/benchmark.py gpflow_sgpr --make_model_option 10 --limit_n_train 102400
python src/benchmark.py gpflow_sgpr --make_model_option 100 --limit_n_train 102400 --optimize 
python src/benchmark.py gpflow_fitc --make_model_option 10 --limit_n_train 102400
python src/benchmark.py gpflow_fitc --make_model_option 100 --limit_n_train 102400 --optimize 
python src/benchmark.py gpflow_sgpr --make_model_option 10 --limit_n_train 102400
python src/benchmark.py gpflow_sgpr --make_model_option 100 --limit_n_train 102400 --optimize 
python src/benchmark.py gpml_msgp --make_model_option 1000 --predict_option 1000 --limit_n_train 102400
python src/benchmark.py gpml_msgp --make_model_option 10000 --predict_option 10000 --limit_n_train 102400 --optimize



