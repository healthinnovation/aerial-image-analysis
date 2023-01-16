cd scripts

python split_data_xval_xtst.py --data_csv '../data/buildings.csv' --num_folds 5 --num_exp 0 --save_dir '../data'
python split_data_xval_xtst.py --data_csv '../data/crops.csv' --num_folds 5 --num_exp 0 --save_dir '../data'
python split_data_xval_xtst.py --data_csv '../data/non_vegetated.csv' --num_folds 5 --num_exp 0 --save_dir '../data'
python split_data_xval_xtst.py --data_csv '../data/roads.csv' --num_folds 5 --num_exp 0 --save_dir '../data'
python split_data_xval_xtst.py --data_csv '../data/tillage.csv' --num_folds 5 --num_exp 0 --save_dir '../data'
python split_data_xval_xtst.py --data_csv '../data/vegetated.csv' --num_folds 5 --num_exp 0 --save_dir '../data'