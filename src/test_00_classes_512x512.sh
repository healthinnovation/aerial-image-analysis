echo "Starting training for test 00 in cuda:0"

# Buildings
#python src/train.py -c 'configs/experiment_000_buildings_512x512/config_test_00_cv_00.yaml'

#sleep 20

#python src/train.py -c 'configs/experiment_000_buildings_512x512/config_test_00_cv_01.yaml'

#sleep 20

#python src/train.py -c 'configs/experiment_000_buildings_512x512/config_test_00_cv_02.yaml'

#sleep 20


# Crops
python src/train.py -c 'configs/experiment_000_crops_512x512/config_test_00_cv_00.yaml'

sleep 20

python src/train.py -c 'configs/experiment_000_crops_512x512/config_test_00_cv_01.yaml'

sleep 20

python src/train.py -c 'configs/experiment_000_crops_512x512/config_test_00_cv_02.yaml'

sleep 20

# Non vegetated

python src/train.py -c 'configs/experiment_000_non_vegetated_512x512/config_test_00_cv_00.yaml'

sleep 20

python src/train.py -c 'configs/experiment_000_non_vegetated_512x512/config_test_00_cv_01.yaml'

sleep 20

python src/train.py -c 'configs/experiment_000_non_vegetated_512x512/config_test_00_cv_02.yaml'

sleep 20

# Roads

python src/train.py -c 'configs/experiment_000_roads_512x512/config_test_00_cv_00.yaml'

sleep 20

python src/train.py -c 'configs/experiment_000_roads_512x512/config_test_00_cv_01.yaml'

sleep 20

python src/train.py -c 'configs/experiment_000_roads_512x512/config_test_00_cv_02.yaml'

sleep 20

# Tillage

python src/train.py -c 'configs/experiment_000_tillage_512x512/config_test_00_cv_00.yaml'

sleep 20

python src/train.py -c 'configs/experiment_000_tillage_512x512/config_test_00_cv_01.yaml'

sleep 20

python src/train.py -c 'configs/experiment_000_tillage_512x512/config_test_00_cv_02.yaml'

sleep 20

# Vegetated

python src/train.py -c 'configs/experiment_000_vegetated_512x512/config_test_00_cv_00.yaml'

sleep 20

python src/train.py -c 'configs/experiment_000_vegetated_512x512/config_test_00_cv_01.yaml'

sleep 20

python src/train.py -c 'configs/experiment_000_vegetated_512x512/config_test_00_cv_02.yaml'

sleep 20
echo "Finished!"