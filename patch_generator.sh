cd scripts

#echo 'Burkina 64x64'
#/usr/bin/time -o time_log_burkina_64x64 -f "Real: %E, User: %U, Sys: %S" python ROI_guided_sampling.py --data_dir "../../dataset-ebs/MasksBurkinaFasoDL/" --output_dir "../../dataset-ebs/dataset-patches/64x64/burkina/" --num_workers 30 --patch_size 64 --threshold 0.05
#sleep 10

echo 'Burkina 128x128'
/usr/bin/time -o time_log_burkina_128x128 -f "Real: %E, User: %U, Sys: %S" python ROI_guided_sampling.py --data_dir "../../dataset-ebs/MasksBurkinaFasoDL/" --output_dir "../../dataset-ebs/dataset-patches-overlap-10/128x128/burkina/" --num_workers 30 --patch_size 128 --threshold 0.05 --overlap 0.1
sleep 10

echo 'Burkina 256x256'
/usr/bin/time -o time_log_burkina_256x256 -f "Real: %E, User: %U, Sys: %S" python ROI_guided_sampling.py --data_dir "../../dataset-ebs/MasksBurkinaFasoDL/" --output_dir "../../dataset-ebs/dataset-patches-overlap-10/256x256/burkina/" --num_workers 30 --patch_size 256 --threshold 0.1 --overlap 0.1
sleep 10

#echo 'Burkina 512x512'
#/usr/bin/time -o time_log_burkina_512x512 -f "Real: %E, User: %U, Sys: %S" python ROI_guided_sampling.py --data_dir "../../dataset-ebs/MasksBurkinaFasoDL/" --output_dir "../../dataset-ebs/dataset-patches/512x512/burkina/" --num_workers 30 --patch_size 512 --threshold 0.2
#sleep 10


#echo 'CIV 64x64'
#/usr/bin/time -o time_log_civ_64x64 -f "Real: %E, User: %U, Sys: %S" python ROI_guided_sampling.py --data_dir "../../dataset-ebs/civ/" --output_dir "../../dataset-ebs/dataset-patches/64x64/civ/" --num_workers 30 --patch_size 64 --threshold 0.05
#sleep 10

echo 'CIV 128x128'
/usr/bin/time -o time_log_civ_128x128 -f "Real: %E, User: %U, Sys: %S" python ROI_guided_sampling.py --data_dir "../../dataset-ebs/civ/" --output_dir "../../dataset-ebs/dataset-patches-overlap-10/128x128/civ/" --num_workers 30 --patch_size 128 --threshold 0.05 --overlap 0.1
sleep 10

echo 'CIV 256x256'
/usr/bin/time -o time_log_civ_256x256 -f "Real: %E, User: %U, Sys: %S" python ROI_guided_sampling.py --data_dir "../../dataset-ebs/civ/" --output_dir "../../dataset-ebs/dataset-patches-overlap-10/256x256/civ/" --num_workers 30 --patch_size 256 --threshold 0.1 --overlap 0.1
#sleep 10

#echo 'CIV 512x512'
#/usr/bin/time -o time_log_civ_512x512 -f "Real: %E, User: %U, Sys: %S" python ROI_guided_sampling.py --data_dir "../../dataset-ebs/civ/" --output_dir "../../dataset-ebs/dataset-patches/512x512/civ/" --num_workers 30 --patch_size 512 --threshold 0.2
