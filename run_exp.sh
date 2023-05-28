parent_folder=$1

for i in $(seq 1 10);
do
    new_folder=$parent_folder/results_seed_$i
    rm -rf $new_folder
    mkdir -p $new_folder

    echo "Staring process in $new_folder"

    python run.py --folder $new_folder \
        --vars $parent_folder/vars_run.yaml \
        --conf "RANDOM_STATE=$i" \
        --save_models
done