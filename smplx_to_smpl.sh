#!/bin/bash

conda init
conda activate smplx

# python /home/mmc-user/smplx/json_to_npz.py

declare -a arr=("s03" "s04" "s05" "s07" "s08" "s09" "s10" "s11")

for i in "${arr[@]}"; do
    echo "$i"
    for filename in /data/datasets/fit3d/train/$i/smplx_npz/*.npz; do
        echo "$filename"
        act=$(grep -E -o "(\w+)\.npz" $filename)
        echo "$act"
        xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
        python write_obj.py \
            --model-folder ../models/ \
            --motion-file $filename \
            --output-folder /data/datasets/fit3d/train/$i/smplx_obj
    done
done

python write_obj.py \
    --model-folder ../models/ \
    --motion-file /data/datasets/fit3d/train/s03/smplx_npz/barbell_dead_row.npz \
    --output-folder /data/datasets/fit3d/train/s03/smplx_obj
