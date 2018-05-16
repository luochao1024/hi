w=4
a=$((w-1))

python3 cnn_flocking_method.py --job_name ps --task_index 0 &>/dev/null &
for ((x=0; x<$a; x++))
do
    python3 cnn_flocking_method.py --job_name worker --task_index $x &>/dev/null &
done

python3 cnn_flocking_method.py --job_name worker --task_index $a