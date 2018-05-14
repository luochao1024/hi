w=50
t=$((w-1))
taskset -c $w,$((w+1)),$((w+2)),$((w+3)),$((w+4)),$((w+5)),$((w+6)),$((w+7)),$((w+8)),$((w+9)),$((w+10)),$((w+11)),$((w+12)),$((w+13)) python3 cnn_flocking_method.py --job_name ps --task_index 0 &
for ((x=0; x<$t; x++))
do
    taskset -c $x python3 cnn_flocking_method.py  --job_name worker --task_index $x &#>/dev/null &
done

taskset -c $t python3 cnn_flocking_method.py --job_name worker --task_index $t

#taskset -c 2 python3 test_task0.py  --task_index 2 &
#taskset -c 1 python3 test_task0.py  --task_index 1 &
#taskset -c 0 python3 test_task0.py  --task_index 0

#taskset -c 0 python3 cnn_flocking_method.py --task_index 0 &
#taskset -c 1 python3 cnn_flocking_method.py --task_index 1