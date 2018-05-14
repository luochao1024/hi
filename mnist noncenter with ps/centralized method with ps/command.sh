python3 cnn_synchronous_method.py --job_name ps --task_index 0
#t=29
#p=3
#for ((x=0; x<$t; x++))
#do
#    taskset -c $x python3 cnn_synchronous_method.py --job_name worker --task_index $x &#>/dev/null &
#done
#
#for ((x=0; x<$p; x++))
#do
#    let cpu=$x+$t
#    taskset -c $cpu python3 cnn_synchronous_method.py --job_name ps --task_index $x &#>/dev/null &
#done
