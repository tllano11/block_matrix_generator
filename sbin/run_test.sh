#!/bin/bash

declare -r exe="../matrix_generator"
declare -r m_exe="./print_metrics.sh"
declare -r delta="1"
declare -r threads="8"
declare -r relaxation="1"
declare -r iters="100"
declare -r error="0.00000001"
# declare -a sizes=( "1000" "2000" "3000" "4000" "5000" "6000" "7000" "8000"
# 		   "9000" "10000" "11000" "12000" "13000" "14000" "15000"
# 		   "16000" "17000" "18000" "19000" "20000" "21000" "22000"
# 		   "23000" "24000" "25000"
# 		 )
declare -a sizes=( "1000" "2000" )

function main {
    declare cmd
    declare -r wdir="$(pwd)/data_$$"

    mkdir -p $wdir
    for n in ${sizes[@]}; do
	$exe -n $n -d $delta -t $threads -r $relaxation -i $iters -e $error \
	     > $wdir/out_$n.log 2> $wdir/err_$n.log
	$m_exe $wdir/out_$n.log $wdir
	sleep 5
    done
}

main
