#!/bin/bash

declare -r exe="../matrix_generator"
declare -r m_exe="./print_metrics.sh"
declare -r threads=8
declare -r runnings=10
declare -r iters=100
declare -r error="1e-15"
# declare -a sizes=( "2000" "4000" "6000" "8000"
#                    "10000" "12000" "14000" "16000"
#                    "18000" "20000" "22000" "24000"
#                  )
declare -a sizes=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
		   11000 12000 13000 14000 15000 16000 17000 18000
		   19000 20000)

function main {
  wdir="$(pwd)/data_$$"

  mkdir -p $wdir/dat
  for n in $(seq 5 5 100); do
    mkdir -p $wdir/$n
    for((i = 1; i<=$runnings; i++)); do
      $exe -n $n -t $threads -i $iters -e $error \
	   > $wdir/$n/out_$i.log
      sleep 1
    done
    $m_exe $wdir $n $runnings
  done
}

main
