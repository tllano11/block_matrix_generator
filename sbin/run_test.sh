#!/bin/bash

declare -r exe="../matrix_generator"
declare -r m_exe="./print_metrics.sh"
declare -r delta="1"
declare -r threads="8"
declare -r relaxation="1"
declare -r iters="100"
declare -r error="1e-14"
#  declare -a sizes=( "2000" "4000" "6000" "8000"
#                     "10000" "12000" "14000" "16000"
#                     "18000" "20000" "22000" "24000"
#                   )
declare -a sizes=( "100" "200" )

function main {
  declare cmd
  declare -r wdir="$(pwd)/data_$$"

  mkdir -p $wdir
  for n in ${sizes[@]}; do
    mkdir -p $wdir/$n
    for i in {1..10}; do
      $exe -n $n -d $delta -t $threads -r $relaxation -i $iters -e $error \
           > $wdir/$n/out_$i.log
      sleep 5
    done
    $m_exe $wdir/$n
  done
}

main
