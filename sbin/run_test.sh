#!/bin/bash

declare -r exe="../matrix_generator"
declare -r m_exe="./print_metrics.sh"
declare -r threads=8
declare -r runnings=10
declare -r iters=100
declare -r error="10e-15"
# declare -a sizes=( "2000" "4000" "6000" "8000"
#                    "10000" "12000" "14000" "16000"
#                    "18000" "20000" "22000" "24000"
#                  )
declare -a sizes=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
		   11000 12000 13000 14000 15000 16000 17000 18000
		   19000 20000)

function run_test_1a {
    local wdir="$(pwd)/data_$$/1a"

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

function run_test_1b {
    local wdir="$(pwd)/data_$$/1b"

    mkdir -p $wdir/dat
    for n in $(seq 1000 1000 24000); do
	mkdir -p $wdir/$n
	for((i = 1; i<=$runnings; i++)); do
	    $exe -n $n -t $threads -i $iters -e $error \
		 > $wdir/$n/out_$i.log
	    sleep 1
	done
	$m_exe $wdir $n $runnings
    done
}

function run_test_2a {
    local wdir="$(pwd)/data_$$/2a"

    #cp print_metrics.sh print_metrics.sh.bk
    #sed -i 's,#get_mkl,get_mkl,g' print_metrics.sh
    #sed -i 's,#get_eigen,get_eigen,g' print_metrics.sh

    #local cdir="$(pwd)"
    #cd ..
    #make clean
    #cp matrix_generator-v3.cpp matrix_generator-v3.cpp.bk
    #sed -i 's,//solve,solve,g' matrix_generator-v3.cpp
    #make
    #cd $cdir

    mkdir -p $wdir/dat
    for n in $(seq 2000 2000 24000); do
	mkdir -p $wdir/$n
	for((i = 1; i<=$runnings; i++)); do
	    $exe -n $n -t $threads -i $iters -e $error \
		 > $wdir/$n/out_$i.log
	    sleep 1
	done
	$m_exe $wdir $n $runnings
    done
    #mv print_metrics.sh.bk print_metrics.sh
}

function run_test_3a {
    local wdir="$(pwd)/data_$$/3a"
    #local cdir="$(pwd)"

    #cp print_metrics.sh print_metrics.sh.bk
    #sed -i 's,#get_jacobi_cpu,get_jacobi_cpu,g' print_metrics.sh

    #cd ..
    #make clean
    #cp -f matrix_generator-v3.cpp.bk matrix_generator-v3.cpp
    #sed -i 's,//launch_jacobi_cpu,launch_jacobi_cpu,g' matrix_generator-v3.cpp
    #make
    #cd $cdir

    mkdir -p $wdir/dat
    for n in $(seq 2000 2000 36000); do
	mkdir -p $wdir/$n
	for((i = 1; i<=$runnings; i++)); do
	    $exe -n $n -t $threads -i $iters -e $error \
		 > $wdir/$n/out_$i.log
	    sleep 1
	done
	$m_exe $wdir $n $runnings
    done

    #mv print_metrics.sh.bk print_metrics.sh
}

function run_test_4a {
    local wdir="$(pwd)/data_$$/4a"
    local iters_4a=5
    #local cdir="$(pwd)"
    #cd ..
    #make clean
    #mv matrix_generator-v3.cpp.bk matrix_generator-v3.cpp
    #make
    #cd $cdir

    mkdir -p $wdir/dat
    for n in $(seq 2000 2000 36000); do
	mkdir -p $wdir/$n
	for((i = 1; i<=$runnings; i++)); do
	    $exe -n $n -t $threads -i $iters_4a -e $error \
		 > $wdir/$n/out_$i.log
	    sleep 1
	done
	$m_exe $wdir $n $runnings
    done
}

function main {
    run_test_1a
    #sleep 100
    #run_test_1b
    #sleep 100
    #run_test_2a
    #sleep 100
    #run_test_3a
    #sleep 100
    #run_test_4a
}

main
