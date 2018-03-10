#!/bin/bash

declare -r log=$1
declare -r n=$(echo $log | tr -cd '[[:digit:]]')

function get_jacobi_metrics {
    declare -r jacobi_vals=$(grep -R run_jacobi $log | awk '{print $5}')
    declare -r ceror_vals=$(grep -R compute_error $log | awk '{print $5}')
    declare -r cublas_vals=$(grep -R cublas $log | awk '{print $5}')
    declare -r jacobi_err=$(grep -R jacobi_err $log | awk '{print $3}')
    declare -r jacobi_iters=$(grep -R jacobi_iters $log | awk '{print $3}')
    declare etime=0

    for val in $jacobi_vals; do
	etime+=" + $val"
    done

    for val in $ceror_vals; do
	etime+=" + $val"
    done

    for val in $cublas_vals; do
	etime+=" + $val"
    done

    echo "$n $(echo $etime | bc)" >> jacobi_time.dat
    echo "$n $jacobi_err" >> jacobi_err.dat
    echo "$n $jacobi_iters" >> jacobi_iters.dat
}

function get_mkl_metrics {
    declare -r mkl_err=$(grep -R mkl_err $log | awk '{print $3}')
    declare -r mkl_iters=$(grep -R mkl_iters $log | awk '{print $3}')
    declare etime="$(grep -R mkl_time $log | awk '{print $3}')*1000"

    echo "$n $(echo $etime | bc)" >> mkl_time.dat
    echo "$n $mkl_err" >> mkl_err.dat
    echo "$n $mkl_iters" >> mkl_iters.dat
}

function main {
    declare -r jacobi_success=$(grep -R jacobi_success $log | awk '{print $3}')
    declare -r mkl_success=$(grep -R mkl_success $log | awk '{print $3}')
    if [ "$jacobi_success" == "no" ] || [ "$mkl_success" == "no" ]; then
	exit 1
    fi
    get_jacobi_metrics
    get_mkl_metrics
}

main
exit 0
