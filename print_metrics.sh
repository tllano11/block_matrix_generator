#!/bin/bash

log=$1

jacobi_vals=$(grep -R run_jacobi $log | awk '{print $5}')
ceror_vals=$(grep -R compute_error $log | awk '{print $5}')
cublas_vals=$(grep -R cublas $log | awk '{print $5}')

etime=0
for val in $jacobi_vals; do
    etime="$(echo "$etime + $val" | bc)"
done

echo "run_jacobi elapsed time: $etime ms"

etime=0
for val in $ceror_vals; do
    etime="$(echo "$etime + $val" | bc)"
done

echo "compute_error elapsed time: $etime ms"

etime=0
for val in $cublas_vals; do
    etime="$(echo "$etime + $val" | bc)"
done

echo "cublas elapsed time: $etime ms"

mkl_val=$(grep -R mkl_dgesv $log | awk '{print $5}')
echo "mkl elapsed time: $mkl_val sec"
