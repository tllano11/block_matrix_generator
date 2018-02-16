#!/bin/bash

jacobi_vals=$(grep -R run_jacobi out.log | awk '{print $5}')
ceror_vals=$(grep -R compute_error out.log | awk '{print $5}')
cublas_vals=$(grep -R cublas out.log | awk '{print $5}')

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
