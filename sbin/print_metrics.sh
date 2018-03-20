#!/bin/bash

declare -r log_dir=$1
declare -r odir=$1/dat
declare -r num_runnings=10
declare -r n=$(basename $log)
jacobi_succeed=0
mkl_succeed=0
eigen_succeed=0

function get_jacobi_metrics {
  jacobi_success=$(grep "jacobi_success" $log | awk '{print $3}')

  if [ "$jacobi_success" == "no" ]; then
    jacobi_err=$(grep "jacobi_err" $log | awk '{print $3}')
    jacobi_err_accum=$((jacobi_err_accum + jacobi_err))
    return
  fi

  jacobi_vals=$(grep "run_jacobi" $log | awk '{print $5}')
  cerror_vals=$(grep "compute_error" $log | awk '{print $5}')
  cublas_vals=$(grep "cublas" $log | awk '{print $5}')
  jacobi_rel_err=$(grep "jacobi_rel_err" $log | awk '{print $3}')
  jacobi_iters=$(grep "jacobi_iters" $log | awk '{print $3}')
  etime=0

  for val in $jacobi_vals; do
    etime=$((etime + val))
  done

  for val in $cerror_vals; do
    etime=$((etime + val))
  done

  for val in $cublas_vals; do
    etime=$((etime + val))
  done

  jacobi_time_accum=$((jacobi_time_accum + etime))
  jacobi_rel_err_accum=$((jacobi_rel_err_accum + jacobi_rel_err))
  jacobi_iters_accum=$((jacobi_iters_accum + jacobi_iters))
  jacobi_succeed=$((jacobi_succeed + 1))
}

function get_mkl_metrics {
  mkl_success=$(grep "mkl_success" $log | awk '{print $3}')

  if [ "$mkl_success" == "no" ]; then
    mkl_err=$(grep "mkl_err" $log | awk '{print $3}')
    mkl_err_accum=$((mkl_err_accum + mkl_err))
    return
  fi

  mkl_err=$(grep "mkl_err" $log | awk '{print $3}')
  mkl_iters=$(grep "mkl_iters" $log | awk '{print $3}')
  etime=$(grep "mkl_time" $log | awk '{print $3}')
  etime=$((etime*1000))

  mkl_time_accum=$((mkl_time_accum + etime))
  mkl_rel_err_accum=$((mkl_rel_err_accum + mkl_rel_err))
  mkl_iters_accum=$((mkl_iters_accum + mkl_iters))
  mkl_succeed=$((mkl_succeed + 1))
}

function get_eigen_metrics {
  eigen_success=$(grep "eigen_success" $log | awk '{print $3}')

  if [ "$eigen_success" == "no" ]; then
    eigen_err=$(grep "eigen_err" $log | awk '{print $3}')
    eigen_err_accum=$((eigen_err_accum + eigen_err))
    return
  fi

  eigen_err=$(grep "eigen_err" $log | awk '{print $3}')
  eigen_iters=$(grep "eigen_iters" $log | awk '{print $3}')
  etime="$(grep "eigen_time" $log | awk '{print $3}')"

  eigen_time_accum=$((eigen_time_accum + etime))
  eigen_rel_err_accum=$((eigen_rel_err_accum + eigen_rel_err))
  eigen_iters_accum=$((eigen_iters_accum + eigen_iters))
  eigen_succeed=$((eigen_succeed + 1))
}

function main {
  mkdir -p $odir
  for i in $(ls $log_dir); do
    get_jacobi_metrics
    get_mkl_metrics
    get_eigen_metrics
  done

  if [ $jacobi_succeed > 0 ]; then
    echo "$n $(bc -l <<< \"scale=3;$jacobi_succeeded/$num_runnings\")" >> $odir/jacobi_succeeded.dat
    echo "$n $(bc -l <<< \"scale=3;$jacobi_time_accum/$jacobi_succeeded\")" >> $odir/jacobi_time.dat
    echo "$n $(bc -l <<< \"scale=3;$jacobi_rel_err_accum/$jacobi_succeeded\")" >> $odir/jacobi_err.dat
    echo "$n $(bc -l <<< \"scale=3;$jacobi_iters_accum/$jacobi_succeeded\")" >> $odir/jacobi_iters.dat
  fi

  if [ $jacobi_succeed < $num_runnings ]; then
    jacobi_fails=$((num_runnings - jacobi_succeed))
    echo "$n $(bc -l <<< \"scale=3;$jacobi_err_accum/$jacobi_fails\")" >> $odir/jacobi_failed_err.dat
  fi

  if [ $mkl_succeed > 0 ]; then
    echo "$n $(bc -l <<< \"scale=3;$mkl_succeeded/$num_runnings\")" >> $odir/mkl_succeeded.dat
    echo "$n $(bc -l <<< \"scale=3;$mkl_time_accum/$mkl_succeeded\")" >> $odir/mkl_time.dat
    echo "$n $(bc -l <<< \"scale=3;$mkl_rel_err_accum/$mkl_succeeded\")" >> $odir/mkl_err.dat
    echo "$n $(bc -l <<< \"scale=3;$mkl_iters_accum/$mkl_succeeded\")" >> $odir/mkl_iters.dat
  fi

  if [ $mkl_succeed < $num_runnings ]; then
    mkl_fails=$((num_runnings - mkl_succeed))
    echo "$n $(bc -l <<< \"scale=3;$mkl_err_accum/$mkl_fails\")" >> $odir/mkl_failed_err.dat
  fi

  if [ $eigen_succeed > 0 ]; then
    echo "$n $(bc -l <<< \"scale=3;$eigen_succeeded/$num_runnings\")" >> $odir/eigen_succeeded.dat
    echo "$n $(bc -l <<< \"scale=3;$eigen_time_accum/$eigen_succeeded\")" >> $odir/eigen_time.dat
    echo "$n $(bc -l <<< \"scale=3;$eigen_rel_err_accum/$eigen_succeeded\")" >> $odir/eigen_err.dat
    echo "$n $(bc -l <<< \"scale=3;$eigen_iters_accum/$eigen_succeeded\")" >> $odir/eigen_iters.dat
  fi

  if [ $eigen_succeed < $num_runnings ]; then
    eigen_fails=$((num_runnings - eigen_succeed))
    echo "$n $(bc -l <<< \"scale=3;$eigen_err_accum/$eigen_fails\")" >> $odir/eigen_failed_err.dat
  fi

}

main
exit 0
