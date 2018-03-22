#!/bin/bash

log_dir=$1
odir=$1/dat
num_runnings=10
n=$(basename $log_dir)
jacobi_succeed=0
mkl_succeed=0
eigen_succeed=0
jacobi_time_accum=0
jacobi_rel_err_accum=0
jacobi_iters_accum=0
jacobi_err_accum=0
mkl_time_accum=0
mkl_rel_err_accum=0
mkl_iters_accum=0
mkl_err_accum=0
eigen_time_accum=0
eigen_rel_err_accum=0
eigen_iters_accum=0
eigen_err_accum=0

function get_jacobi_metrics {
  declare -r log=$1
  jacobi_success=$(grep "jacobi_success" $log | awk '{print $3}')

  if [ "$jacobi_success" == "no" ]; then
    jacobi_err=$(grep "jacobi_err" $log | awk '{print $3}')
    jacobi_err_accum=$(bc -l <<< "$jacobi_err_accum + $jacobi_err")
    return
  fi

  jacobi_vals=$(grep "run_jacobi" $log | awk '{print $5}')
  cerror_vals=$(grep "compute_error" $log | awk '{print $5}')
  cublas_vals=$(grep "cublas" $log | awk '{print $5}')
  jacobi_rel_err=$(grep "jacobi_rel_err" $log | awk '{print $3}')
  jacobi_iters=$(grep "jacobi_iters" $log | awk '{print $3}')
  etime=0

  for val in $jacobi_vals; do
    etime=$(bc -l <<< "$etime + $val")
  done

  for val in $cerror_vals; do
    etime=$(bc -l <<< "$etime + $val")
  done

  for val in $cublas_vals; do
    etime=$(bc -l <<< "$etime + $val")
  done

  jacobi_time_accum=$(bc -l <<< "$jacobi_time_accum + $etime")
  jacobi_rel_err_accum=$(bc -l <<< "$jacobi_rel_err_accum + $jacobi_rel_err")
  jacobi_iters_accum=$(bc -l <<< "$jacobi_iters_accum + $jacobi_iters")
  jacobi_succeed=$((jacobi_succeed + 1))
}

function get_mkl_metrics {
  declare -r log=$1
  mkl_success=$(grep "mkl_success" $log | awk '{print $3}')

  if [ "$mkl_success" == "no" ]; then
    mkl_err=$(grep "mkl_err" $log | awk '{print $3}')
    mkl_err_accum=$(bc -l <<< "$mkl_err_accum + $mkl_err")
    return
  fi

  mkl_rel_err=$(grep "mkl_rel_err" $log | awk '{print $3}')
  mkl_iters=$(grep "mkl_iters" $log | awk '{print $3}')
  etime=$(grep "mkl_time" $log | awk '{print $3}')
  etime=$(bc -l <<< "$etime*1000")

  mkl_time_accum=$(bc -l <<< "$mkl_time_accum + $etime")
  mkl_rel_err_accum=$(bc -l <<< "$mkl_rel_err_accum + $mkl_rel_err")
  mkl_iters_accum=$(bc -l <<< "$mkl_iters_accum + $mkl_iters")
  mkl_succeed=$((mkl_succeed + 1))
}

function get_eigen_metrics {
  declare -r log=$1
  eigen_success=$(grep "eigen_success" $log | awk '{print $3}')

  if [ "$eigen_success" == "no" ]; then
    eigen_err=$(grep "eigen_err" $log | awk '{print $3}')
    eigen_err_accum=$(bc -l <<< "$eigen_err_accum + $eigen_err")
    return
  fi

  eigen_rel_err=$(grep "eigen_rel_err" $log | awk '{print $3}')
  eigen_iters=$(grep "eigen_iters" $log | awk '{print $3}')
  etime="$(grep "eigen_time" $log | awk '{print $3}')"

  eigen_time_accum=$(bc -l <<< "$eigen_time_accum + $etime")
  eigen_rel_err_accum=$(bc -l <<< "$eigen_rel_err_accum + $eigen_rel_err")
  eigen_iters_accum=$(bc -l <<< "$eigen_iters_accum + $eigen_iters")
  eigen_succeed=$((eigen_succeed + 1))
}

function main {
  mkdir -p $odir
  for log in $(ls $log_dir/out*); do
    get_jacobi_metrics $log
    get_mkl_metrics $log
    get_eigen_metrics $log
  done

  if [[ $jacobi_succeed > 0 ]]; then
    echo $n $(bc -l <<< "scale=3;$jacobi_succeed/$num_runnings") >> $odir/jacobi_succeed.dat
    echo $n $(bc -l <<< "scale=3;$jacobi_time_accum/$jacobi_succeed") >> $odir/jacobi_time.dat
    echo $n $(bc -l <<< "$jacobi_rel_err_accum/$jacobi_succeed") >> $odir/jacobi_err.dat
    echo $n $(bc -l <<< "scale=3;$jacobi_iters_accum/$jacobi_succeed") >> $odir/jacobi_iters.dat
  fi

  if [[ $jacobi_succeed < $num_runnings ]]; then
    jacobi_fails=$((num_runnings - jacobi_succeed))
    echo $n $(bc -l <<< "scale=3;$jacobi_err_accum/$jacobi_fails") >> $odir/jacobi_failed_err.dat
  fi

  if [[ $mkl_succeed > 0 ]]; then
    echo $n $(bc -l <<< "scale=3;$mkl_succeed/$num_runnings") >> $odir/mkl_succeed.dat
    echo $n $(bc -l <<< "scale=3;$mkl_time_accum/$mkl_succeed") >> $odir/mkl_time.dat
    echo $n $(bc -l <<< "$mkl_rel_err_accum/$mkl_succeed") >> $odir/mkl_err.dat
    echo $n $(bc -l <<< "scale=3;$mkl_iters_accum/$mkl_succeed") >> $odir/mkl_iters.dat
  fi

  if [[ $mkl_succeed < $num_runnings ]]; then
    mkl_fails=$((num_runnings - mkl_succeed))
    echo $n $(bc -l <<< "scale=3;$mkl_err_accum/$mkl_fails") >> $odir/mkl_failed_err.dat
  fi

  if [[ $eigen_succeed > 0 ]]; then
    echo $n $(bc -l <<< "scale=3;$eigen_succeed/$num_runnings") >> $odir/eigen_succeed.dat
    echo $n $(bc -l <<< "scale=3;$eigen_time_accum/$eigen_succeed") >> $odir/eigen_time.dat
    echo $n $(bc -l <<< "$eigen_rel_err_accum/$eigen_succeed") >> $odir/eigen_err.dat
    echo $n $(bc -l <<< "scale=3;$eigen_iters_accum/$eigen_succeed") >> $odir/eigen_iters.dat
  fi

  if [[ $eigen_succeed < $num_runnings ]]; then
    eigen_fails=$((num_runnings - eigen_succeed))
    echo $n $(bc -l <<< "scale=3;$eigen_err_accum/$eigen_fails") >> $odir/eigen_failed_err.dat
  fi

}

main
exit 0
