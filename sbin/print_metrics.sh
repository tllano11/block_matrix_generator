#!/bin/bash

declare -r dir=$1
declare -r n=$2
declare -r odir=$1/dat
declare -r subdir=$dir/$n
declare -r num_runnings=$3

function get_jacobi_metrics {
  jacobi_succeed=$(grep -R "jacobi_success = yes" $subdir | wc -l)
  jacobi_fails=$((num_runnings - jacobi_succeed))

  if (( $jacobi_succeed > 0 )); then
    files=($(grep -R -l "jacobi_success = yes" $subdir))
    jacobi_time_avg=$(awk -v n="$jacobi_succeed" \
                          '/run_jacobi/ || /compute_error/ || /cublas/ {sum+=$5} END {print sum/n}' \
                          ${files[@]})

#    jacobi_rel_err_avg=$(awk -v n="$jacobi_succeed" \
#                             '/jacobi_rel_err/ {sum+=$3} END {print sum/n}' \
#                             ${files[@]})

#    jacobi_iters_avg=$(awk -v n="$jacobi_succeed" \
#                           '/jacobi_iters/ {sum+=$3} END {print sum/n}' \
#                           ${files[@]})

    echo $n $(awk -v success=$jacobi_succeed -v total=$num_runnings \
                  'BEGIN {print success/total}') >> $odir/jacobi_succeed.dat
    echo $n $jacobi_time_avg >> $odir/jacobi_time.dat
#    echo $n $jacobi_rel_err_avg >> $odir/jacobi_rel_err.dat
#    echo $n $jacobi_iters_avg >> $odir/jacobi_iters.dat
  fi

#  if (( $jacobi_succeed < $num_runnings )); then
#    files=($(grep -R -l "jacobi_success = no" $subdir))
#    jacobi_err=$(awk -v n="$jacobi_fails" \
#                     '/jacobi_err/ {sum+=$3} END {print sum/n}' \
#                     ${files[@]})
#    echo $n $jacobi_err >> $odir/jacobi_failed_err.dat
#  fi
}

function get_jacobi_cpu_metrics {
  jacobi_succeed=$(grep -R "jacobi_cpu_success = yes" $subdir | wc -l)
  jacobi_fails=$((num_runnings - jacobi_succeed))

  if (( $jacobi_succeed > 0 )); then
    files=($(grep -R -l "jacobi_cpu_success = yes" $subdir))
    jacobi_time_avg=$(awk -v n="$jacobi_succeed" \
                          '/jacobi_cpu_time/ {sum+=$3} END {print sum/n}' ${files[@]})

    echo $n $(awk -v success=$jacobi_succeed -v total=$num_runnings \
                  'BEGIN {print success/total}') >> $odir/jacobi_cpu_succeed.dat
    echo $n $jacobi_time_avg >> $odir/jacobi_cpu_time.dat
  fi

#  if (( $jacobi_succeed < $num_runnings )); then
#    files=($(grep -R -l "jacobi_cpu_success = no" $subdir))
#    jacobi_err=$(awk -v n="$jacobi_fails" \
#                     '/jacobi_cpu_err/ {sum+=$3} END {print sum/n}' \
#                     ${files[@]})
#    echo $n $jacobi_err >> $odir/jacobi_cpu_failed_err.dat
#  fi
}

function get_mkl_metrics {
  mkl_succeed=$(grep -R "mkl_success = yes" $subdir | wc -l)
  mkl_fails=$((num_runnings - mkl_succeed))

  if (( $mkl_succeed > 0 )); then
    files=($(grep -R -l "mkl_success = yes" $subdir))
    mkl_time_avg=$(awk -v n="$mkl_succeed" \
                       '/mkl_time/ {sum+=$3} END {print (sum*1000)/n}' \
                       ${files[@]})

    mkl_rel_err_avg=$(awk -v n="$mkl_succeed" \
                          '/mkl_rel_err/ {sum+=$3} END {print sum/n}' \
                          ${files[@]})

    mkl_iters_avg=$(awk -v n="$mkl_succeed" \
                        '/mkl_iters/ {sum+=$3} END {print sum/n}' \
                        ${files[@]})

    echo $n $(awk -v success=$mkl_succeed -v total=$num_runnings \
                  'BEGIN {print success/total}') >> $odir/mkl_succeed.dat
    echo $n $mkl_time_avg >> $odir/mkl_time.dat
    echo $n $mkl_rel_err_avg >> $odir/mkl_rel_err.dat
    echo $n $mkl_iters_avg >> $odir/mkl_iters.dat
  fi

  if (( $mkl_succeed < $num_runnings )); then
    files=($(grep -R -l "mkl_success = no" $subdir))
    mkl_err=$(awk -v n="$mkl_fails" \
                  '/mkl_err/ {sum+=$3} END {print sum/n}' \
                  ${files[@]})
    echo $n $mkl_err >> $odir/mkl_failed_err.dat
  fi
}

function get_eigen_metrics {
  eigen_succeed=$(grep -R "eigen_success = yes" $subdir | wc -l)
  eigen_fails=$((num_runnings - eigen_succeed))

  if (( $eigen_succeed > 0 )); then
    files=($(grep -R -l "eigen_success = yes" $subdir))
    eigen_time_avg=$(awk -v n="$eigen_succeed" \
                         '/eigen_time/ {sum+=$3} END {print sum/n}' \
                         ${files[@]})

    eigen_rel_err_avg=$(awk -v n="$eigen_succeed" \
                            '/eigen_rel_err/ {sum+=$3} END {print sum/n}' \
                            ${files[@]})

    eigen_iters_avg=$(awk -v n="$eigen_succeed" \
                          '/eigen_iters/ {sum+=$3} END {print sum/n}' \
                          ${files[@]})

    echo $n $(awk -v success=$eigen_succeed -v total=$num_runnings \
                  'BEGIN {print success/total}') >> $odir/eigen_succeed.dat
    echo $n $eigen_time_avg >> $odir/eigen_time.dat
    echo $n $eigen_rel_err_avg >> $odir/eigen_rel_err.dat
    echo $n $eigen_iters_avg >> $odir/eigen_iters.dat
  fi

  if (( $eigen_succeed < $num_runnings )); then
    files=($(grep -R -l "eigen_success = no" $subdir))
    eigen_err=$(awk -v n="$eigen_fails" \
                    '/eigen_err/ {sum+=$3} END {print sum/n}' \
                    ${files[@]})
    echo $n $eigen_err >> $odir/eigen_failed_err.dat
  fi
}

function main {
  get_jacobi_metrics
  get_jacobi_cpu_metrics
#  get_mkl_metrics
#  get_eigen_metrics
}

main
exit 0
