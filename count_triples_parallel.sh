for j in 0 `seq 9`
do 
  for part in 0 `seq 9`
  do 
    time nice python count_triples_single_thread.py $j${part}
  done & 
done
