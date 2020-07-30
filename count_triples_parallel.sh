for j in 0 `seq 9`
do 
  for part in 00 `seq -w 99`
  do 
    time nice python count_triples_pyconll.py $j${part}
  done & 
done
