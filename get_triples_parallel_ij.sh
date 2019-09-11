for j in `seq 9`
do 
  for part in `seq 9`
  do 
    time nice python get_DepCC.py $j${part}
  done & 
done
