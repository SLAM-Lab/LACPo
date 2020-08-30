echo "$1"
grep " $3" $1 | head -1
#grep -e Time -e "^$3" $1 > $2.temp.parsed.out
grep -e Time -e "^$3 " $1 > $2.temp.parsed.out
perl  parsedout_to_csv.pl $2.temp.parsed.out $2.temp.parsed.csv
rm $2.temp.parsed.out
mv $2.temp.parsed.csv $2

