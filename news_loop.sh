filename="partitions.txt"
while read -r line; do
    name="$line"
    python news.py $name
done  < "$filename"
