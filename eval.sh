#!/bin/bash
# New York, Chicago, Los Angeles, San Francisco
for city in "NY" "CH" "LA" "SF" "ALL"
do
	echo "`cut -f3,4,5 result_newfinal/${city}Q5Cate.result | python mrr.py | cut -d\  -f2,4,6` `cut -f3,4,5 result_newfinal/${city}Q5Base.result | python mrr.py | cut -d\  -f2,4,6`"
done

