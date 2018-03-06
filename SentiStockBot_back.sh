rm hypothetical_output.txt
rm hypothetical_test.txt 
echo "Please enter you hypothetical comments:"
read in 

echo $in>>hypothetical_test.txt

python SentimentTest.py

echo "Your text:" >> hypothetical_output.txt
cat hypothetical_test.txt >> hypothetical_output.txt
echo  "-------------------------------------------------------------------------------------------" >> hypothetical_output.txt
echo "On the day of the event, stock price change (in %):" >> hypothetical_output.txt
python mainModel.py Day0.csv 1 Day0_pred.csv Day0.png hypothetical_sentiment.csv hypothetical_output.txt 1000 1

echo "On the day 1 after the event, stock price change (in %):" >> hypothetical_output.txt
python mainModel.py Day1.csv 1 Day1_pred.csv Day1.png hypothetical_sentiment.csv hypothetical_output.txt 1000 1

echo "On the day 2 after the event, stock price change (in %):" >> hypothetical_output.txt
python mainModel.py Day2.csv 1 Day2_pred.csv Day2.png hypothetical_sentiment.csv hypothetical_output.txt 1000 1

gsutil cp -r Day?_pred.csv gs://ml_bucket_rwa
clear
cat hypothetical_output.txt


