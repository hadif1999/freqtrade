#!/bin/bash

strategy_name="KalmanFilterStrategy"
fee=0.001
start_time="20240201"
timeframe="5m" 
config_file="user_data/config.json"
plot_result=true

output_folder=${PWD}/user_data/backtest_results/${strategy_name}_backtest_results
mkdir -p $output_folder
project_path=$PWD

echo -e "\ndownloading data: \n"
freqtrade download-data --timerange ${start_time}- -t $timeframe -c $config_file  
echo -e "\n****************************************************\n"

echo -e "\nrunning backtest: \n"
freqtrade backtesting --fee $fee -s $strategy_name --timerange ${start_time}- -c $config_file --export=signals 
# --enable-protections 
echo -e "\n****************************************************\n"

cd $output_folder
# removing previous files
rm $(ls|grep group)
rm $(ls|grep indicators)
cd $project_path

echo -e "\nbacktest analysis: \n"
freqtrade backtesting-analysis -c $config_file --analysis-groups 0 1 2 3 4 --indicator-list open_date close_date profit_abs close profit_ratio --analysis-to-csv --analysis-csv-path $output_folder 
echo -e "\n****************************************************\n"


exit

old_plot_dir=${project_path}/user_data/plot
# deleting old plots
cd $old_plot_dir
rm -r $(ls)
cd $project_path

echo -e "\nploting backtest results: \n"
freqtrade plot-dataframe -s $strategy_name --timerange ${start_time}- -c $config_file 
echo -e "\n****************************************************\n"

mkdir -p ${output_folder}/plot
cd $old_plot_dir
for plot in $(ls)
do 
	mv plot ${output_folder}/plot
	if $plot_result = true
	then
		open plot
	fi
done
