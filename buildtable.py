import datetime
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def build_table():

    #### dates to crop the data for the learning range ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    starting_date = "2019-03-29"
    ending_date = "2021-11-18"

    #### Function that extract data from raw daily indicator csv from Alphavantage API + cropping for desired dates, output is pd_dataframe.
    def import_indicator_csv_to_pd(csv_file_name, start_date, end_date):
        # add a stop condition if first or last date of the csv are < to one of the edge dates
        csv_file = pd.read_csv(csv_file_name+'.csv')
        csv_file['Date'] = pd.to_datetime(csv_file['Date'], format='%Y-%m-%d')
        indexnames = csv_file[csv_file['Date'] < datetime.datetime.strptime(start_date, '%Y-%m-%d') ].index
        indexnames = indexnames.append(csv_file[csv_file['Date'] > datetime.datetime.strptime(end_date, '%Y-%m-%d') ].index)
        csv_file.drop(indexnames, inplace=True)
        csv_file.reset_index(drop=True, inplace=True)
        return csv_file

    #### GET DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    test_sample = pd.read_csv('csv/BTCDAILY.csv')
    test_sample = test_sample.reindex(index=test_sample.index[::-1])
    #test_sample = test_sample[12:70] ####To remove~- minimize sample to be faster coding

    #### Extract relevant time data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    day = []
    month = []

    for i in range(0,test_sample.shape[0]):

        day.append(datetime.datetime.strptime(test_sample.iloc[i, 1], '%Y-%m-%dT%H:%M:%S.%f0Z').strftime('%a'))
        month.append(datetime.datetime.strptime(test_sample.iloc[i, 1], '%Y-%m-%dT%H:%M:%S.%f0Z').strftime('%m'))

    test_sample["day"] = day
    test_sample['day'].replace(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],[1,2,3,4,5,6,7], inplace=True)
    test_sample["month"] = month
    test_sample['month'] = test_sample['month'].apply(int)

    #### Rearrange date and convert into '%Y-%m-%d'

    test_sample['time_period_start'] = pd.to_datetime(test_sample['time_period_start'], format='%Y-%m-%dT%H:%M:%S.%f0Z')
    test_sample['Date'] = test_sample['time_period_start']
    test_sample.drop(['time_period_start', 'time_period_end', 'time_open', 'time_close'], axis = 1, inplace=True)
    test_sample = test_sample[['Date', 'price_open', 'price_high', 'price_low', 'price_close', 'volume_traded', 'trades_count', 'day', 'month' ]]
    test_sample.reset_index(drop=True, inplace=True)

    #### Crop dates outside range
    test_sample.to_csv("csv/test_sample.csv", index=False)
    test_sample = import_indicator_csv_to_pd("csv/test_sample", starting_date, ending_date)

    #### Import INDICATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    sma20 = import_indicator_csv_to_pd("csv/BTCDAILYSMA20", starting_date, ending_date)
    sma8 = import_indicator_csv_to_pd("csv/BTCDAILYSMA8", starting_date, ending_date)
    rsi14 = import_indicator_csv_to_pd("csv/BTCDAILYRSI14", starting_date, ending_date)
    macd12269 = import_indicator_csv_to_pd("csv/BTCDAILYMACD12269", starting_date, ending_date)
    rsi9 = import_indicator_csv_to_pd("csv/BTCDAILYRSI9", starting_date, ending_date)
    rsi25 = import_indicator_csv_to_pd("csv/BTCDAILYRSI25", starting_date, ending_date)
    dxy = import_indicator_csv_to_pd("csv/DXYDAILY", starting_date, ending_date)
    gold = import_indicator_csv_to_pd("csv/DAILYGOLD", starting_date, ending_date)
    gtrend_btc = import_indicator_csv_to_pd("csv/GTRENDBTCDAILY", starting_date, ending_date)
    GNFRM = import_indicator_csv_to_pd("csv/GNFRM", starting_date, ending_date)
    GNHRate = import_indicator_csv_to_pd("csv/GNHRate", starting_date, ending_date)
    GNNetRealProfitLoss = import_indicator_csv_to_pd("csv/GNNetRealProfitLoss", starting_date, ending_date)
    GNPUELLM = import_indicator_csv_to_pd("csv/GNPUELLM", starting_date, ending_date)
    GNRealLoss = import_indicator_csv_to_pd("csv/GNREAL_LOSS", starting_date, ending_date)
    GNRealPro = import_indicator_csv_to_pd("csv/GNREAL_PRO", starting_date, ending_date)
    GNRealProfitsToValueRatio = import_indicator_csv_to_pd("csv/GNRealProfitsToValueRatio", starting_date, ending_date)
    GNSOPR = import_indicator_csv_to_pd("csv/GNSOPR", starting_date, ending_date)
    GNSupplyProfit = import_indicator_csv_to_pd("csv/GNSupplyProfit", starting_date, ending_date)
    GNSVL124 = import_indicator_csv_to_pd("csv/GNSVL124", starting_date, ending_date)
    GNSVLDW = import_indicator_csv_to_pd("csv/GNSVLDW", starting_date, ending_date)
    GNTransacSize = import_indicator_csv_to_pd("csv/GNTransacSize", starting_date, ending_date)

    #### merge INDICATORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    test_sample= test_sample.join(sma20['SMA_20'])
    test_sample = test_sample.join(sma8['SMA_8'])
    test_sample= test_sample.join(rsi14['RSI'])
    test_sample = test_sample.join(rsi9['RSI9'])
    test_sample = test_sample.join(rsi25['RSI25'])
    test_sample= test_sample.join(macd12269['MACD_Hist'])
    test_sample = test_sample.join(dxy['DXY'])
    test_sample = test_sample.join(gold['GOLD'])
    test_sample = test_sample.join(gtrend_btc['GTREND_BTC'])
    test_sample = test_sample.join(GNFRM['GNFRM'])
    test_sample = test_sample.join(GNHRate['GNHRate'])
    test_sample = test_sample.join(GNNetRealProfitLoss['GNNetRealProfitLoss'])
    test_sample = test_sample.join(GNPUELLM['GNPUELLM'])
    test_sample = test_sample.join(GNRealLoss['REAL_LOSS'])
    test_sample = test_sample.join(GNRealPro['REAL_PRO'])
    test_sample = test_sample.join(GNRealProfitsToValueRatio['GNRealProfitsToValueRatio'])
    test_sample = test_sample.join(GNSOPR['GNSOPR'])
    test_sample = test_sample.join(GNSupplyProfit['GNSupplyProfit'])
    test_sample = test_sample.join(GNSVL124['SVL124'])
    test_sample = test_sample.join(GNSVLDW['SVLDW'])
    test_sample = test_sample.join(GNTransacSize['GNTransacSize'])

    #### Add volatility data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    volatility = []

    for i in range(0,test_sample.shape[0]):
        volatility.append((test_sample['price_high'].iloc[i] - test_sample['price_low'].iloc[i])/test_sample['price_open'].iloc[i]*100)
    test_sample["volatility"] = volatility
    #### Add candle representation + market direction ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    candle_rep = []
    market_dir = []

    for i in range(0,test_sample.shape[0]):
        candle_rep.append((test_sample['price_close'].iloc[i] - test_sample['price_open'].iloc[i])/test_sample['price_open'].iloc[i]*100)
    test_sample['candle_rep'] = candle_rep

    for i in range(0,test_sample.shape[0]):
        market_dir.append(0)
    test_sample.loc[test_sample["candle_rep"] >= 0, "market_dir"] = 1
    test_sample.loc[test_sample["candle_rep"] < 0, "market_dir"] = -1

    #### Add up and down volatility~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    volatility_up = []
    volatility_down = []

    for i in range(0,test_sample.shape[0]):
        if test_sample['market_dir'].iloc[i] == 1:
            volatility_up.append((test_sample['price_high'].iloc[i] - test_sample['price_close'].iloc[i]) / test_sample['price_open'].iloc[i] * 100)
            volatility_down.append((test_sample['price_low'].iloc[i] - test_sample['price_open'].iloc[i]) / test_sample['price_open'].iloc[i] * 100)
        elif test_sample['market_dir'].iloc[i] == -1:
            volatility_up.append((test_sample['price_high'].iloc[i] - test_sample['price_open'].iloc[i]) / test_sample['price_open'].iloc[i] * 100)
            volatility_down.append((test_sample['price_low'].iloc[i] - test_sample['price_close'].iloc[i]) / test_sample['price_open'].iloc[i] * 100)

        else:   print("error format market_dir or wrong column called")

    test_sample["volatility_up"] = volatility_up
    test_sample["volatility_down"] = volatility_down

    #### Calculate SMA-price/price to normalize SMA value~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    sma20_norm = []

    for i in range(0,test_sample.shape[0]):
        sma20_norm.append((test_sample['price_close'].iloc[i] - test_sample['SMA_20'].iloc[i])/test_sample['price_close'].iloc[i]*100)

    test_sample["sma20_norm"] = sma20_norm

    sma8_norm = []

    for i in range(0,test_sample.shape[0]):
        sma8_norm.append((test_sample['price_close'].iloc[i] - test_sample['SMA_8'].iloc[i])/test_sample['price_close'].iloc[i]*100)

    test_sample["sma8_norm"] = sma8_norm

    sma_diff_20_8 = []

    for i in range(0, test_sample.shape[0]):
        sma_diff_20_8.append(
            (test_sample['SMA_20'].iloc[i] - test_sample['SMA_8'].iloc[i]) / test_sample['SMA_20'].iloc[i] * 100)

    test_sample["sma_diff_20_8"] = sma_diff_20_8


    #### add avg btc/trade adjusted on price = mic-mac ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    btc_per_trade_adjusted = [0]

    for i in range(1,test_sample.shape[0]):
        btc_per_trade_adjusted.append(test_sample['volume_traded'].iloc[i] / test_sample['trades_count'].iloc[i]*test_sample['price_open'].iloc[i])
    test_sample['btc_per_trade_adjusted'] = btc_per_trade_adjusted

    #### add volume change ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    volume_change = [0]

    for i in range(1,test_sample.shape[0]):
        volume_change.append((test_sample['volume_traded'].iloc[i] - test_sample['volume_traded'].iloc[i-1])/test_sample['volume_traded'].iloc[i]*100)
    test_sample['volume_change'] = volume_change

    #### add tradecount change and delete first row (no calculus possible for 1st row for volume & tradecount )~~~~
    trade_count_change = [0]

    for i in range(1,test_sample.shape[0]):
        trade_count_change.append((test_sample['trades_count'].iloc[i] - test_sample['trades_count'].iloc[i-1])/test_sample['trades_count'].iloc[i]*100)
    test_sample['trade_count_change'] = trade_count_change

    #### Clean up ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    test_sample.drop(0, inplace=True)
    test_sample.reset_index(drop=True, inplace=True)

    ## selectable data to add to the set
    #   'Date','market_dir', 'volume_traded'
    # , 'GNHRate', 'GNNetRealProfitLoss', 'GNRealProfitsToValueRatio','GNSupplyProfit', 'SVL124', 'GNTransacSize'
    # , 'GNPUELLM', 'REAL_LOSS', 'REAL_PRO', 'GNRealProfitsToValueRatio', 'GNSOPR', 'GNSupplyProfit', 'SVL124', 'SVLDW', 'GNTransacSize'
    # ,'price_low', 'price_high', 'price_open',  'trade_count_change', 'btc_per_trade_adjusted', 'RSI9', 'RSI25',  'DXY', 'GOLD', 'GTREND_BTC','month'
    # , 'sma8_norm', 'sma20_norm', 'sma_diff_20_8', 'volume_change', 'volatility_down', 'MACD_Hist'
    # , 'trades_count', 'SMA_20', 'SMA_8','candle_rep', 'price_close', 'volatility', 'volatility_up', 'day', 'RSI'

    ## select data that will be in the set
    data = test_sample[['candle_rep', 'price_close', 'volatility', 'volatility_up', 'day', 'RSI', 'sma8_norm']]
    data.to_csv("csv/prepared_data.csv", index=False)

    ## Visualize input dataset
    tab = pd.DataFrame()
    tab['MIN'] = data.min()
    tab['MAX'] = data.max()
    tab['MEAN'] = data.mean()
    tab['MEDIAN'] = data.median()
    tab['STD'] = data.std()
    tab = tab.round(2)
    print(tab)
    print(data.shape[1])

    return

build_table()
