# BTCP_project
This project aim at creating an efficient neural network able to predict future BTC price using daily time series data

The base model is an LSTM NN, using 2 LSTM layers and as many input values as available.

Data range from 2019-03-29 to 2021-11-18.

The best results where obtained using daily % of change on the bitcoin instead of real price, it avoids autocorrelation phenomenon.

n_day_used_to_predict is the number of days in the past to look at to make the prediction

n_days_in_the_future is the number of days ahead the NN will try to predict. if = 0 NN will predict the day right after the n_day_used_to_predict sequence. 

Data are obtained through diverse API and converted to a CSV file with 2 columns : Date ('%Y-%m-%d') and value (float).

First results are encouraging as the NN was able to predict with great accuracy a punctual event of 5 days (see picture in the encouraging results folder).

GlassNode data were tested with little success.

More data can be tested, I'm thinking of sentiment data from Twitter and social network. Wich I do not have and failed to find.

I also didn't test premium GlassNode data, as I onmy have a regular paid plan.

Gtrend daily data with the word "bitcoin" are also disappointing. Other keywords could be tested.

In order to dismiss the auto-corelation phenomenon, I also tried to implement an attention mechanism (SetNeuralNetworkAttention()). It gave me few results and I'm not sure I'm using it the best way possible as I struggle to deeply understand how it works. 

My code is not perfect, I'm a MD not an engineer 😅 but I try my best ! Feel free to correct my mistakes and any help to improve the model/code would be appreciated. 

To resume, we need to 1/perfect the model, 2/improve the quantity of data 3/ extend the time range of the learning set 4/try, try, and try again.

With several brains I think we can make something work. I obtained a 60% up/down accuracy with some runs...

Bitcoin may not be predictible, but humans are !
