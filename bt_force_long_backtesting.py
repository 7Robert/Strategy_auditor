import math
import logging
import numpy as np
import pandas as pd
from colorama import *
from itertools import groupby
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from stockstats import StockDataFrame as Sdf
from bokeh.models import ColumnDataSource, HoverTool, CrosshairTool

# --- This script is not finalized yet --- #

# Logging to evaluate events.
log_format = '%(levelname)s %(asctime)s - %(message)s'
logging.basicConfig(filename='long_status.Log', level=logging.DEBUG, format=log_format, filemode='w')
logger = logging.getLogger()

# Visual configurations
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None

# -- Options to configure the program -- #

# Data options
TIME = ' 4 H'  # resample size  | T = minutes, H = hours, D = days
DATA = pd.read_csv('cl_1m.csv')
DATA = DATA.rename(columns={'timestamp': 'date'})

DATA = DATA[(DATA['date'] >= '2013-02-01T00:00:00') & (DATA['date'] <= '2017-12-05T24:00:00')]

# Number of times the function strategy has been executed
global x
x = 0

# Indicators
EMA = 8
WMA = 30
ATR_COEF = 2

# Parameters
FEE = 1.52
INIT_CAP = 100000
SLIPPAGE_DEF = 0.01

TICK_SIZE = 0.01
TICK_VALUE = 10
CONTRACT_SIZE = 1000

# --- code ---- #


def resample_data(data: pd.DataFrame, time: str) -> pd.DataFrame:
    """
    :param data: Historical data in minutes [Open high low close]
    :param time: String with the temporality of the data that you want to obtain
     T = minutes, H = hours, D = days
    :return: Dataframe with data resampled
    """

    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data.sort_index(inplace=True)

    # Converting to OHLC format

    data_ohlc = data.resample(time).apply({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                                           'volume': 'sum'})
    data_ohlc.dropna(inplace=True)
    data_ohlc = data_ohlc.reset_index()

    logger.info('Resample of the data was successful')

    return data_ohlc


def strategy(data: pd.DataFrame, x: int) -> tuple:
    """
    :param data: DataFrame with resampled data
    :param x: Number of times the function has been executed
    :return: tuple with trades, capital line and data with indicators calculated
    """

    df_data = data
    df_data = Sdf.retype(df_data)
    df_data.reset_index(inplace=True)
    df_data.date = pd.to_datetime(df_data.date)

    # ATR
    df_data['atr'] = df_data['atr']

    # Ema
    df_data['ema'] = df_data[f'open_{EMA}_ema']

    # WMA
    df_data.dropna(inplace=True)
    df_data.reset_index(inplace=True)

    def wma(prices, period, df_data):

        num_prices = len(prices)

        if num_prices < period:
            raise SystemExit('Error: num_prices < period')

        wma_range = num_prices - period + 1
        wmas = np.zeros(wma_range)
        values_wma = np.zeros(period - 1)

        k = (period * (period + 1)) / 2.0

        for idx in range(wma_range):
            for period_num in range(period):
                weight = period_num + 1
                wmas[idx] += prices[idx + period_num] * weight

            wmas[idx] /= k

        wmas = np.concatenate([values_wma, wmas])
        df_wma = pd.DataFrame(wmas)
        df_wma = df_wma.rename(columns=({0: 'wma'}))
        df_data = df_data.join(df_wma)

        return df_data

    df_data = wma(prices=df_data.close, period=WMA, df_data=df_data)

    condition = False

    capital_line = [100000]
    capital = INIT_CAP
    df_trades = pd.DataFrame()

    for i in range(WMA, len(df_data)):

        # Buy condition
        if condition is False and df_data.ema[i] > df_data.wma[i] and df_data.close[i] < df_data.ema[i]:

            if i + 1 < len(df_data):

                buy_atr = df_data.atr[i]
                buy_price = df_data.open[i + 1] + SLIPPAGE_DEF
                p_size_contracts = np.ceil((capital * 0.01) / (ATR_COEF * buy_atr * (1 / TICK_SIZE) * TICK_VALUE))

                margin = 5600
                margin_total = margin * p_size_contracts

                if margin_total > capital:
                    p_size_contracts = np.floor(capital / margin)
                    if p_size_contracts == 0:
                        break

                p_size_usd_buy = (p_size_contracts * CONTRACT_SIZE) * buy_price
                total_fee = (FEE * 2) * p_size_contracts
                buy_date = df_data.date[i]

                condition = True

                dfm_aux = pd.DataFrame([{'buy_date': buy_date, 'sell_date': '', 'buy_price': buy_price,
                                         'sell_price': np.NaN, 'quantity': p_size_contracts, 'fee': total_fee,
                                         'trade': '', 'capital': '', 'mae': '', 'quantity_active': p_size_usd_buy,
                                         'incurred_sp': '', 'pl': ''}])

                df_trades = df_trades.append(dfm_aux)
                df_trades.reset_index(inplace=True, drop=True)

                if x == 0:
                    print(Fore.GREEN + "\n[BUY] " + Fore.WHITE + "Details: Quantity: {} contracts | Price: {} usd"
                          .format(int(p_size_contracts), round(buy_price, 2)))

        # Stop loss condition
        if condition is True and df_data['date'][i] != buy_date:

            if df_data.low[i] <= buy_price - (ATR_COEF * buy_atr):

                if i + 1 < len(df_data):

                    sell_price = df_data.open[i + 1] - SLIPPAGE_DEF
                    p_size_usd_sell = (p_size_contracts * CONTRACT_SIZE) * sell_price
                    total = p_size_usd_sell - p_size_usd_buy - total_fee
                    sell_date = df_data.date[i]
                    capital += total
                    condition = False

                    capital = round(capital, 2)
                    sell_price = round(sell_price, 2)
                    total_fee = round(total_fee, 2)
                    p_size_usd_buy = round(p_size_usd_buy, 2)
                    p_size_usd_sell = round(p_size_usd_sell, 2)
                    total = round(total, 2)

                    capital_line.append(capital)

                    # Incurred Slippage
                    incurred_sp = 2 * p_size_contracts * CONTRACT_SIZE * SLIPPAGE_DEF

                    df_trades.at[df_trades.index[-1], 'sell_date'] = sell_date
                    df_trades.at[df_trades.index[-1], 'sell_price'] = sell_price
                    df_trades.at[df_trades.index[-1], 'trade'] = 'stop loss'
                    df_trades.at[df_trades.index[-1], 'capital'] = capital_line[-1]
                    df_trades.at[df_trades.index[-1], 'mae'] = 0
                    df_trades.at[df_trades.index[-1], 'incurred_sp'] = incurred_sp
                    df_trades.at[df_trades.index[-1], 'pl'] = total

                    df_trades.reset_index(inplace=True, drop=True)

                if x == 0:
                        print(Fore.RED + "[STOP LOSS] " + Fore.WHITE + "Details: Quantity: {} contracts| Price: {} usd\n"
                              .format(int(p_size_contracts), round(sell_price, 2)))

                        print("---------- Trade details ----------")
                        print(f'p size usd buy: {p_size_usd_buy} usd')
                        print(f'p size usd sell: {p_size_usd_sell} usd')
                        print(f'fee: {total_fee} usd')
                        print(f'result trade: {total} usd')
                        print(f'capital: {capital} usd\n')
                        print(Fore.GREEN + 'buy date: ' + Fore.WHITE + f'{buy_date}')
                        print(Fore.RED + 'sell date: ' + Fore.WHITE + f'{sell_date}')
                        print("------------------------------------")

        # Sell condition
        if condition is True and df_data['date'][i] != buy_date:

            if df_data.ema[i - 1] > df_data.wma[i - 1] and df_data.ema[i] < df_data.wma[i]:

                if i + 1 < len(df_data):

                    sell_price = df_data.open[i + 1] - SLIPPAGE_DEF
                    p_size_usd_sell = (p_size_contracts * CONTRACT_SIZE) * sell_price
                    total = p_size_usd_sell - p_size_usd_buy - total_fee
                    sell_date = df_data.date[i]
                    capital += total
                    condition = False

                    capital = round(capital, 2)
                    total_fee = round(total_fee, 2)
                    p_size_usd_buy = round(p_size_usd_buy, 2)
                    p_size_usd_sell = round(p_size_usd_sell, 2)
                    total = round(total, 2)

                    capital_line.append(capital)

                    # Here mae is calculated
                    data_aux = df_data[(df_data['date'] >= f'{buy_date}') & (df_data['date'] <= f'{sell_date}')]
                    min_price = min(data_aux.low)
                    mae = (min_price / buy_price) - 1

                    # Incurred Slippage
                    incurred_sp = 2 * p_size_contracts * CONTRACT_SIZE * SLIPPAGE_DEF

                    df_trades.at[df_trades.index[-1], 'sell_date'] = sell_date
                    df_trades.at[df_trades.index[-1], 'sell_price'] = sell_price
                    df_trades.at[df_trades.index[-1], 'trade'] = 'strategy'
                    df_trades.at[df_trades.index[-1], 'capital'] = capital_line[-1]
                    df_trades.at[df_trades.index[-1], 'mae'] = mae
                    df_trades.at[df_trades.index[-1], 'incurred_sp'] = incurred_sp
                    df_trades.at[df_trades.index[-1], 'pl'] = total

                    df_trades.reset_index(inplace=True, drop=True)

                    if x == 0:
                        print(Fore.RED + "[SELL] " + Fore.WHITE + "Details: Quantity: {} contracts| Price: {} usd\n"
                              .format(int(p_size_contracts), round(sell_price, 2)))

                        print("---------- Trade details ----------")
                        print(f'p size usd buy: {p_size_usd_buy} usd')
                        print(f'p size usd sell: {p_size_usd_sell} usd')
                        print(f'fee: {total_fee} usd')
                        print(f'result trade: {total} usd')
                        print(f'capital: {capital} usd\n')
                        print(Fore.GREEN + 'buy date: ' + Fore.WHITE + f'{buy_date}')
                        print(Fore.RED + 'sell date: ' + Fore.WHITE + f'{sell_date}')
                        print("------------------------------------")

    logger.info('Correctly created DataFrame with strategy trades')
    df_trades = df_trades[np.isfinite(df_trades['sell_price'])]
    df_trades.reset_index(inplace=True, drop=True)
    return df_trades, capital_line, df_data


def statistics(trades: pd.DataFrame, capital_line: list) -> tuple:
    """
    :param trades: Dataframe with trades
    :param capital_line: Capital line of trades

    :return: Drawdown, positive and negative trades
    """
    if len(trades) > 1:

        # Drawdown

        capital_1 = np.array(capital_line[1:])
        capital_line1 = np.array(capital_line[:-1])

        capital_return = (capital_1 / capital_line1) - 1
        cum_prod = np.cumprod(capital_return + 1)
        cum_max = np.maximum.accumulate(cum_prod)

        drawdown = cum_prod / cum_max - 1
        drawdown = np.delete(drawdown, 0)
        drawdown = np.round(drawdown, 5)

        # statistics

        num_trade_positive = (trades[trades['pl'] > 0].shape[0])
        num_trade_negative = (trades[trades['pl'] < 0].shape[0])

        largest_win_trade = trades.pl.max()
        largest_loss_trade = trades.pl.min()

        return_accumulated = (capital_line[-1] / capital_line[0]) - 1
        num_trades = num_trade_positive + num_trade_negative

        percent_profit = num_trade_positive / num_trades

        risk_return = return_accumulated / (drawdown.min() * -1)
        result_trade = trades['pl'].apply(lambda x: 1 if x > 0 else 0).sum()
        percent_profitable = result_trade / num_trades
        percent_loss = 1 - percent_profitable
        avg_profit = trades['pl'][trades['pl'] > 0].mean()
        avg_loss = trades['pl'][trades['pl'] < 0].mean()
        profit_factor = avg_profit * percent_profitable / (-1 * avg_loss * percent_loss)

        percent_strategy_exit = 0
        percent_stoploss_exit = 0

        for i in range(0, len(trades)):
            if trades.trade[i] == 'strategy':
                percent_strategy_exit += 1
            else:
                percent_stoploss_exit += 1
        if percent_stoploss_exit == 0:
            percent_strategy_exit = percent_strategy_exit / num_trades

        else:
            percent_strategy_exit = percent_strategy_exit / num_trades
            percent_stoploss_exit = 1 - percent_strategy_exit

        mae_mean = trades.mae.mean()
        mae_max = trades.mae.max()
        mae_min = trades.mae.min()

        trades_consectv = (trades['pl'].apply(lambda x: 1 if x > 0 else -1))
        trades_consectv = [(k, sum(1 for k in g)) for k, g in groupby(trades_consectv)]
        r_trades_aux = []

        for i in trades_consectv:
            r_trades_aux.append(i[0] * i[1])

        max_consectv_wins = max(r_trades_aux)
        max_consectv_loss = min(r_trades_aux) * -1

        print(Fore.CYAN + "\n# ------- basic statistics ------- #")
        print(Fore.WHITE + 'Cumulative return: ', round(return_accumulated, 2))
        print('Risk return', round(risk_return, 2))
        print('Largest win trade: ', round(largest_win_trade, 2))
        print('Largest loss trade: ', round(largest_loss_trade, 2))
        print('Percent profit: ', round(percent_profit, 2))
        print('Percent loss: ', round(percent_loss, 2))
        print('Avg profit: ', round(avg_profit, 2))
        print('Avg loss: ', round(avg_loss, 2))
        print('Profit factor: ', round(profit_factor, 2))
        print('Percent strategy exit', percent_strategy_exit)
        print('Percent stoploss exit', percent_stoploss_exit)
        print('Mae mean: ', mae_mean)
        print('Mae max: ', mae_max)
        print('Mae min: ', mae_min)
        print('Max consectv wins: ', max_consectv_wins)
        print('Max consectv loss: ', max_consectv_loss)
        print('Number of positive trades: ', num_trade_positive)
        print('Number of negative trades: ', num_trade_negative)
        print('Num trades: ', num_trades)
        print(Fore.CYAN + "# --------------------------------- #\n")

        logger.info('Statistics created successfully')

        return drawdown, num_trade_positive, num_trade_negative

    else:

        print(Fore.RED + 'The number of trades is not enough to calculate the statistics and show the graphs')
        logger.info('Error: The number of trades is not enough to calculate the statistics and show the graphs')
        exit()


def graphics():
    """
    :param max_drawdown: Drawdowon from statistics function
    :param po_trades: Positive trades
    :param ne_trades: Negative trades
    :return: Graphics of the strategy, capital line, drawdown, positive and negative trades
    """

    # Graph of strategy
    x = 1
    df_trades, capital_line, data = strategy(data=df_data_fixed, x=x)

    # Candlestick
    inc = data.close > data.open
    dec = data.open > data.close
    w = 30 * 70 * 70 * 50

    source = ColumnDataSource(df_trades)

    p_hover_entry = HoverTool(
        names=["buy_condition"],

        tooltips=[
            ("buy date", "@buy_date{%Y-%m-%d %H hour}"),
            ("buy price", "@buy_price"),
            ("type", "Buy")
        ],

        formatters={
            'buy_date': 'datetime',  # use 'datetime' formatter for 'date' field
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='mouse'
    )
    p_hover_exit = HoverTool(
        names=["sell_condition"],

        tooltips=[
            ("sell date", "@sell_date{%Y-%m-%d %H hour}"),
            ("sell price", "@sell_price"),
            ("type", "@type")
        ],

        formatters={
            'sell_date': 'datetime',
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='mouse'
    )

    crosshair = CrosshairTool(dimensions='both')

    data['stoploss'] = data.close - (data.atr * ATR_COEF)  # ATR

    # Figures
    p = figure(x_axis_type="datetime", plot_height=500, plot_width=1500, title="CL CRUDE OIL with indicators")
    p1 = figure(x_axis_type="datetime", plot_height=350, plot_width=1500, title="Capital Line", x_range=p.x_range)

    p.segment(data.date, data.high, data.date, data.low, color="black")
    p.vbar(data.date[inc], w, data.open[inc], data.close[inc], fill_color="green", line_color="black")
    p.vbar(data.date[dec], w, data.open[dec], data.close[dec], fill_color="red", line_color="black")

    # Tools
    p.add_tools(p_hover_entry, p_hover_exit,  crosshair)
    p1.add_tools(crosshair)

    # Graphics
    p.line(data.date, data.ema, line_color="blue", legend='EMA')
    p.line(data.date, data.wma, line_color="red", legend='WMA')
    p.line(data.date, data.stoploss, line_color="purple", legend='ATR')

    p1.line(df_trades.buy_date, capital_line, line_color="blue", legend='Capital Line')

    # Axis of graphics
    p.xaxis.axis_label = 'TIME'
    p.yaxis.axis_label = 'PRICE (USD)'

    # Buy and Sell condition
    p.circle('buy_date', 'buy_price', fill_color="green", line_color="black", legend='BUY CONDITION',
             size=12, fill_alpha=0.8, source=source, name='buy_condition')

    p.circle('sell_date', 'sell_price', fill_color="red", line_color="black", legend='SELL CONDITION',
             size=12, fill_alpha=0.8, source=source, name='sell_condition')

    p1.left[0].formatter.use_scientific = False
    g = gridplot([[p], [p1]], sizing_mode='scale_width')
    show(g)

    logger.info('Graphics generated correctly')


if __name__ == '__main__':

    print(Fore.YELLOW + '\n# --- BT FORCE BEGINS --- #')
    logger.info('--- BACK TESTING PROCESS BEGINS ---')

    df_data_fixed = resample_data(data=DATA, time=TIME)
    df_trades, capital_line, data = strategy(data=df_data_fixed, x=x)

    # Statistics and graphics
    statistics(trades=df_trades, capital_line=capital_line)
    graphics()

    logger.info('Successful! all functions worked perfectly')
    logger.info('--- BT FORCE ENDS ---')
    print(Fore.YELLOW + '\n# --- BT FORCE ENDS --- #')
