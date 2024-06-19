from datetime import datetime, timedelta
from itertools import cycle
from typing import Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import gridspec
from matplotlib.pyplot import figure
from matplotlib.ticker import Formatter
from numpy.lib.stride_tricks import as_strided
from plotly.subplots import make_subplots
from scipy.stats import linregress

CYCOL = cycle("bgrcmk")
PAPER_COLOR = '#F1F1F1'
RED_GREEN_SCALE = ['rgba(220, 0, 33, 0.15)', 'rgba(220, 220, 220, 0.15)', 'rgba(0, 204, 102, 0.15)']
bt_engine_YELLOW = '#FFDC35'
bt_engine_BLUE = 'mediumblue'
bt_engine_CYCOL = cycle([bt_engine_YELLOW, 'blueviolet'])


class MyDayFormatter(Formatter):
    def __init__(self, dates, fmt="%Y-%m-%d"):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        "Return the label for time x at position pos"
        ind = int(np.round(x))
        if ind >= len(self.dates) or ind < 0:
            return ""

        return self.dates[ind].strftime(self.fmt)


class MySecondFormatter(Formatter):
    def __init__(self, dates, fmt="%Y-%m-%d %H:%M:%S"):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        "Return the label for time x at position pos"
        ind = int(np.round(x))
        if ind >= len(self.dates) or ind < 0:
            return ""

        return self.dates[ind].strftime(self.fmt)


class Plotter:

    def __init__(self, strategy=None):
        self._strategy = strategy

    def plot_table(self, df, show=True):
        
        def format_value(x):
            return f'{x:.3f}' if isinstance(x, float) else x

        # Drop rows where 'value' is a DataFrame
        df = df[~df['value'].apply(lambda x: isinstance(x, pd.DataFrame))]
        df = df.applymap(format_value)
        
        c1 = df.index.to_list()
        c2 = df['value'].tolist()

        fig = go.Figure([go.Table(header=dict(values=['Metric', 'Value']), cells=dict(values=[c1, c2]))])
        fig.update_layout(title_text='stats table ', width=600, height=len(c1) * 25)
        if show:
            fig.show()
        return fig

    def plot_params(self, show=True):
        if self._strategy is not None:
            params = {**self._strategy.config.prepare_data_param, **self._strategy.config.trading_data_param}
        else:
            params = {}
        fig = go.Figure([go.Table(header=dict(values=['Param', 'Value']), cells=dict(values=[list(params.keys()), list(params.values())]))])
        fig.update_layout(title_text='params selected', width=600, height=len(list(params.keys())) * 40)
        if show:
            fig.show()
        return fig

    def plot_trade_dist(self, trade_dist, show=True):
        if show:
            plt.figure(figsize=(10, 5))
            plt.title("Trade Frequency", weight="bold", size=18, color="DarkGrey")
            plt.bar(x=trade_dist.index, height=trade_dist.values, color="Blue", width=0.8)
            plt.axhline(0, linestyle="dashed", color="grey")
            plt.xticks(rotation=75)
            # plt.show() # close
        else:
            plt.close()

    def plot_trade_dist_interactive(self, trade_dist, show=False):
        fig = go.Figure([go.Bar(x=trade_dist.index, y=trade_dist.values, marker_color='grey')])
        fig.update_layout(title_text='Trade Frequency', autosize=True, width=1600, height=400, plot_bgcolor=PAPER_COLOR)
        fig.update_layout(barmode='group', bargap=0.0, bargroupgap=0.0)
        if show:
            fig.show()
        return fig

    def rolling_view(self, x, rolling_mdd_p):

        y = as_strided(x, shape=(x.size - rolling_mdd_p + 1, rolling_mdd_p),
                       strides=(x.strides[0], x.strides[0]))

        return y

    def rolling_max_dd(self, x, rolling_mdd_p, min_periods=1):

        if min_periods < rolling_mdd_p:
            pad = np.empty(rolling_mdd_p - min_periods)
            pad.fill(x[0])
            x = np.concatenate((pad, x))

        y = self.rolling_view(x, rolling_mdd_p)

        running_max_y = np.maximum.accumulate(y, axis=1)
        dd = y - running_max_y

        return dd.min(axis=1)

    def plot_return_series(self, day_return, bm_cum_return=None, portfolio=None, show=True, plot_dd=True):
        if show:
            if plot_dd:
                dd = (day_return["cum_return"] + 1) / (day_return["cum_return"] + 1).cummax() - 1
                mdd = dd.cummin()
                rolling_mdd = self.rolling_max_dd((day_return["cum_return"] + 1), 5, min_periods=1)

            formatter = MyDayFormatter(day_return.index)

            fig = plt.figure(figsize=(12, 8))
            fig.tight_layout()
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            ax1 = fig.add_subplot(gs[0])
            ax1.set_title(f"{self._strategy.name}", weight="bold", size=18, color="DarkGrey")
            ax1.xaxis.set_major_formatter(formatter)
            ax1.set_ylabel("return")
            ax1.plot(day_return["cum_return"])
            # ax[0].plot(np.arange(len(day_return)), day_return['cum_return'])

            if bm_cum_return is not None:
                ax1.plot(bm_cum_return)

            if plot_dd:
                ax2 = fig.add_subplot(gs[1])
                ax2.bar(x=dd.index.values, height=dd.values, color="IndianRed")
                ax2.plot(mdd)
            if show:
                plt.show()

            # if portfolio is not None:
            #     if "position" in portfolio.columns:
            #         fig, ax = plt.subplots(figsize=(12, 8))
            #         ax.set_title("Positions", weight="bold", size=18, color="DarkGrey")
            #         ax.plot(portfolio["position"])
            #     fig.autofmt_xdate()
            #     # plt.show()

            #     if "layers" in portfolio.columns:
            #         fig, ax = plt.subplots(figsize=(12, 8))
            #         ax.set_title("Layers", weight="bold", size=18, color="DarkGrey")
            #         ax.plot(portfolio["layers"])
            #     fig.autofmt_xdate()
            #     # plt.show()
        else:
            pass

    def plot_return_series_interactive(
            self,
            day_return: pd.DataFrame,
            bm_cum_return: pd.Series = None,
            portfolio: pd.DataFrame = None,
            data_df: pd.DataFrame = None,
            plot_fields: list = ['open'],
            pos_anchor_field: str = 'open',
            from_: datetime = None,
            to_: datetime = None,
            pos_field: str = 'position',
            show: bool = True,
            y_log_scale: bool = True,
            live_trade_return=None,
            live_trade_position_series=None
    ) -> Tuple[go.Figure, Union[go.Figure, None]]:

        if data_df is not None:
            assert portfolio is not None, f"`portfolio` cannot be None if `data_df` is specified."

        dd = (day_return["cum_return"] + 1) / (day_return["cum_return"] + 1).cummax() - 1
        mdd = dd.cummin()
        rolling_mdd = self.rolling_max_dd((day_return["cum_return"] + 1), 7, min_periods=1)

        # return fig
        # upper plot
        fig1 = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
        fig1.add_trace(go.Scatter(x=day_return.index, y=day_return['cum_return'] + 1, name='Return', line_color=bt_engine_YELLOW), row=1, col=1)

        if live_trade_return is not None:
            fig1.add_trace(go.Scatter(x=live_trade_return.index, y=live_trade_return['cum_return'] + 1, name='Live Trade Return', line_color=bt_engine_BLUE), row=1, col=1)
        if bm_cum_return is not None:
            fig1.add_trace(go.Scatter(x=bm_cum_return.index, y=bm_cum_return + 1, name='Benchmark', line_color='silver'), row=1, col=1)

        # lower plot
        fig1.add_trace(go.Scatter(x=day_return.index, y=dd.values, name='Drawdown', line_color='#F59282', fill='tozeroy'), row=2, col=1)
        fig1.add_trace(go.Scatter(x=day_return.index, y=mdd.values, name='Max Drawdown', line_color='#EF553B'), row=2, col=1)
        fig1.add_trace(go.Scatter(x=day_return.index, y=rolling_mdd, name='Rolling MDD', line_color='#2196F3'), row=2, col=1)
        fig1.update_layout(title_text='Overview: Equity and Drawdown', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), autosize=True, width=1600, height=800,
                           plot_bgcolor=PAPER_COLOR, hovermode='x unified')
        fig1.update_yaxes(title_text="Drawdown", row=2, col=1)
        scale_type = 'log' if y_log_scale else None
        fig1.layout['yaxis1'].update(type=scale_type)

        # position fig
        fig2 = None
        if data_df is not None:
            fig2 = make_subplots(rows=3, cols=1, row_heights=[0.7, 0.15, 0.15], shared_xaxes=True)
            # make enter and exit points
            plot_df = data_df[from_:to_] if from_ is not None and to_ is not None else data_df
            plot_pos = portfolio[from_:to_] if from_ is not None and to_ is not None else portfolio
            plot_df['l_pos'] = np.where(plot_pos[pos_field] > plot_pos[pos_field].shift(1), plot_df[pos_anchor_field], None)
            plot_df['s_pos'] = np.where(plot_pos[pos_field] < plot_pos[pos_field].shift(1), plot_df[pos_anchor_field], None)

            if plot_fields is None:
                plot_fields = []

            for plot_field in plot_fields:
                fig2.add_trace(go.Scatter(x=plot_df.index, y=plot_df[plot_field], name=plot_field, line_color=bt_engine_YELLOW))
            fig2.add_trace(go.Scatter(x=plot_df.index, y=plot_df['l_pos'], name='l_pos', mode='markers', marker=dict(color='rgb(0, 204, 102)', size=4)), row=1, col=1)
            fig2.add_trace(go.Scatter(x=plot_df.index, y=plot_df['s_pos'], name='s_pos', mode='markers', marker=dict(color='rgb(220, 0, 33)', size=4)), row=1, col=1)
            fig2.layout['yaxis1'].update(type=scale_type)

            fig2.add_trace(go.Scatter(x=portfolio.index, y=portfolio['position'], name='Position', line=dict(color='grey', width=0.7)), row=2, col=1)
            fig2.add_trace(go.Scatter(x=portfolio.index, y=portfolio['layers'], name='Layer', line=dict(color='grey', width=0.7)), row=3, col=1)
            if live_trade_position_series is not None:
                fig2.add_trace(go.Scatter(x=live_trade_position_series.index, y=live_trade_position_series, name='Live Trade Position', line=dict(color='lightblue', width=0.7)),
                               row=3, col=1)
            fig2.update_layout(title_text='Trade Signals, Positions and Layers', autosize=True, width=1600, height=1200, plot_bgcolor=PAPER_COLOR, hovermode='x unified')
            fig2.update_yaxes(title_text="Positions", row=2, col=1)
            fig2.update_yaxes(title_text="Layers", row=3, col=1)

        if show:
            fig1.show()
            if fig2 is not None:
                fig2.show()

        # fig1 upper: returns, lower: mdd
        # fig2 upper: l_pos s_pos (pos entry anchors) and `plot_fields`, middle: position, lower: layers, or None
        return fig1, fig2

    def plot_monthly_return(self, day_return, show=True):
        if show:
            month_ret = day_return['return'].resample("M").sum()
            datetime_index = [i.strftime("%Y%m") for i in month_ret.index]
            plt.figure(figsize=(10, 5))
            plt.title("Strategy Monthly Return", weight="bold", size=18, color="DarkGrey")
            plot_color = np.where(month_ret > 0, "Orange", "IndianRed")
            plt.bar(x=datetime_index, height=month_ret.values, color=plot_color, width=0.8)
            plt.axhline(0, linestyle="dashed", color="grey")
            plt.xticks(rotation=75)
            plt.show()
        else:
            pass

    def plot_weekly_return(self, day_return, show=True):
        if show:
            week_ret = day_return['return'].resample("W").sum()
            datetime_index = [i.strftime("%Y%m%a") for i in week_ret.index]
            plt.figure(figsize=(10, 5))
            plt.title("Strategy Weekly Return", weight="bold", size=18, color="DarkGrey")
            plot_color = np.where(week_ret > 0, "Orange", "IndianRed")
            plt.bar(x=datetime_index, height=week_ret.values, color=plot_color, width=0.8)
            plt.axhline(0, linestyle="dashed", color="grey")
            plt.xticks(rotation=75)
            plt.show()
        else:
            pass

    def plot_monthly_return_interactive(self, day_return, show=True):
        month_ret = day_return['return'].resample("M").sum()
        colors = [bt_engine_YELLOW if v > 0 else 'red' for v in month_ret.values]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=month_ret.index, y=month_ret.values, name='Monthly return', marker_color=colors))
        fig.update_layout(title_text='Strategy Monthly Returns', autosize=True, width=1600, height=400, plot_bgcolor=PAPER_COLOR)
        if show:
            fig.show()
        return fig

    def plot_weekly_return_interactive(self, day_return, show=True):
        week_ret = day_return['return'].resample("W").sum()
        colors = [bt_engine_YELLOW if v > 0 else 'red' for v in week_ret.values]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=week_ret.index, y=week_ret.values, name='Weekly return', marker_color=colors))
        fig.update_layout(title_text='Strategy Weekly Returns', autosize=True, width=1600, height=400, plot_bgcolor=PAPER_COLOR)
        if show:
            fig.show()
        return fig

    def plot_trade_signal(self, data_df, portfolio, plot_fields, pos_anchor_field, from_=None, to_=None, pos_field='position'):
        pd.options.mode.chained_assignment = None
        # train_df[:5000]['l_spread'].plot()
        figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
        ax1 = plt.subplot2grid((6, 4), (1, 0), rowspan=4, colspan=4)

        formatter = MySecondFormatter(data_df[from_:to_].index)

        plot_df = data_df[from_:to_] if from_ is not None and to_ is not None else data_df
        plot_pos = portfolio[from_:to_] if from_ is not None and to_ is not None else portfolio
        plot_df['l_pos'] = None
        plot_df.loc[plot_pos[plot_pos[pos_field] > plot_pos[pos_field].shift(1)].index, 'l_pos'] = plot_df.loc[
            plot_pos[plot_pos[pos_field] > plot_pos[pos_field].shift(1)].index, pos_anchor_field]
        plot_df['s_pos'] = None
        plot_df.loc[plot_pos[plot_pos[pos_field] < plot_pos[pos_field].shift(1)].index, 's_pos'] = plot_df.loc[
            plot_pos[plot_pos[pos_field] < plot_pos[pos_field].shift(1)].index, pos_anchor_field]

        for plot_field in plot_fields:
            ax1.plot(pd.Series(range(0, len(plot_df))), plot_field, data=plot_df, color='skyblue', linewidth=1)

        ax1.plot(pd.Series(range(0, len(plot_df))), 'l_pos', data=plot_df, marker='o', markerfacecolor='blue', linewidth=1)
        ax1.plot(pd.Series(range(0, len(plot_df))), 's_pos', data=plot_df, marker='o', markerfacecolor='red', linewidth=1)

        ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
        ax1.xaxis.set_major_formatter(formatter)
        plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))

        plt.legend(loc=9, ncol=2, prop={'size': 10}, fancybox=True, borderaxespad=0.)

        ax2 = plt.subplot2grid((6, 4), (5, 0), sharex=ax1, rowspan=1, colspan=4)
        ax2.plot(pd.Series(range(0, len(plot_df))), plot_pos[pos_field], linewidth=1.5)
        ax2.axhline(0)

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=True, rotation=70)
        plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)
        plt.show()

    @staticmethod
    def plot_trade_signal_interactive(data_df, portfolio, plot_fields, pos_anchor_field, from_=None, to_=None, pos_field='position', plot_subplot_fields=None, show=True):

        plot_df = data_df[from_:to_] if from_ is not None and to_ is not None else data_df
        plot_pos = portfolio[from_:to_] if from_ is not None and to_ is not None else portfolio
        plot_df['l_pos'] = np.where(plot_pos[pos_field] > plot_pos[pos_field].shift(1), plot_df[pos_anchor_field], None)
        plot_df['s_pos'] = np.where(plot_pos[pos_field] < plot_pos[pos_field].shift(1), plot_df[pos_anchor_field], None)

        # position fig
        if plot_subplot_fields is None:
            fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
        else:
            fig = make_subplots(rows=3, cols=1, row_heights=[0.7, 0.15, 0.15], shared_xaxes=True)
            for plot_subplot_field in plot_subplot_fields:
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[plot_subplot_field], name=plot_subplot_field, line=dict(color='grey', width=0.7)), row=3, col=1)

        for plot_field in plot_fields:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[plot_field], name=plot_field, line_color=next(bt_engine_CYCOL)), row=1, col=1)

        fig.add_trace(go.Scatter(x=plot_pos.index, y=plot_pos['position'], name='Position', line=dict(color='grey', width=0.7)), row=2, col=1)

        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['l_pos'], name='l_pos', mode='markers', marker=dict(color='rgb(0, 204, 102)', size=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['s_pos'], name='s_pos', mode='markers', marker=dict(color='rgb(220, 0, 33)', size=4)), row=1, col=1)
        # fig.layout['yaxis1'].update(type=scale_type)

        fig.update_layout(title_text='Trade Signals', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), autosize=True, width=1600, height=800, plot_bgcolor=PAPER_COLOR)

        if show:
            fig.show()
        return fig

    @staticmethod
    def plot_indicator(data_df, price_field='open', same_plot_fields=None, subplot_fields=None, marker_fields=None, subplot_marker_fields=None, subplot_fields2=None, show=True):
        if subplot_fields is None:
            fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        else:
            if subplot_fields2 is None:
                fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
            else:
                fig = make_subplots(rows=3, cols=1, row_heights=[0.7, 0.15, 0.15], shared_xaxes=True)
                for plot_subplot_field in subplot_fields2:
                    fig.add_trace(go.Scatter(x=data_df.index, y=data_df[plot_subplot_field], name=plot_subplot_field, line=dict(color=next(bt_engine_CYCOL), width=0.7)), row=3,
                                  col=1)

            for plot_subplot_field in subplot_fields:
                fig.add_trace(go.Scatter(x=data_df.index, y=data_df[plot_subplot_field], name=plot_subplot_field, line=dict(color=next(bt_engine_CYCOL), width=0.7)), row=2, col=1)
            if subplot_marker_fields is not None:
                for plot_subplot_field in subplot_marker_fields:
                    fig.add_trace(
                        go.Scatter(x=data_df.index, y=data_df[plot_subplot_field], name=plot_subplot_field, mode='markers', marker=dict(color='rgb(0, 204, 102)', size=4)),
                        row=2, col=1)

        fig.add_trace(go.Scatter(x=data_df.index, y=data_df[price_field], name=price_field, line_color=next(bt_engine_CYCOL)), row=1, col=1)
        if same_plot_fields is not None:
            for plot_field in same_plot_fields:
                fig.add_trace(go.Scatter(x=data_df.index, y=data_df[plot_field], name=plot_field, line_color=next(bt_engine_CYCOL)), row=1, col=1)
        if marker_fields is not None:
            for plot_field in marker_fields:
                fig.add_trace(go.Scatter(x=data_df.index, y=data_df[plot_field], name=plot_field, mode='markers', marker=dict(color='rgb(0, 204, 102)', size=4)), row=1, col=1)

        fig.update_layout(title_text='Trading Indicators', autosize=True, width=1600, height=800, plot_bgcolor=PAPER_COLOR)
        if show:
            fig.show()
        return fig

    @staticmethod
    def scatter_plot(plot_df, x, y, cal_ols=False):
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, data=plot_df, linestyle='none', marker='o')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.axhline(0)
        plt.axvline(0)

        if cal_ols:
            reg = linregress(
                x=plot_df.dropna()[x],
                y=plot_df.dropna()[y],
            )
            plot_df['lin_result'] = reg.slope * plot_df[x] + reg.intercept
            plt.plot(x, 'lin_result', data=plot_df)
            print(f'Slope: {reg.slope}, R: {reg.rvalue}, P: {reg.pvalue}, SD: {reg.stderr}')

        plt.show()

    def pack(self, d1, d2, o1, o2, o3, o4):
        # def pack(self, d1, d2):
        fig = make_subplots(rows=3, cols=2, specs=[[{'type': 'table'}, {'type': 'table'}], [{'type': 'scatter'}, {'type': 'scatter'}], [{'type': 'bar'}, {'type': 'bar'}]])
        for i in d1.data:
            fig.add_trace(i, row=1, col=1)
        for i in d2.data:
            fig.add_trace(i, row=1, col=2)
        for i in o1.data:
            fig.add_trace(i, row=2, col=1)
        for i in o2.data:
            fig.add_trace(i, row=2, col=2)
        for i in o3.data:
            fig.add_trace(i, row=3, col=1)
        for i in o4.data:
            fig.add_trace(i, row=3, col=2)
        fig.show()

    def plot_metrics(self, pf_df, day_return, benchmark, trade_dist, show=True):
        # showing return interactive or not
        if show:
            self.plot_return_series(day_return, benchmark, pf_df, show=show)
            if trade_dist is not None:
                self.plot_trade_dist(trade_dist, show=show)
            self.plot_monthly_return(day_return, show=show)

    def plot_metrics_interactive(self, pf_df, data_df, day_return, benchmark, trade_dist, strategy_summary_df, plot_fields=['open'], pos_anchor_field='open', from_=None,
                                 to_=None, show=True, y_log_scale=True):
        # showing return interactive or not
        d1 = self.plot_params(show=show)
        d2 = self.plot_table(strategy_summary_df, show=show)
        o1, o2 = self.plot_return_series_interactive(day_return, benchmark, pf_df, data_df, plot_fields, pos_anchor_field, from_, to_, show=show, y_log_scale=y_log_scale)
        o3 = None
        if trade_dist is not None:
            o3 = self.plot_trade_dist_interactive(trade_dist, show=show)
        o4 = self.plot_monthly_return_interactive(day_return, show=show)
        o5 = self.plot_weekly_return_interactive(day_return, show=show)
        return d1, d2, o1, o2, o3, o4, o5

    def plot_multi_returns(self, portfolio, construct_cum_ret=False, ignore_col=[]):
        num_ = len(portfolio.columns)
        f, tuples = plt.subplots(int(num_ / 5) + 1, 5, figsize=(17, 21))
        f.tight_layout()
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2, )
        i, j = 0, 0
        for col in portfolio.columns:
            if col in ignore_col:
                continue
            # value = (portfolio[col] + 1).cumprod() if construct_cum_ret else portfolio[col]
            value = portfolio[col].cumsum() if construct_cum_ret else portfolio[col]
            if num_ < 5:
                tuples[j].plot(portfolio.index, value)
                tuples[j].set_title(col)
                tuples[j].tick_params(labelrotation=45)
            else:
                tuples[i][j].plot(portfolio.index, value)
                tuples[i][j].set_title(col)
                tuples[i][j].tick_params(labelrotation=45)
            # ax1.set(xlabel='race', ylabel='pnl in $')
            # slice_step = math.floor(len(portfolio.index) / 10) if len(portfolio.index) > 10 else 1
            # tuples[i][j].set_xticklabels(portfolio.index[::slice_step].strftime("%Y-%m"), rotation=70)

            if j < 4:
                j = j + 1
            else:
                i = i + 1
                j = 0

        plt.show()

    @staticmethod
    def plot_series_interactive(df1, cols, from_=0, to_=10000, graph_name='Series', show=True, diff_y_axis=False):
        fig = go.Figure()
        for col in cols:
            fig.add_trace(go.Scatter(
                x=df1.index[from_:to_],
                # x=np.arange(len(df1[:length])),
                y=df1[col][from_:to_],
                name=col,
                # line_color='deepskyblue',
                opacity=0.8))

        # if not diff_y_axis:
        #     fig.add_trace(go.Scatter(
        #         x=df2.index[:length],
        #         # x=np.arange(len(df2[:length])),
        #         y=df2[:length],
        #         name="beta",
        #         opacity=0.8))
        # else:
        #     fig.add_trace(go.Scatter(
        #         # x=df2.index[:length],
        #         x=np.arange(len(df2[:length])),
        #         y=df2[:length],
        #         yaxis='y2',
        #         name="zscore",
        #         opacity=0.8))
        #
        # # Use date string to set xaxis range
        fig.update_layout(title_text=graph_name)
        #                   xaxis=dict(tickmode='array', tickvals=np.arange(0, len(df1), 500), ticktext=df1[::500].index),
        #                   yaxis=dict(domain=[0.2, 0.8], ),
        #                   yaxis2=dict(domain=[0, 0.2], )
        #                   )
        if show:
            fig.show()
        else:
            return fig
