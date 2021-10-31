import time
import calendar
import pandas as pd
import warnings
from collections.abc import Iterable
import os
import re
import requests
import cgi
import shutil
import datetime
import ntpath
import numpy as np
import math


def get_instrument_data_from_path(path_to_files):
    """Reads instrument details from a path.

    Args:
        path_to_files (:obj:`str`): path to folder or files with instruments details
    Returns:
        instruments (:obj:`List[dict]`): instrument names.
        transactions (:obj:`pandas.DataFrame`): transactions organized in a tabular manner.
    """
    files = get_list_of_files(path_to_files, return_full_path=True)

    input_data = pd.DataFrame()

    for file in files:
        aux_data = pd.read_csv(file)
        ticker = ntpath.basename(file).split(".")[0]
        aux_data.insert(0, 'Ticker', ticker)
        input_data = input_data.append(aux_data, ignore_index=True)
    return input_data


def transform_instrument_data(input_data, event="div"):
    if event == 'div':
        input_data.Date = pd.to_datetime(input_data.Date)
        input_data['Entry_type'] = 'Actuals'
        input_data['Month'] = input_data.Date.dt.month
        input_data['Month_name'] = input_data.Date.dt.month_name()
        input_data['Year'] = input_data.Date.dt.year
        input_data['Period'] = input_data.apply(lambda row: '-'.join([row['Month_name'][:3], str(row['Year'])]), axis=1)
        input_data['Quarter'] = input_data.Date.dt.quarter
        input_data['Week'] = input_data.Date.dt.isocalendar().week
        input_data['Id'] = input_data.apply(lambda row: '-'.join([row['Ticker'], row['Period']]), axis=1)
        input_data.set_index('Id', inplace=True)
        input_data.sort_values(['Ticker', 'Date'], ascending=(True, False), inplace=True)
    return input_data


def create_dividend_calendar(input_data, time_window=5, n_largest=24):
    """

    :param n_largest:
    :param input_data: dataframe
    :param time_window: in years
    :return:
    """
    def filter_by_ticker_and_time(data, row, time_range=None):
        _time_range = (data['Year'].isin(time_range)) if time_range else (data['Year'] == row['Year'])
        return data[(data['Ticker'] == row['Ticker']) & _time_range]

    def format_column_by(column):
        return lambda row: ', '.join([str(x) for x in row[column]])

    data_range = range(input_data['Year'].max() - time_window, input_data['Year'].max())
    group_levels = ['Month', 'Month_name']
    for level in group_levels:
        input_data[f'ex_div_dates_{level}'] = input_data.apply(
            lambda x:
            filter_by_ticker_and_time(
                input_data, x, data_range
            ).groupby(['Ticker', level])[level].count().nlargest(n_largest).unstack().columns.values,
            axis=1)
        input_data[f'ex_div_dates_{level}'] = input_data[f'ex_div_dates_{level}'].apply(np.sort)

    input_data['dividend_freq'] = input_data.apply(
        lambda x:
        filter_by_ticker_and_time(input_data, x, data_range)['Month'].count() / len(data_range),
        axis=1)
    slicer = input_data['ex_div_dates_Month'].apply(np.count_nonzero, axis=0) != input_data['dividend_freq']
    input_data['dividend_freq'][slicer] = input_data[slicer]['ex_div_dates_Month'].apply(np.count_nonzero, axis=0)
    for level in group_levels:
        input_data[f'dividend_calendar_{level}'] = input_data.apply(format_column_by(f'ex_div_dates_{level}'), axis=1)

    columns = ['Ticker', 'dividend_freq', 'dividend_calendar_Month', 'dividend_calendar_Month_name']
    dividend_calendar = input_data[columns].drop_duplicates().set_index('Ticker')

    return dividend_calendar


def forecast_dividend_dates(input_data, dividend_calendar, target=None):
    """

    :param dividend_calendar:
    :param input_data: dataframe
    :param target: target year, i.e. forecast until target year (included)
    :return:
    """
    def filter_by_ticker_and_time(data, row, time_range=None):
        _time_range = (data['Year'].isin(time_range)) if time_range else (data['Year'] == row['Year'])
        return data[(data['Ticker'] == row['Ticker']) & _time_range]

    def create_forecasted_records(records_dict,  installments):
        _new_records = []
        __row = records_dict.copy()
        __row['Date'] = pd.to_datetime(__row['Date'])
        __row['Entry_type'] = 'Forecast'
        for installment in installments:
            _row = __row.copy()
            step = installment * row['installment_step_weeks']
            _row['Date'] = _row['Date'] + pd.DateOffset(weeks=step)
            _row['Week'] = _row['Date'].week
            _row['Month'] = _row['Date'].month
            _row['Month_name'] = _row['Date'].month_name()
            _row['Quarter'] = _row['Date'].quarter
            _row['Year'] = _row['Date'].year
            _row['Period'] = '-'.join([_row['Month_name'][:3], str(_row['Year'])])
            _row['Id'] = '-'.join([_row['Ticker'], _row['Period']])
            _new_records.append(_row)
        return _new_records

    input_data['installments_to_date'] = input_data.apply(
        lambda x:
        filter_by_ticker_and_time(input_data, x)['Period'].nunique(),
        axis=1)
    input_data['pending_installments'] = input_data['dividend_freq'] - input_data['installments_to_date']
    input_data['installment_step'] = 12 / input_data['dividend_freq']
    input_data['installment_step_weeks'] = 52 / input_data['dividend_freq']

    slicer = input_data['pending_installments'] > 0
    aux_pending = input_data[slicer]
    aux_pending['last_installment'] = aux_pending.apply(
        lambda x:
        filter_by_ticker_and_time(aux_pending, x)['Date'].max(),
        axis=1)

    max_year = aux_pending['Year'].max()

    max_year_slicer = aux_pending['Year'] == max_year

    time_range = range(max_year, target + 1) if target else [max_year]
    aux_pending = aux_pending[max_year_slicer].sort_values(['Ticker', 'Date'])
    aux_pending = aux_pending.drop_duplicates(subset=['Ticker', 'Year', 'last_installment'], keep="last")

    new_records = []
    for year in time_range:
        pending_dict = aux_pending.to_dict('records')
        for row in pending_dict:
            pending = row['pending_installments'] if year == max_year \
                else dividend_calendar.loc[row['Ticker']]['dividend_freq']
            pending_installments = range(1, int(pending) + 1)
            records = create_forecasted_records(row, pending_installments)
            new_records.extend(records)
        aux_pending = pd.DataFrame(new_records).set_index('Id', inplace=True)

    aux_installments = pd.DataFrame(new_records)
    aux_installments.set_index('Id', inplace=True)
    aux_installments.to_csv(r"check_forecast.csv")
    input_data = pd.concat([input_data, aux_installments]).sort_values(['Ticker', 'Date'], ascending=[True, False])

    return input_data


def save_to_path(dataframe, destination_path, filename=None, event="div"):
    _filename = filename if filename else f"output_file_{event}_{get_timestamp()}.csv"
    _path = os.path.join(destination_path, _filename)
    dataframe.to_csv(_path)


def get_list_of_files(folder_path, return_full_path=False, extensions=None, pattern=None):
    """
    Retrieves a generator with the file names in a specific folder

     :param str folder_path: full path to files (e.g. "C:\\Users\\GUID\\Downloads\\", "/home/user/Downloads/")
     :param bool return_full_path: if True returns full path, other returns file name
     :param list[str] extensions: list of admitted extensions, e.g. ['txt','pdf']; if None, all extensions are admitted
     :param str pattern: regex pattern of filename
     :return:
     """

    def validate_extensions(ext, list_of_ext):
        try:
            return True if ext in list_of_ext or ext.replace(".", "") in list_of_ext else False
        except TypeError:
            return True if list_of_ext is None else False

    def validate_pattern(text, regex_pattern):
        match = '' if regex_pattern is None else re.search(regex_pattern, text)
        return True if match is not None else False

    try:
        for file in os.listdir(folder_path):
            file_full_path = os.path.join(folder_path, file)
            _, extension = os.path.splitext(file)
            if validate_extensions(extension, extensions) and validate_pattern(file, pattern) \
                    and os.path.isfile(file_full_path):
                yield file_full_path if return_full_path else file
    except Exception as ex:
        print(f"Something went wrong, please check your inputs. Exception raised: {ex}")


def download_instrument_data(tickers, destination_path, event="div", start_period=None, end_period=None):
    """Downloads instrument data from a list of tickers to a specified directory

    :param tickers:
    :param destination_path:
    :param event: "div" for dividend historical data, "history" for price historical data
    :param start_period:
    :param end_period:
    :return:
    """
    urls = [get_download_url(t, events=event, start_period=start_period, end_period=end_period) for t in tickers]
    for url in urls:
        download_from_url(url, destination_path)


def get_download_url(ticker, start_period=None, end_period=None, interval="1d", events="div", include_adj_close="true"):
    """Function that constructs the download url based of different parameters

    :param ticker:
    :param start_period:
    :param end_period:
    :param interval:
    :param events: "div" for dividend historical data, "history" for price historical data
    :param include_adj_close:
    :return:
    """
    _start_period = start_period if start_period else 345427200  # 12-12-1980
    _end_period = end_period if end_period else datetime.date.today().timetuple()

    base_url = r"https://query1.finance.yahoo.com/v7/finance/download/"
    p1 = get_date_in_epochs(_start_period)
    p2 = get_date_in_epochs(_end_period)
    download_url = f"{base_url}{ticker}?period1={p1}&period2={p2}&interval={interval}&events={events}" \
                   f"&includeAdjustedClose={include_adj_close}"
    return download_url


def download_from_url(url, destination_path, header=None):
    """Download file from url to directory

        URL is expected to have a Content-Disposition header telling us what
        filename to use.

        Returns filename of downloaded file.

        Code based from:
        [1] https://stackoverflow.com/questions/34252553/downloading-file-in-python-with-requests
        [2] https://stackoverflow.com/questions/38489386/python-requests-403-forbidden

        """
    if not header:
        user_agents = ['Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5)', 'AppleWebKit/537.36 (KHTML, like Gecko)',
                       'Chrome/50.0.2661.102', 'Safari/537.36']
        headers = {'User-Agent': ' '.join(user_agents)}
    try:
        r = requests.get(url, headers=headers, stream=True)

        ticker = url.rsplit('/', 1)[1].rsplit('?', 1)[0]

        if r.status_code != 200:
            raise ValueError(f'Failed to download: {ticker}')

        params = cgi.parse_header(r.headers.get('Content-Disposition', ''))[-1]

        if 'filename' not in params:
            raise ValueError('Could not find a filename')

        filename = os.path.basename(params['filename'])

        abs_path = os.path.join(destination_path, filename)

        with open(abs_path, 'wb') as target:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, target)
    except Exception as e:
        print("Oops! Something went wrong. ", e)


def set_directory_tree(root_path, data_folders=None, output_folder='outputs', replace=True, return_outputs=True):
    """Creates directory structure required for the ETL

    :param return_outputs:
    :param root_path:
    :param data_folders:
    :param output_folder:
    :param replace:
    :return:
    """

    _data_path = get_timestamp(str_format="%Y-%m-%d")
    _data_dump_folder = os.path.join(root_path, _data_path)
    _output_folder = os.path.join(_data_dump_folder, output_folder)

    _paths = [_data_dump_folder, _output_folder]

    if data_folders:
        for folder in data_folders:
            _paths.append(os.path.join(_data_dump_folder, folder))
    if replace:
        try:
            shutil.rmtree(_data_dump_folder)
        except FileNotFoundError:
            pass

    for directory in _paths:
        if not os.path.exists(directory):
            os.makedirs(directory)
    if return_outputs:
        return _data_dump_folder, _output_folder


def get_timestamp(str_format="%Y-%m-%dT%H%M%S"):
    return time.strftime(str_format)


def get_date_in_epochs(date):
    if type(date) is int:
        return date
    elif type(date) is time.struct_time:
        return calendar.timegm(date)
    else:
        patterns = ["%Y-%m-%d", "%m-%d-%Y", "%m/%d/%Y", "%m/%d/%y"]
        for pattern in patterns:
            try:
                date_struct = time.strptime(date, pattern)
                time_gm = calendar.timegm(date_struct)
                return time_gm
            except ValueError:
                continue
        return None


class TypeCheck:
    """TypeCheck class is a descriptor which verifies if the type of a value is valid.
    If it's not valid, the descriptor will raise error, otherwise it will set the values to the variables.

    Adapted from: https://pythonguide.readthedocs.io/en/latest/python/moreex.html#py-attval-rectgeneralized

    """

    def __init__(self, parameter, expected_type):
        self.parameter = parameter
        self.expected_type = expected_type

    def __get__(self, instance, cls):
        if instance is None:  # required if descriptor is
            return self  # used as class variable
        else:  # in this code, only following line is required
            return instance.__dict__[self.parameter]

    def __set__(self, instance, value):
        # print("setting %s to %s" % (self.parameter, value))
        if not isinstance(value, self.expected_type):
            if value is None or pd.isna(value):
                warnings.warn(f"Expected {str(self.expected_type)} but {type(value)} was provided.", Warning)
            else:
                raise TypeError(f"Expected {str(self.expected_type)} but {type(value)} was provided.")

        expected_types = self.expected_type if isinstance(self.expected_type, Iterable) else [self.expected_type]

        if pd.Timestamp in expected_types:
            instance.__dict__[self.parameter] = pd.Timestamp(value)
        else:
            instance.__dict__[self.parameter] = value


def typeAssert(**kwargs):
    """Decorator typeAssert takes the name of the attribute along with its valid type.
    These ‘key-value’ pairs (‘parameter-expected_type’) are passed to descriptor ‘TypeCheck’.

    Borrowed from: https://pythonguide.readthedocs.io/en/latest/python/moreex.html#py-attval-rectgeneralized
    """

    def decorate(cls):
        for parameter, expected_type in kwargs.items():
            setattr(cls, parameter, TypeCheck(parameter, expected_type))
        return cls

    return decorate
