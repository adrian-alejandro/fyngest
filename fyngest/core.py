import json
import os.path

from fyngest.helpers import download_instrument_data, get_instrument_data_from_path, transform_instrument_data
from fyngest.helpers import set_directory_tree, save_to_path, create_dividend_calendar, forecast_dividend_dates
from pathlib import Path

_path = Path(__file__).parent.parent

inputs = 'tests/files/sample_inputs.json'#'inputs.json'

path = os.path.join(_path, inputs)

with open(path, ) as f:
    inputs = json.load(f)
    instruments = ['div', 'history']
    data_path, output_path = set_directory_tree(inputs['path'], data_folders=instruments)
    for instrument in instruments:
        _data_path = os.path.join(data_path, instrument)
        download_instrument_data(inputs['tickers'], _data_path, event=instrument)#, start_period='2020-07-01')
        instrument_data = get_instrument_data_from_path(_data_path)
        instrument_data = transform_instrument_data(instrument_data, event=instrument)
        if instrument == 'div':
            dividend_calendar = create_dividend_calendar(instrument_data)
            save_to_path(dividend_calendar, output_path, event='dividend_calendar')
            instrument_data = forecast_dividend_dates(instrument_data, dividend_calendar, 2022)
        save_to_path(instrument_data, output_path, event=instrument)

