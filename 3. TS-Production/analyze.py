import os
import pickle
import argparse
import warnings
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMAResults


def print_progress_bar(iteration: int, total: int, progress: str = '') -> None:
    percent = f"{100 * (iteration / float(total)):3.2f}"
    length = 100 - len(progress) - len(percent) - 15
    filled_length = int(length * iteration // total)
    bar = '=' * filled_length + ('>' if filled_length + 1 <= length else '') + ' ' * (length - filled_length - 1)
    print(f'\r{progress}: |{bar}| {percent}% Complete', end='\n' if iteration == total else '')


def predict_sarimax(**kwargs) -> dict:
    with open('utils/last_exogs.pkl', 'rb') as f:
        exogs = pickle.load(f)
    exogs = pd.DataFrame.from_dict(exogs)
    exogs = exogs[['InTransit', 'Export', 'Import', 'Transit']]
    models = {}
    for col in cols:
        models[col] = ARIMAResults.load(f'utils/models/sarimax_{col}.pkl')
    start = models['Export'].fittedvalues.index.max()
    exogs.index = [start]

    def iteract(start):
        start = start + pd.DateOffset(months=1)
        tmp = {}
        for col in cols:
            tmp[col] = models[col].predict(start, exog=exogs.loc[:, exogs.columns != col])[0]
        return start, tmp

    if 'next' in kwargs.keys():
        for _ in range(args.next):
            start, tmp = iteract(start)
            exogs.loc[start] = tmp
        return exogs.iloc[1:].to_dict('series')
    elif 'start' in kwargs.keys() and 'end' in kwargs.keys():
        while str(start)[:7] != kwargs['end']:
            start, tmp = iteract(start)
            exogs.loc[start] = tmp
        return exogs.loc[kwargs['start']:kwargs['end']].to_dict('series')


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
    description='Анализ и прогнозирование объёмов экспорта перевозок древесины',
    formatter_class=argparse.RawTextHelpFormatter,
    epilog='Для корректной работы CSV таблицы должны иметь следующие поля: \n'
           ' * Month - дата в формате YYYY-MM; \n'
           ' * InTransit - объёмы внутренних перевозок; \n' 
           ' * Export - объёмы экспорта. \n'
           ' * Import - объёмы импорта; \n'
           ' * Transit - объёмы транзита; \n\n'
           'Внимание: Порядок колонок должен быть таким же, как и в списке.'
)
parser.add_argument('-r', '--retrain', type=str, metavar='table',
                    help='Запуск в режиме обучения / повторного обучения. Принимает название \n'
                         'файла-таблицы в формате CSV или XSLX (Excel)')

parser.add_argument('-p', '--predict', type=str, metavar='range',
                    help='Запуск в режиме прогнозирования. Принимает диапазон дат в формате \n'
                         'формате: ГГГГ-ММ:ГГГГ-ММ. Внимание: Прогнозирование на несколько \n'
                         'месяцев вперёд менее точно, чем прогнозирование на следующий месяц')

parser.add_argument('-n', '--next', type=int, metavar='amount',
                    help='Делает прогноз в следующих n месяцах. Принимает число - количество\n'
                         'месяцев')

parser.add_argument('-x', '--sarimax', action='store_true',
                    help='Флаг, указывающий запуск прогнозирования с использованием модели \n'
                         'SARIMAX. Модель SARIMAX ищет завивимости между временными рядами. \n' 
                         'Прогнозирование с помощью SARIMAX даёт более точную оценку для \n' 
                         'показателей объёма экспорт и внутренних перевозок, но худшую для \n' 
                         'транзита и импорта (по крайней мере на тренировочных данных). Без \n' 
                         'указания этого флага прогноз будет построен моделью SARIMA')

parser.add_argument('-a', '--append', action='store_true',
                    help='Используйте этот флаг, если необходимо сохранить данные в файле с \n'
                         'исходными данными. Этот флаг необходимо указывать совместно с \n'
                         'флагами -r / --retrain и -n / --next')

parser.add_argument('-o', '--out', type=str, metavar='output_table',
                    help='Если необходимо сохранить файл в формате Excel, введите ключ -o c \n' 
                         'указанием имени файла (можно без расширения .xslx). Если ключ не \n' 
                         'указан, вывод проноза будет в терминале')
args = parser.parse_args()


os.makedirs('utils/models', exist_ok=True)

if args.retrain:
    if args.retrain.endswith('.csv'):
        df = pd.read_csv(args.retrain, index_col='Month', parse_dates=['Month'])
    elif args.retrain.endswith('.xlsx'):
        df = pd.read_excel(args.retrain, index_col='Month', parse_dates=['Month'])
    else:
        print('Указаный формат таблицы с исходными данными неверен!')
        exit(1)
    df = df[['InTransit', 'Export', 'Import', 'Transit']]

    for col in df.columns:
        df[f'{col}_S'] = df[col].shift(1)

    df = df.loc['2009':]
    df = df.dropna()
    df = df.astype(int)
    df = df.asfreq('MS')

    sarima_models = {
        'Export': [(2, 0, 3), (1, 1, 1, 12)],
        'InTransit': [(2, 1, 2), (0, 1, 1, 12)],
        'Import': [(1, 1, 0), (0, 1, 2, 12)],
        'Transit': [(0, 1, 1), (1, 1, 2, 12)],
    }

    sarimax_models = {
        'Export': sarima_models['Export'],
        'InTransit': sarima_models['InTransit'],
        'Import': [(0, 1, 0), (0, 1, 2, 12)],
        'Transit': [(0, 1, 1), (0, 1, 2, 12)],
    }

    i = 0
    for col, [o, so] in sarima_models.items():
        model = sm.tsa.arima.ARIMA(df[col], order=o, seasonal_order=so).fit()
        model.save(f'utils/models/sarima_{col}.pkl')
        i += 1
        print_progress_bar(i, 8, 'Прогресс обучения')

    exogs = df.loc[:, ['InTransit_S', 'Export_S', 'Import_S', 'Transit_S']]
    for col, [o, so] in sarimax_models.items():
        exog = exogs.loc[:, exogs.columns != f'{col}_S']
        model = sm.tsa.arima.ARIMA(df[col], order=o, seasonal_order=so, exog=exog).fit()
        model.save(f'utils/models/sarimax_{col}.pkl')
        i += 1
        print_progress_bar(i, 8, 'Прогресс обучения')

    last_exogs = {}
    for key in sarima_models.keys():
        last_exogs[key] = [df.iloc[-1][key]]
    with open('utils/last_exogs.pkl', 'wb') as f:
        pickle.dump(last_exogs, f)


predictions = {}
cols = ['InTransit', 'Export', 'Import', 'Transit']

if args.predict:
    start, end = args.predict.split(':')
    print(f'Прогнозирование в диапазоне от {start} до {end}.')
    try:
        if args.sarimax:
            predictions = predict_sarimax(start=start, end=end)
        else:
            for col in cols:
                model = ARIMAResults.load(f'utils/models/sarima_{col}.pkl')
                predictions[col] = model.predict(start, end)
    except FileNotFoundError:
        print('Перед прогнозированием нужно создать модель. Запустите программу с флагом -r/--retrain и '
              'укажите путь до таблицы с исходными данными в формате CSV или XLSX')
        exit(1)

if args.next:
    print(f'Прогнозирование на ближайшие месяцы: {args.next}.')
    try:
        if args.sarimax:
            predictions = predict_sarimax(next=args.next)
        else:
            for col in cols:
                model = ARIMAResults.load(f'utils/models/sarima_{col}.pkl')
                start = model.fittedvalues.index.max() + pd.DateOffset(months=1)
                predictions[col] = model.predict(start, start + pd.DateOffset(months=args.next - 1))
    except FileNotFoundError:
        print('Перед прогнозированием нужно создать модель. Запустите программу с флагом -r/--retrain и '
              'укажите путь до таблицы с исходными данными в форматe CSV или XLSX')
        exit(1)

if args.next or args.predict:
    predictions = pd.DataFrame.from_dict(predictions)
    predictions.columns = ['Внутренние перевозки', 'Экспорт', 'Импорт', 'Транзит']
    predictions.index = list(map(lambda x: str(x)[:7], predictions.index))

    if args.out:
        args.out = args.out if args.out.endswith('.xlsx') else f'{args.out}.xlsx'
        print(f'Сохрание прогноза в файле {args.out}')
        predictions.astype(int).to_excel(args.out, index_label='Месяц')
    elif args.append:
        if not args.retrain:
            print('Флаг -a / --append необходимо указывать совместно с флагом -r / --retrain')
            exit(1)
        if args.retrain.endswith('.csv'):
            tmp = pd.read_csv(args.retrain, index_col='Month')
        elif args.retrain.endswith('.xlsx'):
            tmp = pd.read_excel(args.retrain, index_col='Month')
        predictions.columns = ['InTransit', 'Export', 'Import', 'Transit']
        predictions = tmp.append(predictions.astype(int))
        if args.retrain.endswith('csv'):
            predictions.to_csv(args.retrain, index_label='Month')
        else:
            predictions.to_excel(args.retrain, index_label='Month')
        print(f'Сохрание прогноза в файле {args.retrain}')
    else:
        for col in predictions.columns:
            predictions[col] = predictions[col].apply(lambda x: f'{x:12,.0f}'.replace(',', '\''))
        print(predictions)

if not args.retrain and not args.predict and not args.next:
    parser.print_help()
