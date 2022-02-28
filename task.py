import tabula
import sys
sys.path.append('..')
from vars import *
import pandas as pd
import re
from fuzzywuzzy import process, fuzz
import PyPDF2
import datetime
import os


def init_columns(df):
    # df[13] = df[7].combine_first(df[8])
    df = df.rename(columns={
        'Year': 'Year',
        'Application': 'Application',
        'Entry\nshocks': 'Front Entry Shock',
        'Premium\nshocks': 'Front Premium Shock',
        'Springs': 'Front Springs',
        'Protection\nkits/\nMounting\nkits': 'Front Protection kits/ Mounting kits',
        'Mounting information': 'Front Mounting information',
        'Entry\nshocks.1': 'Rear Entry Shock',
        'Premium\nshocks.1': 'Rear Premium Shock',
        'Springs.1': 'Rear Springs',
        'Protection\nkits/\nMounting\nkits.1': 'Rear Protection kits/ Mounting kits',
        'Mounting information.1': 'Rear Mounting information',
    })[COLUMN_HEADINGS]

    return df


def get_make_model_index(makes, models, df):
    """Returns a dictionary with the start index of the makes and models"""
    make_dict = {}
    model_dict = {}
    for i, year_item in enumerate(df.Year):
        if any(make in str(year_item).upper() for make in makes):
            make_dict[i] = year_item
        if any(model in str(year_item).upper() for model in models):
            if '-' not in year_item:
                model_dict[i] = year_item

    return make_dict, model_dict


def get_make_or_model(row, makes_or_models, model=True, check_col=False):
    if '-' not in row['Year']:
        matches = []
        row_make_or_model = row['Year']
        if model:
            row_make_or_model = row_make_or_model.rstrip(' K').rstrip(' H').rstrip(' S').rstrip(' W').rstrip(' F')
        for make_or_model in makes_or_models:
            if make_or_model in row_make_or_model.upper():
                matches.append(make_or_model)
                # print(row_make_or_model, process.extract(row_make_or_model, matches))
        if matches:
            best_match = process.extractOne(row_make_or_model, matches)
            if check_col and best_match[1] < 95:
                return 'Check'
            elif check_col:
                return 'In DB'
            return row_make_or_model
    
    return None


def get_start_or_end_year(row, start_year=True):
    """
    Returns either the starting year or ending. If `start_year` is set to True then the starting
    year is returned, otherwise the ending year is returned.
    """
    if start_year:
        i = 0
    else:
        i = 1
    
    _year = str(row['Year'])
    _year = _year.replace('only', '')

    if '-' in _year:
        try:
            _year = _year.split('-')[i]

        except IndexError:
            if not start_year:
                return row['Start Year']
            else:
                return None

    if 'All' in _year:
        if start_year:
            _year = 1900
        else:
            _year = 'on'

    elif 'on' not in _year:
        _year = re.sub('\D', '', _year)
        try:
            _year = int(_year)
            if _year < 30:
                _year += 2000
            else:
                _year += 1900
        except ValueError:
            print('Value error with year:', row['Year'])

    if _year == '':
        print('Year:', row['Year'])

    if isinstance(_year, int):
        if _year > 2021:
            _year = ''

    elif isinstance(_year, str):
        if _year != 'on':
            _year = ''
        
    return _year


def extract_incl_derivatives(row):
    try:
        deriv = row.Application.replace(row.Model.replace('/', '&'), '').strip()
        init_deriv = deriv.replace('\r', '')
        deriv = init_deriv
        if '(EXCL' in deriv.upper():
            deriv_i = deriv.upper().find('(EXCL')
            deriv = deriv[:deriv_i].strip()
            
        # print('Deriv before', deriv)
        if '(INCL' in init_deriv.upper():
            # print('Incl found in', init_deriv)
            other_includes_i = init_deriv.upper().find('(INCL')
            deriv = deriv[:other_includes_i]
            deriv = [x.strip() for x in re.split(',|&', deriv)]
            # print('Deriv after', deriv)
            other_includes = init_deriv[other_includes_i:].strip()
            # other_includes = other_includes[other_includes_i:].strip()
            other_includes = re.sub('\(incl|\(INCL\.*|\)', '', other_includes)
            other_includes = [x.strip() for x in re.split(',|&|\/', other_includes)]
            # print('Other includes:', other_includes)
            deriv.extend(other_includes)
        else:
            deriv = [x.strip() for x in re.split(',|&|\/', deriv)]

        if isinstance(deriv, list):
            if len(list(filter(None, deriv))) == 0:
                return ''
            else:
                return deriv
        else:
            return ''

    except AttributeError:
        return ''


def extract_excl_derivatives(row):
    try:
        deriv = row.Application.replace(row.Model.replace('/', '&'), '').strip()
        init_deriv = deriv.replace('\r', '')
        deriv = init_deriv
        if '(EXCL' in deriv.upper():
            deriv_i = deriv.upper().find('(EXCL')
            deriv = deriv[deriv_i:].strip()
            deriv = re.sub('\(excl|\(EXCL\.*|\)', '', deriv)
            deriv = [x.strip() for x in re.split(',|&|\/', deriv)]
        else:
            return ''

        return deriv

    except AttributeError:
        return ''


def remove_funnies(row, col):
    value = row[col]
    if isinstance(value, str):
        value = re.sub('\r', ' ', value)
        value = re.sub(r'\“*|\”*', '', value)
        return value
    else:
        return row[col]


def shift_from(row, col, rear_prem_shock=False):
    if rear_prem_shock:
        return ''
    if not row['Rear Mounting information']:
        return row[col]


def extract_with_regex(row, col, search_pattern, sub_pattern):
    value = row[col]
    if value and isinstance(value, str):
        if re.findall(search_pattern, value):
            value = re.sub(r'\“*|\”*', '', value)
            return re.sub(sub_pattern, '', value)

    return ''


def prepare_final_dataframe(df, MODELS):
    # make_dict, model_dict = get_make_model_index(MAKES, MODELS, df)
    df = init_columns(df)
    df.Year.fillna(method='ffill', inplace=True)
    df['Make'] = df.apply(lambda row: get_make_or_model(row, MAKES, model=False), axis=1)
    df['Model'] = df.apply(lambda row: get_make_or_model(row, MODELS, model=True), axis=1)
    df['Check Model'] = df.apply(lambda row: get_make_or_model(row, MODELS, model=True, check_col=True), axis=1)
    df.Model.fillna(method='ffill', inplace=True)
    df.Make.fillna(method='ffill', inplace=True)
    df['Check Model'].fillna(method='ffill', inplace=True)
    df = df[df.Application.notna()]
    df = df.assign(Year=df['Year'].str.split('/')).explode('Year')
    df['Start Year'] = df.apply(lambda row: get_start_or_end_year(row), axis=1)
    df['End Year'] = df.apply(lambda row: get_start_or_end_year(row, start_year=False), axis=1)
    df = df.drop(df[df['Start Year'] == ''].index)
    df = df.drop(df[df['End Year'] == ''].index)
    df['Include Derivatives'] = df.apply(lambda row: extract_incl_derivatives(row), axis=1)
    df['Exclude Derivatives'] = df.apply(lambda row: extract_excl_derivatives(row), axis=1)
    for col in PART_COLS:
        df[col] = df.apply(lambda row: remove_funnies(row, col), axis=1)
    m = df['Rear Mounting information'].isna()
    df.loc[m, ['Rear Premium Shock', 'Rear Springs', 'Rear Protection kits/ Mounting kits', 'Rear Mounting information']] = (
        df.loc[m, ['Rear Mounting information', 'Rear Premium Shock', 'Rear Springs', 'Rear Protection kits/ Mounting kits']].values)
    df['Rear Protection kits'] = df.apply(lambda row: extract_with_regex(row, 'Rear Protection kits/ Mounting kits', r'PK\d*', r'\/\s*MK\d*\s*'), axis=1)
    df['Rear Mounting kits'] = df.apply(lambda row: extract_with_regex(row, 'Rear Protection kits/ Mounting kits', r'MK\d*', r'\“*PK\d*\”*\s*\/*\s*'), axis=1)
    df['Front Protection kits'] = df.apply(lambda row: extract_with_regex(row, 'Front Protection kits/ Mounting kits', r'PK\d*', r'\/\s*MK\d*\s*'), axis=1)
    df['Front Mounting kits'] = df.apply(lambda row: extract_with_regex(row, 'Front Protection kits/ Mounting kits', r'MK\d*', r'\“*PK\d*\”*\s*\/*\s*'), axis=1)
    df = df.assign(Model=df['Model'].str.split('/')).explode('Model')
    fdf = df[FINAL_COLS]

    return fdf


def match_cols(x, row):

    if row['End Year'] == 'on':
        end_year = datetime.datetime.now().year
    else:
        end_year = row['End Year']

    try:
        if isinstance(end_year, str):
            end_year = int(end_year)
    except ValueError:
        print('End Year:', end_year, ', Type:', type(end_year))
        row.to_csv('value_err.csv', mode='a', header=False, sep='|')
        print('ValueError while matching columns')
        return False

    fill = False

    if x.Make == row.Make and\
        x.Model == row.Model and\
            x['Model Year'] > row['Start Year'] and\
                x['Model Year'] < end_year:
                fill = True
                incl_derivs = row['Include Derivatives']
                excl_derivs = row['Exclude Derivatives']
                if incl_derivs:
                    # print('Include Derivs:', incl_derivs)
                    # print( x['Derivative'])
                    if not any(fuzz.WRatio(deriv.upper(), x['Derivative']) >= 90 for deriv in incl_derivs):
                        fill = False
                if excl_derivs:
                    # print('Exclude Derivs', excl_derivs)
                    # print( x['Derivative'])
                    if any(fuzz.WRatio(deriv.upper(), x['Derivative']) >= 90 for deriv in excl_derivs):
                        fill = False
    
    if fill:
        return True
    
    return False


def derivative_match(x, row, include=True):
    if include:
        in_or_ex = 'Include'
    else:
        in_or_ex = 'Exclude'
    key = f'{in_or_ex} Derivatives'
    incl_derivs = row[key]
    for deriv in incl_derivs:
        # print(x.keys())
        if fuzz.WRatio(deriv, x['Derivative']) >= 90:
            return deriv
    return ''


def extract_dataframes():
    parent = '/Users/omrijacobsz/otomatika/voomer/vmr1-monroe-data-mapping/notebooks/processed'
    files = os.listdir(parent)
    files.remove('.DS_Store')
    dfs = []
    for file in files:
        xls = pd.ExcelFile(os.path.join(parent, file), engine='openpyxl')
        for sheet in range(10):
            dfs.append(pd.read_excel(
                xls,
                'Sheet{}'.format(sheet+1),
                skiprows=3,
                skipfooter=4
            ))
    return dfs


def remove_extra_columns(dfs):
    fdfs = []
    for df in dfs:
        if df.shape[1] > 12:
            df['Mounting information.1'] = df['Mounting information.1'].combine_first(df['Unnamed: 12'])
            df = df.drop(columns='Unnamed: 12')
        fdfs.append(df)

    return fdfs


def main():
    # pages = list(range(116))[4:116]
    # tables = tabula.read_pdf(MONROE_CATALOGUE_PATH, pages=pages, area=[[121.6, 10.0, 728.0, 575.0]], pandas_options={'header':None})

    # dfs = []

    # for table in tables:
    #     if table.shape[1] == 13:
    #         dfs.append(table)
    print('Reading XLXSes')
    dfs = extract_dataframes()
    print('Remove extra columns')
    dfs = remove_extra_columns(dfs)
    
    print('List of dataframes created')
    print('Rading Mapping System')
    ms_df = pd.read_excel(MAPPING_SYSTEM_PATH, sheet_name='Vehicle Part Mapping System', header=1)
    print('Mapping System in memory')

    MAKES = ms_df['Make'].unique()
    MODELS = ms_df['Model'].unique()

    num_dfs = len(dfs)

    for i, df in enumerate(dfs):
        print(f'Processing DF {i}')
        fdf = prepare_final_dataframe(df, MODELS)
        for index, row in fdf.iterrows():
            indexes = ms_df[ms_df.apply(lambda x: match_cols(x, row), axis=1)].index
            
            ms_df.loc[indexes, 'Include Derivative'] = ms_df.loc[indexes].apply(lambda x: derivative_match(x, row, include=True), axis=1)
            ms_df.loc[indexes, 'Exclude Derivative'] = ms_df.loc[indexes].apply(lambda x: derivative_match(x, row, include=False), axis=1)

            for key in COLUMN_MAPPING.keys():
                ms_df.loc[indexes, key] = row[COLUMN_MAPPING[key]]
        percentage = round(i / num_dfs * 100)
        print(f'Completed DF {i} / {num_dfs}: {percentage}% complete')

    print('Writing final output file')
    ms_df.to_excel('output.xlsx')
    print('Final output file complete')


if __name__ == '__main__':
    print('Starting Process')
    main()
    print('Process Done!')



