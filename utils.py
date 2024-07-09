import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from samplics.estimation import TaylorEstimator
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from math import ceil
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)


def get_vars():
    """
    This function scrapes the NHANES website for the variable names and descriptions
    :return: Pandas DataFrame with the variable names and descriptions
    """
    start_time = time.time()

    url = 'https://wwwn.cdc.gov/nchs/nhanes/search/variablelist.aspx?Component=Laboratory&Cycle='
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')
    tbody = table.find('tbody')

    data = []

    rows = tbody.find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        row_data = [cell.get_text(strip=True) for cell in cells]
        data.append(row_data)

    df = pd.DataFrame(data, columns=['Variable Name', 'Variable Description', 'Data File Name', 'Data File Description',
                                     'Begin Year', 'EndYear',
                                     'Component', 'Use Constraints'])

    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Update complete in: {elapsed_time:.1f} seconds")

    return df


def filter_vars(df_all):
    """
    This function filters the variables to only include the desired variables
    :param df:
    :return: Pandas DataFrame with the filtered variables
    """
    # Filter All Variables
    df_all = df_all[
        ~df_all['Variable Description'].str.contains('sequence number|weight|comment|comt|code', case=False)]

    # Change Variable Types
    df_all['EndYear'] = df_all['EndYear'].astype(int)
    df_all['Begin Year'] = df_all['Begin Year'].astype(int)

    # Only get Variables that start with LBX or URX
    df_all = df_all[df_all['Variable Name'].str.startswith(('LBX', 'URX'))]

    # Find the largest 'Year End' value for each unique 'Variable Name'
    max_years = df_all.groupby('Variable Name')['EndYear'].max()

    # Merge the original DataFrame with the 'max_years' to filter rows
    filtered_df = df_all.merge(max_years, on=['Variable Name', 'EndYear'])

    # Merge on Varaible Name to assign most recent variable description to all past variables.
    df_all = df_all.merge(filtered_df[['Variable Name', 'Variable Description']], on='Variable Name', how='left',
                          suffixes=(' Old', ' New'))

    # Add a Column with Variable Names and Descriptions
    df_all['Full Name'] = df_all['Variable Name'] + ': ' + df_all['Variable Description New']

    # Remove Variables with Use Constraints
    df_all = df_all[df_all['Use Constraints'] == 'None']

    return df_all


def add_yr_demo(df_all):
    """
    This function adds the 'Year' and 'Demo File Name' columns to the DataFrame
    :param df_all:
    :return: Pandas DataFrame with the added columns
    """
    # Add a column with the year range
    df_all['Year'] = df_all.apply(calculate_year, axis=1)

    # Add a column with the demo file name
    df_all['Demo File Name'] = df_all.apply(calculate_demo_file_name, axis=1)

    return df_all


# Custom function to calculate the "Year" column value
def calculate_year(row):
    if row['Data File Name'].startswith('P_'):
        return '2017-2018'
    else:
        return f"{row['Begin Year']}-{row['EndYear']}"


# Custom function to calculate the "Demo File Name" column value
def calculate_demo_file_name(row):
    if row['Data File Name'].startswith('P_'):
        return 'P_DEMO'
    else:
        year_range = row['Year'].split('-')
        start_year = int(year_range[0])
        end_year = int(year_range[1])
        if start_year < 2001:
            return 'DEMO'
        else:
            return f"DEMO_{chr(ord('B') + (start_year - 2001) // 2)}"


def add_columns(df_all):
    """
    This function adds the 'Year' and 'Demo File Name' columns to the DataFrame
    :param df_all:
    :return: Pandas DataFrame with the added columns
    """
    # Add a column with the year range
    df_all['Year'] = df_all.apply(calculate_year, axis=1)

    # Add a column with the demo file name
    df_all['Demo File Name'] = df_all.apply(calculate_demo_file_name, axis=1)

    return df_all


# Helper Functions

# Get list of years variable was measured and not constrained
def create_variable_df(var, df_all):
    df = df_all[df_all['Variable Name'] == var].sort_values(by='EndYear')
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


# Choose the appropriate Weights for the analysis
def choose_weights(data_file, data_df, demo_df):
    weights = ''

    # Count data_df columns that start with 'WT'
    list_data_df_weights = [col for col in data_df.columns if col.startswith('WT')]

    if len(list_data_df_weights) == 0:
        # Handle Pandemic
        if data_file.startswith('P_'):
            weights = 'WTMECPRP'
        else:
            weights = 'WTMEC2YR'
    else:
        weights = list_data_df_weights[0]

    return weights


# Get single year data
def get_single_yr_df(var, year, data_file, demo_file):
    data_url = 'https://wwwn.cdc.gov/Nchs/Nhanes/' + year + '/' + data_file + '.XPT'
    demo_url = 'https://wwwn.cdc.gov/Nchs/Nhanes/' + year + '/' + demo_file + '.XPT'
    data_df = pd.read_sas(data_url)
    demo_df = pd.read_sas(demo_url)
    weights = choose_weights(data_file, data_df, demo_df)
    merged_df = data_df.merge(demo_df, on='SEQN', how='inner')
    merged_df['Weights'] = weights

    # Handle Pandemic
    if data_file.startswith('P_'):
        merged_df['Year'] = '2017-2020 Pre-Pandemic'
    else:
        merged_df['Year'] = year

    # Subset the merged df
    merged_df = merged_df[
        ['SEQN', var, 'Year', 'Weights', weights, 'RIAGENDR', 'RIDAGEYR', 'RIDRETH1', 'SDMVSTRA', 'SDMVPSU']]

    return merged_df


def get_multi_yr_df(var_df, container):
    # For each year/data file combination, get the data and stack by year.
    i = 0
    var = var_df['Variable Name'][i]
    for i in range(len(var_df)):
        var = var_df['Variable Name'][i]
        year = var_df['Year'][i]
        data_file = var_df['Data File Name'][i]
        demo_file = var_df['Demo File Name'][i]

        update = f'{var} {year} {data_file} {demo_file}'
        container.text(update)
        print(update)
        if i == 0:
            multi_yr_df = get_single_yr_df(var, year, data_file, demo_file)
            container.empty()
        else:
            single_yr_df = get_single_yr_df(var, year, data_file, demo_file)
            multi_yr_df = pd.concat([multi_yr_df, single_yr_df])
            container.empty()
    print('Download complete')
    container.empty()

    return multi_yr_df


def get_means(df_all, variable, mean_type, domain, max_value, min_value, purpose=None):
    domain = get_domain(domain)
    df_all = recode_df_domains(df_all)

    # checks for case where user entered a min/max value and then removed it
    if max_value == '':
        max_value = None

    if min_value == '':
        min_value = None

    if max_value is None:
        pass
    else:
        max_value = float(max_value)

    if min_value is None:
        pass
    else:
        min_value = float(min_value)

    unique_pairs = df_all[['Year', 'Weights']].drop_duplicates().values.tolist()

    for i in range(len(unique_pairs)):
        df_part = df_all[(df_all['Year'] == unique_pairs[i][0]) & (df_all['Weights'] == unique_pairs[i][1])]
        weight = df_part['Weights'][0]
        df_part.reset_index(inplace=True)

        if mean_type == 'Geometric':
            if i == 0:
                mean_part = get_geomean(df_part, variable, weight, domain, max_value, min_value, purpose)
                df_means = mean_part
            else:
                mean_part2 = get_geomean(df_part, variable, weight, domain, max_value, min_value, purpose)
                df_means = pd.concat([df_means, mean_part2], ignore_index=True)

        else:
            if i == 0:
                mean_part = get_amean(df_part, variable, weight, domain, max_value, min_value, purpose)
                df_means = mean_part
            else:
                mean_part2 = get_amean(df_part, variable, weight, domain, max_value, min_value, purpose)
                df_means = pd.concat([df_means, mean_part2], ignore_index=True)

    df_means = sort_means(df_means)

    return df_means


def handle_max_min(unweighted_df, variable, max_value, min_value):
    if max_value is not None and min_value is None:
        unweighted_df = unweighted_df[unweighted_df[variable] <= max_value]

    if min_value is not None and max_value is None:
        unweighted_df = unweighted_df[unweighted_df[variable] >= min_value]

    if max_value is not None and min_value is not None:
        unweighted_df = unweighted_df[(unweighted_df[variable] >= min_value) & (unweighted_df[variable] <= max_value)]

    unweighted_df.reset_index(inplace=True)

    return unweighted_df


def get_amean(unweighted_df, variable, weight, domain=None, max_value=None, min_value=None, purpose=None):
    """Computes mean and 95% confidence intervals for single survey period

    Params:
    unweighted_df - unweighted_df for single survey period
    variable - varaible of interest as string (e.g. 'LBXCOT')
    weight - specified sample weights to use (e.g. 'WTMEC2YR')
    domain - optional param for specifiying result split by subgroups

    Returns:
    df of means and 95% confidence intervals for specified arguments
    """

    if max_value is not None or min_value is not None:
        unweighted_df = handle_max_min(unweighted_df, variable, max_value, min_value)

    var_prop = TaylorEstimator("mean")

    if domain == None:
        var_prop.estimate(y=unweighted_df[variable],
                          samp_weight=unweighted_df[weight],
                          stratum=unweighted_df["SDMVSTRA"],
                          psu=unweighted_df["SDMVPSU"],
                          remove_nan=True)
    else:
        var_prop.estimate(y=unweighted_df[variable],
                          samp_weight=unweighted_df[variable],
                          stratum=unweighted_df["SDMVSTRA"],
                          psu=unweighted_df["SDMVPSU"],
                          domain=unweighted_df[domain],
                          remove_nan=True)

    df = var_prop.to_dataframe()

    if purpose != 'line graph':
        df = get_percentiles(df, unweighted_df, variable, weight, domain=domain)

    if domain == None:
        df['Sample Size'] = sum(~np.isnan(unweighted_df[variable]))
        min_val = min(unweighted_df[variable][~np.isnan(unweighted_df[variable])])
        df['Approx. LOD'] = min_val  #* np.sqrt(2)
        weighted_prop_ = sum(unweighted_df[weight][unweighted_df[variable] > min_val]) / sum(
            unweighted_df[weight][~np.isnan(unweighted_df[variable])])
        df['Weighted Proportion > LOD'] = weighted_prop_

    else:
        n_container = []
        prop_container = []
        for d in df['_domain']:
            min_val_domain = min(unweighted_df[unweighted_df[domain] == d][variable][
                                     ~np.isnan(unweighted_df[unweighted_df[domain] == d][variable])])
            n_container.append(sum(~np.isnan(unweighted_df[unweighted_df[domain] == d][variable])))
            prop_container.append(
                sum(unweighted_df[unweighted_df[domain] == d][weight][
                        unweighted_df[unweighted_df[domain] == d][variable] > min_val_domain]) /  # numerator calc
                sum(unweighted_df[unweighted_df[domain] == d][weight][
                        ~np.isnan(unweighted_df[unweighted_df[domain] == d][variable])])  # denominator calc
            )

        df['Sample Size'] = n_container
        df['Weighted Proportion > LOD'] = prop_container
    return format_means(df, unweighted_df, domain, 'Arithmetic', purpose)


def get_geomean(unweighted_df, variable, weight, domain=None, max_value=None, min_value=None, purpose=None):
    """Computes geomean and 95% confidence intervals for single survey period

    Params:
    unweighted_df - unweighted_df for single survey period
    variable - varaible of interest as string (e.g. 'LBXCOT')
    weight - specified sample weights to use (e.g. 'WTMEC2YR')
    domain - optional param for specifiying result split by subgroups

    Returns:
    df of geomeans and 95% confidence intervals for specified arguments
    """

    if max_value is not None or min_value is not None:
        unweighted_df = handle_max_min(unweighted_df, variable, max_value, min_value)

    var_prop = TaylorEstimator("mean")

    if domain == None:
        var_prop.estimate(y=np.log(unweighted_df[variable]),
                          samp_weight=unweighted_df[weight],
                          stratum=unweighted_df["SDMVSTRA"],
                          psu=unweighted_df["SDMVPSU"],
                          remove_nan=True)

    else:
        var_prop.estimate(y=np.log(unweighted_df[variable]),
                          samp_weight=unweighted_df[weight],
                          stratum=unweighted_df["SDMVSTRA"],
                          psu=unweighted_df["SDMVPSU"],
                          domain=unweighted_df[domain],
                          remove_nan=True)

    df = var_prop.to_dataframe()
    df['_estimate'] = np.e ** df['_estimate']
    df['_lci'] = np.e ** df['_lci']
    df['_uci'] = np.e ** df['_uci']

    if purpose != 'line graph':
        df = get_percentiles(df, unweighted_df, variable, weight, domain=domain)

    # determine the number of observations for the table row entry
    if domain == None:
        df['Sample Size'] = sum(~np.isnan(unweighted_df[variable]))
        min_val = min(unweighted_df[variable][~np.isnan(unweighted_df[variable])])
        df['Approx. LOD'] = min_val * np.sqrt(2)
        weighted_prop_ = sum(unweighted_df[weight][unweighted_df[variable] > min_val]) / sum(
            unweighted_df[weight][~np.isnan(unweighted_df[variable])])
        df['Weighted Proportion > LOD'] = weighted_prop_

    else:
        n_container = []
        prop_container = []
        for d in df['_domain']:
            min_val_domain = min(unweighted_df[unweighted_df[domain] == d][variable][
                                     ~np.isnan(unweighted_df[unweighted_df[domain] == d][variable])])
            n_container.append(sum(~np.isnan(unweighted_df[unweighted_df[domain] == d][variable])))
            prop_container.append(
                sum(unweighted_df[unweighted_df[domain] == d][weight][
                        unweighted_df[unweighted_df[domain] == d][variable] > min_val_domain]) /  # numerator calc
                sum(unweighted_df[unweighted_df[domain] == d][weight][
                        ~np.isnan(unweighted_df[unweighted_df[domain] == d][variable])])  # denominator calc
            )

        df['Sample Size'] = n_container
        df['Weighted Proportion > LOD'] = prop_container

    return format_means(df, unweighted_df, domain, 'Geometric', purpose)


def get_percentiles(df, unweighted_df, variable, weight, domain=None):

    biomarker_stratified_proportion = TaylorEstimator("proportion")

    if domain == None:
        biomarker_stratified_proportion.estimate(
            y=unweighted_df[variable],
            samp_weight=unweighted_df[weight],
            stratum=unweighted_df["SDMVSTRA"],
            psu=unweighted_df["SDMVPSU"],
            remove_nan=True)

        df_props = biomarker_stratified_proportion.to_dataframe()
        df_props['cumulative'] = df_props['_estimate'].cumsum()

        quantiles = [0.95, 0.90, 0.75, 0.50]
        pcntl_ests = []

        for quantile in quantiles:
            for i in range(len(df_props)):
                if df_props['cumulative'][i] >= quantile:
                    pcntl_ests.append(df_props['_level'][i])
                    break

        df['50th Percentile'] = pcntl_ests[3].round(3)
        df['75th Percentile'] = pcntl_ests[2].round(3)
        df['90th Percentile'] = pcntl_ests[1].round(3)
        df['95th Percentile'] = pcntl_ests[0].round(3)

    else:
        biomarker_stratified_proportion.estimate(
            y=unweighted_df[variable],
            samp_weight=unweighted_df[weight],
            stratum=unweighted_df["SDMVSTRA"],
            psu=unweighted_df["SDMVPSU"],
            domain=unweighted_df[domain],
            remove_nan=True)

        df_props = biomarker_stratified_proportion.to_dataframe()

        pctl_ests_list = []
        for d in df_props['_domain'].unique():
            grp = df_props.copy()
            grp = grp[grp['_domain'] == d]
            grp['cumulative'] = grp['_estimate'].cumsum()
            grp.reset_index(inplace=True)

            quantiles = [0.95, 0.90, 0.75, 0.50]
            pcntl_ests = []

            for quantile in quantiles:
                for i in range(len(grp)):
                    if grp['cumulative'][i] >= quantile:
                        pcntl_ests.append(grp['_level'][i])
                        break

            pctl_ests_list.append(pcntl_ests)

        percentiles = {50: [], 75: [], 90: [], 95: []}
        for d in range(len(df_props['_domain'].unique())):
            percentiles[50].append(pctl_ests_list[d][3])
            percentiles[75].append(pctl_ests_list[d][2])
            percentiles[90].append(pctl_ests_list[d][1])
            percentiles[95].append(pctl_ests_list[d][0])

        df['50th Percentile'] = percentiles[50]
        df['75th Percentile'] = percentiles[75]
        df['90th Percentile'] = percentiles[90]
        df['95th Percentile'] = percentiles[95]

    return df


def format_means(df, unweighted_df, domain, mean, purpose=None):
    weighted_thresh = 0.6

    if domain == None:
        df.rename(columns={"_estimate": f"{mean} Mean", "_lci": "lower_95%CI", "_uci": "upper_95%CI"}, inplace=True)
        df['Weights'] = unweighted_df['Weights'][0]
        df['Year'] = unweighted_df['Year'][0]

        df['Category'] = 'Total Population'
        if purpose == 'line graph':
            df = df[['Category', 'Year', f'{mean} Mean', 'lower_95%CI', 'upper_95%CI', 'Weights', 'Sample Size', 'Weighted Proportion > LOD']]
        else:
            df = df[['Category', 'Year', f'{mean} Mean', 'lower_95%CI', 'upper_95%CI', '50th Percentile', '75th Percentile', '90th Percentile', '95th Percentile', 'Weights', 'Sample Size', 'Weighted Proportion > LOD']]
        df = df.copy()
        df['Category'] = 'Total Population'

        if purpose != 'line graph':
            for i in range(len(df)):
                if df.loc[i, 'Weighted Proportion > LOD'] >= weighted_thresh:
                    df.loc[i, f'{mean} Mean'] = f"{round(df.loc[i, f'{mean} Mean'], 3)} ({round(df.loc[i, 'lower_95%CI'], 3)} - {round(df.loc[i, 'upper_95%CI'], 3)})"
                else:
                    df.loc[i, f'{mean} Mean'] = '*'
                    df.loc[i, 'lower_95%CI'] = '*'
                    df.loc[i, 'upper_95%CI'] = '*'

        if purpose == 'line graph':
            df = df[['Category', 'Year', f'{mean} Mean', 'lower_95%CI', 'upper_95%CI', 'Weights', 'Sample Size']]
        else:
            df = df[['Category', 'Year', f'{mean} Mean', '50th Percentile', '75th Percentile', '90th Percentile', '95th Percentile', 'Weights', 'Sample Size']]

        return df

    else:
        df.rename(columns={"_estimate": f"{mean} Mean", "_lci": "lower_95%CI", "_uci": "upper_95%CI", "_domain": "Category"},
                  inplace=True)
        df['Weights'] = unweighted_df['Weights'][0]
        df['Year'] = unweighted_df['Year'][0]
        if purpose == 'line graph':
            df = df[['Category', 'Year', f'{mean} Mean', 'lower_95%CI', 'upper_95%CI', 'Weights', 'Sample Size', 'Weighted Proportion > LOD']]
        else:
            df = df[['Category', 'Year', f'{mean} Mean', 'lower_95%CI', 'upper_95%CI', '50th Percentile', '75th Percentile', '90th Percentile', '95th Percentile', 'Weights', 'Sample Size', 'Weighted Proportion > LOD']]

        df.loc[:, f'{mean} Mean'] = df[f'{mean} Mean'].round(3)
        df.loc[:, 'lower_95%CI'] = df['lower_95%CI'].round(3)
        df.loc[:, 'upper_95%CI'] = df['upper_95%CI'].round(3)

        if purpose != 'line graph':
            for i in range(len(df)):
                if df.loc[i, 'Weighted Proportion > LOD'] >= weighted_thresh:
                    df.loc[i, f'{mean} Mean'] = f"{round(df.loc[i, f'{mean} Mean'], 3)} ({round(df.loc[i, 'lower_95%CI'], 3)} - {round(df.loc[i, 'upper_95%CI'], 3)})"
                else:
                    df.loc[i, f'{mean} Mean'] = '*'
                    df.loc[i, 'lower_95%CI'] = '*'
                    df.loc[i, 'upper_95%CI'] = '*'

        if purpose == 'line graph':
            df = df[['Category', 'Year', f'{mean} Mean', 'lower_95%CI', 'upper_95%CI', 'Weights', 'Sample Size']]
        else:
            df = df[['Category', 'Year', f'{mean} Mean', '50th Percentile', '75th Percentile', '90th Percentile', '95th Percentile', 'Weights', 'Sample Size']]

        return df


def get_domain(domain_name):
    if domain_name == 'Race/Hispanic origin':
        domain = 'RIDRETH1'
    elif domain_name == 'Gender':
        domain = 'RIAGENDR'
    elif domain_name == 'Age':
        domain = 1
    elif domain_name == 'Total Population':
        domain = None
    elif domain_name == 'Youth Age Group':
        domain = 'Youth Age Group'
    else:
        print('Error: No domain selected')
    return domain


def recode_df_domains(df):
    # Map number to Race
    race_mapping = {
        1.0: 'Mexican American',
        2.0: 'Other Hispanic',
        3.0: 'Non-Hispanic White',
        4.0: 'Non-Hispanic Black',
        5.0: 'Other Race - Including Multi-Racial'}
    df['RIDRETH1'] = df['RIDRETH1'].replace(race_mapping)

    # Map number to Gender
    gender_mapping = {
        1.0: 'Male',
        2.0: 'Female'}
    df['RIAGENDR'] = df['RIAGENDR'].replace(gender_mapping)

    # Create Youth Age Groups
    conditions = [
        df['RIDAGEYR'].apply(lambda x: 3 <= x < 6),
        df['RIDAGEYR'].apply(lambda x: 6 <= x < 12),
        df['RIDAGEYR'].apply(lambda x: 12 <= x < 20)]

    choices = [
        'a. Age 03-05 years',
        'b. Age 06-11 years',
        'c. Age 12-19 years']

    df['Youth Age Group'] = np.select(conditions, choices, default='d. Age 20+ years')

    return df


def sort_means(df_means):
    df_means = df_means.sort_values(by=['Category', 'Year'])
    #     df_means = df_means.sort_values(by=[('', 'Category'), ('Survey','Years')])
    df_means.reset_index(drop=True, inplace=True)

    return df_means


## Functions for comparing histograms ##

def get_weighted_df(unweighted_df, variable, weight, domain=None):
    """Computes % population at each unique value of variable for single survey period

    Params:
    unweighted_df - unweighted_df for single survey period
    variable - varaible of interest as string (e.g. 'LBXCOT')
    weight - specified sample weights to use (e.g. 'WTMEC2YR')
    domain - optional param for specifiying result split by subgroups

    Returns:
    df of % of population at each unique value of selected variable
    """
    var_prop = TaylorEstimator("proportion")
    
    if domain == None:
        var_prop.estimate(y=unweighted_df[variable],
                          samp_weight=unweighted_df[weight],
                          stratum=unweighted_df["SDMVSTRA"],
                          psu=unweighted_df["SDMVPSU"],
                          remove_nan=True)
    else:
        var_prop.estimate(y=unweighted_df[variable],
                          samp_weight=unweighted_df[weight],
                          stratum=unweighted_df["SDMVSTRA"],
                          psu=unweighted_df["SDMVPSU"],
                          domain=unweighted_df[domain],
                          remove_nan=True)

    df = var_prop.to_dataframe()
    df.reset_index(inplace=True)

    return df


def rescale_x(a):
    axis_vals = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
    min_x = min(axis_vals, key=lambda x: abs(x - np.min(a)))
    max_x = min(axis_vals, key=lambda x: abs(x - np.max(a)))
    scale = [x for x in axis_vals if x >= min_x]
    scale = [x for x in scale if x <= max_x]
    return scale


def plot_domain_dist(df, variable, easy_name, year, weight, domain, bins, log, limit):
    bins = int(bins)
    limit = float(limit)

    # checking weights to get available for that year, default to weight param if present
    weights = list(set(df[df['Year'] == year]['Weights']))

    selected_weight = ''
    if weight in weights:
        selected_weight = weight
    elif len(weights) == 1:
        selected_weight = weights[0]
    else:
        # we can make this more dynamic if we deem it worthwhile, but this will prevent errors
        selected_weight = weights[0]
        df = df[df['Weights'] == selected_weight]
        

    w_df = get_weighted_df(df, variable, selected_weight, domain)

    num_plots = len(w_df['_domain'].unique())
    rows = ceil(num_plots / 2)
    cols = 2
    domains = w_df['_domain'].unique()
    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True, figsize=(10, 10 / 2 * rows))
    colors = ['#5cc1ba', '#fd6906', '#3cb8e3', '#fde702', '#fb9fd1']

    i = 0
    row = 0
    col = 0
    for group in w_df['_domain'].unique():
        df_p = w_df[w_df['_domain'] == group]
        a = df_p['_level'].to_list()
        b = df_p['_estimate'].to_list()  # list of percentages (y vals)

        if limit != 0:
            a = [x for x in a if x >= limit]

        if rows > 1:
            if log == True:
                axes[row, col].hist(df_p['_level'],
                                    weights=df_p['_estimate'] * 100,
                                    bins=np.logspace(np.log10(np.nanmin(a)), np.log10(np.nanmax(a)), bins + 1),
                                    edgecolor='black',
                                    color=colors[i])

                scale = rescale_x(a)
                # Rescale x axis, set tick locations, and set x tick labels
                axes[row, col].set_xscale("log")
                axes[row, col].set_xticks(scale)
                axes[row, col].set_xticklabels(scale)

            else:
                axes[row, col].hist(df_p['_level'],
                                    weights=df_p['_estimate'] * 100,
                                    bins=np.linspace(np.nanmin(a), np.nanmax(a), bins + 1),
                                    edgecolor='black',
                                    color=colors[i])

            # Create Labels
            x_label = easy_name

            # Create legend
            handles = [Rectangle((0, 0), 1, 1, color=colors[i])]

            labels = [domains[i] + ' ' + x_label]
            axes[row, col].legend(handles, labels, edgecolor='black')

            # Create Title and x/y labels
            axes[row, col].set_title(
                'Frequency Distribution of\n' + domains[i] + ' ' + str(x_label) + '\n— NHANES ' + str(year) + '\nSelected Weights: ' + str(selected_weight))
            axes[row, col].set_xlabel(x_label)
            axes[row, col].set_ylabel('Percent of Population')
            axes[row, col].tick_params('x', labelbottom=True)
            axes[row, col].tick_params('y', labelleft=True)

            i += 1
            col = 1
            if i % 2 == 0:
                col = 0
                row += 1

            # don't display empty ax
            if i >= num_plots:
                axes[row, col].remove()
        else:
            if log == True:
                axes[col].hist(df_p['_level'],
                               weights=df_p['_estimate'] * 100,
                               bins=np.logspace(np.log10(np.nanmin(a)), np.log10(np.nanmax(a)), bins + 1),
                               edgecolor='black',
                               color=colors[i])
                scale = rescale_x(a)

                # Rescale x axis, set tick locations, and set x tick labels
                axes[col].set_xscale("log")
                axes[col].set_xticks(scale)
                axes[col].set_xticklabels(scale)


            else:
                axes[col].hist(df_p['_level'],
                               weights=df_p['_estimate'] * 100,
                               bins=np.linspace(np.nanmin(a), np.nanmax(a), bins + 1),
                               edgecolor='black',
                               color=colors[i])

            # Create Labels
            x_label = easy_name

            # Create legend
            handles = [Rectangle((0, 0), 1, 1, color=colors[i])]

            labels = [domains[i] + ' ' + x_label]
            axes[col].legend(handles, labels, edgecolor='black')

            # Create Title and x/y labels
            axes[col].set_title(
                'Frequency Distribution of\n' + str(domains[i]) + ' ' + str(x_label) + '\n— NHANES ' + str(year) + '\nSelected Weights: ' + str(selected_weight))
            axes[col].set_xlabel(x_label)
            axes[col].set_ylabel('Percent of Population')
            axes[col].tick_params('x', labelbottom=True)
            axes[col].tick_params('y', labelleft=True)

            i += 1
            col = 1
            if i % 2 == 0:
                col = 0
                row += 1

            # don't display empty ax
            if i > num_plots:
                axes[col].remove()

    plt.tight_layout()
    st.pyplot(fig)

def plot_total_dist(df, variable, easy_name, year, weight, domain, bins, log, limit):
    bins = int(bins)
    limit = float(limit)

    # checking weights to get available for that year, default to WTMEC2YR if present
    weights = list(set(df[df['Year'] == year]['Weights']))

    selected_weight = ''
    if weight in weights:
        selected_weight = weight
    elif len(weights) == 1:
        selected_weight = weights[0]
    else:
        # we can make this more dynamic if we deem it worthwhile, but this will prevent errors
        selected_weight = weights[0]
        df = df[df['Weights'] == selected_weight]
        
        

    w_df = get_weighted_df(df, variable, selected_weight, domain)
    
    a = w_df['_level'].to_list()
    fig, ax = plt.subplots()

    if log == True:
        ax.hist(w_df['_level'],
                weights=w_df['_estimate'] * 100,
                bins=np.logspace(np.log10(np.nanmin(a)), np.log10(np.nanmax(a)), bins + 1),
                edgecolor='black')
        scale = rescale_x(a)

        # Rescale x axis, set tick locations, and set x tick labels
        ax.set_xscale("log")
        ax.set_xticks(scale)
        ax.set_xticklabels(scale)

    else:
        ax.hist(w_df['_level'],
                        weights=w_df['_estimate'] * 100,
                        bins=np.linspace(np.nanmin(a), np.nanmax(a), bins + 1),
                        edgecolor='black')
    
    ax.set_title(
        'Frequency Distribution of\n' + 'Total Population' + ' ' + str(easy_name) + '\n— NHANES ' + str(year) + ', Selected Weights: ' + str(selected_weight))
    ax.set_xlabel(easy_name)
    ax.set_ylabel('Percent of Population')
    
    plt.tight_layout()
    st.pyplot(fig)


def compare_frequency(df_all,
                      easy_name,
                      Year='2001-2002',
                      Variable='Cotinine, Serum (ng/mL)',
                      Domain='Gender',
                      Bins=40,
                      Log=False,
                      Lower_Limit=0.0,
                      Upper_Limit=0.0):
    df_all = recode_df_domains(df_all)
    df = df_all[df_all['Year'] == Year]
    Weights = 'WTMEC2YR'

    if Lower_Limit is None or Lower_Limit == '0' or Lower_Limit == '0.0' or Lower_Limit == '':
        Lower_Limit = 0.0
        Lower_Limit = float(Lower_Limit)

    if Upper_Limit is None or Upper_Limit == '0' or Upper_Limit == '0.0' or Upper_Limit == '':
        Upper_Limit = 0.0
        Upper_Limit = float(Upper_Limit)

    domain = get_domain(Domain)

    if Domain == 'Total Population':
        plot_total_dist(df, Variable, easy_name, Year, Weights, domain, Bins, Log, Lower_Limit)
        
    else:
        try:
            if Upper_Limit == 0.0:
                plot_domain_dist(df, Variable, easy_name, Year, Weights, domain, Bins, Log, Lower_Limit)
            else:
                st.write('Sorry: "Max" value is not currently supported. Please try another example.')
        except:
            st.write('Sorry: your request could not be completed. Please try another example.')
