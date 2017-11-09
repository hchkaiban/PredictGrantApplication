import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime


def get_tables(filename='data/unimelb_training.csv'):
    df_raw = pd.read_csv(filename)
    # delete the last column which has only Nan
    del df_raw[df_raw.columns[-1]]
    # extract the first 26 columns that are the non-repeated columns to each application
    df_grants = df_raw[df_raw.columns[:26]]
    # Create an array with 15 entries containing each a dataframe with 41 columns
    researcher_columns = [df_raw[list(df_raw.columns[0:26]) + list(df_raw.columns[26 + 15 * i: 26 + 15 * (i + 1)])] for i in range(int(len(df_raw.columns[26:]) / 15))]
    # put the same column name to each entries in the researcher columns
    for table in researcher_columns:
        table.columns = [list(df_raw.columns[0:26]) + list(researcher_columns[0].columns[26:])]
    # Concatenate the array of each entries in the researcher_columns into one big dataframe
    researchers = pd.concat(researcher_columns)
    # drop duplicate - repeated missing candidates per team
    unique_researchers = researchers.drop_duplicates()
    # return the dataframe sorted by "application ID"
    return unique_researchers.sort_values('Grant.Application.ID')


def combine_columns(dfOrig, codeName='SEO.Code.', prcName='SEO.Percentage.', codeRange=5, index='Grant.Application.ID'):
    """ Goes through all codeNames + codeRange, , impute '99' to blanks, get_dummies on them, drops colums with code 0 and
    add up each throughout the range."""
    # get a copy of the original dataframe
    df = dfOrig.copy()
    # create a dataframe cleanDF that will get five columns for the SEO code
    cleanDf = df[['{}{}'.format(codeName, i) for i in range(1, codeRange+1)]].fillna(990000) // 10000
    cleanDf[index] = df[index]
    dummyDf = []
    for i in range(1, codeRange + 1):
        # add to cleanDf the columns SEO percentage
        cleanDf['{}{}'.format(prcName, i)] = df['{}{}'.format(prcName, i)]
        # create 'currDummy' dataframe that will have the SEO columns from cleanDF
        currDummy = cleanDf[[index] + ['{}{}'.format(codeName, i)]]
        # create dummies from the SEO columns
        currDummy = pd.get_dummies(currDummy['{}{}'.format(codeName, i)], prefix=codeName)
        # add the 'Grant.Application.ID' column to currDummy
        currDummy[index] = cleanDf[index]
        # add the percentage column
        currDummy['{}{}'.format(prcName, i)] = cleanDf['{}{}'.format(prcName, i)]
        # groupby the 'Grant.Application.ID'
        currDummy = currDummy.groupby(index)[currDummy.columns].max()
        # Create a new DF "currDUmmy2" and multiply each individual cell by its percentage
        currDummy2 = currDummy.apply(lambda x: x[:-2] * x[-1], axis=1)
        currDummy2[index] = currDummy[index]
        # Append currDummy2 to a list - after the for loop ends, there will be five entries in this list
        dummyDf.append(currDummy2)
    currDummy = dummyDf[0]
    # Concatenate each entry in the list into a big dataframe
    for i in range(1, codeRange):
        currDummy = currDummy.add(dummyDf[i], fill_value=0.)
        currDummy[index] = dummyDf[i][index]
        currDummy.fillna(0, inplace=True)
    currDummy.set_index('Grant.Application.ID', inplace=True)
    return currDummy


def munge_data(df_orig):
    df = df_orig.copy()

    # Remove the Person-ID this information is useless
    del df['Person.ID.1']

    # Create oldest DF where applications are grouped and only year of birth column is kept with its min value for each team
    oldest = pd.DataFrame(df.groupby('Grant.Application.ID')['Year.of.Birth.1'].min())

    # Create a numRole DF and get the number of researchers for each role. Groupby application ID
    numRole = pd.get_dummies(df['Role.1'])
    numRole['Grant.Application.ID'] = df['Grant.Application.ID']
    numRole = numRole.groupby('Grant.Application.ID')[numRole.columns].sum()

    # Create a numAussies DF and get the % of aussies per application ID
    numAussies = pd.get_dummies(df['Country.of.Birth.1'])
    numAussies.set_index(df['Grant.Application.ID'], inplace=True)
    numAussies = numAussies.groupby('Grant.Application.ID')[numAussies.columns].sum()

    # Create a prcAussies DF + imput all values with NaN (no country info) to zero
    prcAussies = pd.DataFrame((numAussies['Australia'] / numAussies.sum(axis=1)).fillna(0), columns=['% Australians'])

    # Create a numPapers DF with the sum of the # of published papers per team
    numPapers = df.groupby('Grant.Application.ID')['A..1', 'A.1', 'B.1', 'C.1', 'Number.of.Successful.Grant.1', 'Number.of.Unsuccessful.Grant.1'].sum()

    # Replace all the missing values in the 'Contract.Value.Band...see.note.A' by 'A' (the mode value)
    df['Contract.Value.Band...see.note.A'].fillna('A', inplace=True)
    # Remove all the white space in the 'Contract.Value.Band...see.note.A' column and replace the letter by their ASCII numerical code
    df['Contract.Value.Band...see.note.A'] = df['Contract.Value.Band...see.note.A'].apply(lambda x: ord(x.rstrip(' ')))

    # Create a 'grant_cats' DF converting categories to dummy variables
    grant_cats = pd.get_dummies(df['Grant.Category.Code'], dummy_na=True)
    grant_cats['Grant.Application.ID'] = df['Grant.Application.ID']
    grant_cats = grant_cats.groupby('Grant.Application.ID')[grant_cats.columns].min()
    grant_cats = pd.DataFrame(grant_cats)

    # imputing missing percentages for RFCD.Percentage columns with the mean
    df['RFCD.Percentage.1'].fillna(df['RFCD.Percentage.1'].mean(), inplace=True)
    df['RFCD.Percentage.2'].fillna(df['RFCD.Percentage.2'].mean(), inplace=True)
    df['RFCD.Percentage.3'].fillna(df['RFCD.Percentage.3'].mean(), inplace=True)
    df['RFCD.Percentage.4'].fillna(df['RFCD.Percentage.4'].mean(), inplace=True)
    df['RFCD.Percentage.5'].fillna(df['RFCD.Percentage.5'].mean(), inplace=True)

    # doing the same as above with SEO.Percentage columns
    df['SEO.Percentage.1'].fillna(df['SEO.Percentage.1'].mean(), inplace=True)
    df['SEO.Percentage.2'].fillna(df['SEO.Percentage.2'].mean(), inplace=True)
    df['SEO.Percentage.3'].fillna(df['SEO.Percentage.3'].mean(), inplace=True)
    df['SEO.Percentage.4'].fillna(df['SEO.Percentage.4'].mean(), inplace=True)
    df['SEO.Percentage.5'].fillna(df['SEO.Percentage.5'].mean(), inplace=True)

    rfcds = combine_columns(df, 'RFCD.Code.', 'RFCD.Percentage.')
    seos = combine_columns(df, 'SEO.Code.', 'SEO.Percentage.')

    # Get rid of everything we don't need
    df.drop(['A..1', u'A.1', u'B.1', u'C.1', u'Country.of.Birth.1', u'Dept.No..1', u'Faculty.No..1', u'Home.Language.1', u'No..of.Years.in.Uni.at.Time.of.Grant.1', u'Number.of.Successful.Grant.1', u'Number.of.Unsuccessful.Grant.1', u'Role.1', u'Sponsor.Code', u'With.PHD.1', u'Year.of.Birth.1', u'SEO.Code.4', u'SEO.Code.5', u'SEO.Code.1', u'SEO.Code.2', u'SEO.Code.3', u'RFCD.Code.1', u'RFCD.Code.2', u'RFCD.Code.3', u'RFCD.Code.4', u'RFCD.Code.5', 'Grant.Category.Code', u'RFCD.Percentage.1', u'RFCD.Percentage.2', u'RFCD.Percentage.3', u'RFCD.Percentage.4', u'RFCD.Percentage.5', u'SEO.Percentage.1', u'SEO.Percentage.2', u'SEO.Percentage.3', u'SEO.Percentage.4', u'SEO.Percentage.5'], inplace=True, axis=1)

    df.drop_duplicates(inplace=True)

    # set the index to the 'Grant.Application.ID' - very important that all DFs have the same index in order to merge
    df.set_index('Grant.Application.ID', inplace=True)

    # Merge all the DF created
    finalDf = pd.merge(df, oldest, left_index=True, right_index=True)
    finalDf = pd.merge(finalDf, numRole, left_index=True, right_index=True)
    finalDf = pd.merge(finalDf, prcAussies, left_index=True, right_index=True)
    finalDf = pd.merge(finalDf, numPapers, left_index=True, right_index=True)
    finalDf = pd.merge(finalDf, grant_cats, left_index=True, right_index=True)
    finalDf = pd.merge(finalDf, rfcds, left_index=True, right_index=True)
    finalDf = pd.merge(finalDf, seos, left_index=True, right_index=True)

    # imputing ages with median
    finalDf['Year.of.Birth.1'] = finalDf['Year.of.Birth.1'].fillna(finalDf['Year.of.Birth.1'].median())

    # imputing missing papers with 0
    finalDf['A..1'] = finalDf['A..1'].fillna(0)
    finalDf['A.1'] = finalDf['A.1'].fillna(0)
    finalDf['B.1'] = finalDf['B.1'].fillna(0)
    finalDf['C.1'] = finalDf['C.1'].fillna(0)

    # imputing missing successful and unsuccessful grants with 0
    finalDf['Number.of.Successful.Grant.1'] = finalDf['Number.of.Successful.Grant.1'].fillna(0)
    finalDf['Number.of.Unsuccessful.Grant.1'] = finalDf['Number.of.Unsuccessful.Grant.1'].fillna(0)

    # Convert the date column into a usable date format
    # datetime.datetime.strptime(date_string, format) returns a datetime according to a format
    # .timetuple() generates a tuple from the strptime with all the time information
    # time.mktime generates a single time value from the tuple

    del finalDf['Grant.Application.ID_y']
    del finalDf['Grant.Application.ID_x']

    finalDf['Proc.Start.Date'] = finalDf['Start.date'].apply(lambda x: time.mktime(datetime.datetime.strptime(x, '%d/%m/%y').timetuple()))

    return finalDf